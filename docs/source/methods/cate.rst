CatE: order embeddings of the ALC concept lattice
=================================================

:class:`~mowl.models.cate.model.CatE` reimplements CatE of Zhapa-Camacho and Hoehndorf,
*Lattice-based ALC ontology embeddings with saturation* (arXiv:2305.07163v3; an earlier
version without saturation appeared at NeSy 2024). The catalogue key is ``cate``. CatE does
not map concepts to regions: it builds a preorder of concept descriptions and embeds that
preorder into :math:`\mathbb{R}^n` with the coordinatewise order.

The lattice
-----------

:class:`~mowl.projection.categorical.lattice.CatELattice` collects the concept descriptions
of the training ontology and all their sub-descriptions, following the category-theoretical
semantics of ALC [duc2021]_. Each constructor and axiom adds a fixed set of morphisms
(paper, Section 3):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Description / axiom
     - Morphisms
   * - :math:`A \sqcap B`
     - :math:`A \sqcap B \sqsubseteq A`, :math:`A \sqcap B \sqsubseteq B`
   * - :math:`A \sqcup B`
     - :math:`A \sqsubseteq A \sqcup B`, :math:`B \sqsubseteq A \sqcup B`
   * - :math:`\neg C`
     - :math:`C \sqcap \neg C \sqsubseteq \bot`, :math:`\top \sqsubseteq C \sqcup \neg C`, with
       the projections and injections of Figure 2c
   * - :math:`\exists r.C`
     - role element :math:`r_{\exists r.C} \sqsubseteq r`;
       :math:`\mathrm{codomain}(r_{\exists r.C}) \sqsubseteq C`;
       :math:`\mathrm{domain}(r_{\exists r.C}) \equiv \exists r.C`
   * - :math:`\forall r.C`
     - :math:`\forall r.C \equiv \neg \exists r. \neg C`
   * - :math:`C \sqsubseteq D`
     - :math:`C \sqsubseteq D` and :math:`\top \sqsubseteq \neg C \sqcup D`
   * - :math:`C(a)`, :math:`r(a, b)`
     - :math:`\{a\} \sqsubseteq C`; :math:`\{a\} \sqsubseteq \exists r.\{b\}`
   * - every element :math:`X`
     - :math:`\bot \sqsubseteq X \sqsubseteq \top`

The construction is done by the existing
:class:`~mowl.projection.categorical.model.CategoricalProjector` (the projector of the
reference code). Where it differs from the paper's text (complex negations, role-dependent
rules, the string names of nodes) is listed on the card; ``complete_negation=True`` adds the
missing morphisms of Figure 2c.

Saturation
----------

:mod:`mowl.projection.categorical.saturation` implements the rules of Section 3.1:
(1) :math:`C \sqsubseteq \neg D \Rightarrow D \sqsubseteq \neg C`,
(2) :math:`C \sqcap D \sqsubseteq \bot \Rightarrow C \sqsubseteq \neg D`,
(3) :math:`\top \sqsubseteq C \sqcup D \Rightarrow \neg C \sqsubseteq D` (printed with
:math:`\bot` in the premise, a typo), together called S1, and (4) the transitive closure
(Tr). ``saturation_steps`` sets the number of S1 iterations (``None``: until a fixed point,
which always exists), ``transitive_closure=True`` adds Tr as the reference code computes it
(not through :math:`\bot` and :math:`\top`), ``"full"`` the complete closure. Rule (5),
:math:`C \sqsubseteq D \Rightarrow \exists r.C \sqsubseteq \exists r.D`, was not used in the
paper; it is available only for existential restrictions already in the lattice.

Embedding and training
----------------------

Every element :math:`X` gets a vector :math:`f(X) \in \mathbb{R}^n`. A morphism
:math:`A \sqsubseteq B` is scored by the order violation (Eq. 6)

.. math:: s(A, B) = \lVert \max(0, f(A) - f(B)) \rVert_2 ,

which is zero iff :math:`f(A) \le f(B)` in every coordinate, and trained with the margin
loss of Eq. 7, :math:`s(A, B) + \max(0, \gamma - s(A, B'))`, where :math:`B'` is drawn
uniformly from the lattice. If the loss is zero, every morphism is preserved by :math:`f`
(Theorem 2). Optimisation follows the reference code: Adam with weight decay
:math:`10^{-4}` and a cyclic learning rate between ``learning_rate / 10`` and
``learning_rate``.

Two choices differ from the reference code and can be switched back:
``filter_negatives=True`` discards corruptions that are morphisms (Eq. 7 sums over
non-morphisms only), and ``negative_aggregation="per_negative"`` puts one hinge on each
corruption as in Eq. 7 (the code averages the scores first, ``"mean"``).

Scoring
-------

A query is scored by :math:`s` between two elements, lower is better:
:math:`C \sqsubseteq D` by :math:`s(C, D)`, :math:`C(a)` by :math:`s(\{a\}, C)` and
:math:`C \sqsubseteq \exists r.D` by :math:`s(C, \exists r.D)`. Both sides must be elements of
the lattice; CatE cannot embed a concept description that is not one (paper, Section 5), and
:meth:`~mowl.models.cate.model.CatE.score` raises
:class:`~mowl.projection.categorical.lattice.UnseenConceptError` for one. To rank
:math:`C \sqsubseteq \exists r.D'` over candidates :math:`D'`, add the elements
:math:`\exists r.D'` before training with ``existentials`` (the paper does this for protein
interactions; for datasets that name an evaluation property it is done automatically).
``model.module(rows, gci_name)`` scores rows in the layout of :mod:`mowl.evaluation`, so
the CatE benchmarks (``python -m mowl.benchmarks run cate-go-completion --method cate``)
can evaluate the model.

Example
-------

.. testcode::

    import torch as th
    from mowl.datasets import Dataset
    from mowl.models.cate.model import CatE
    from mowl.owlapi import OWLAPIAdapter

    adapter = OWLAPIAdapter()
    ns = "http://example.org/pets#"
    cat, dog, pet = (adapter.create_class(ns + n) for n in ("Cat", "Dog", "Pet"))
    ontology = adapter.create_ontology(ns + "ontology")
    for axiom in (adapter.create_subclass_of(pet, adapter.create_object_union_of(cat, dog)),
                  adapter.create_subclass_of(cat, adapter.create_complement_of(dog))):
        adapter.owl_manager.addAxiom(ontology, axiom)

    th.manual_seed(0)
    model = CatE(Dataset(ontology), embed_dim=8, margin=0.5, learning_rate=0.01,
                 batch_size=64, saturation_steps=1, transitive_closure=True)
    losses = model.train(epochs=100)
    print(losses[-1] < losses[0])
    # Cat ⊑ ¬Dog is a morphism; by rule (1) so is Dog ⊑ ¬Cat.
    dog_not_cat = adapter.create_subclass_of(dog, adapter.create_complement_of(cat))
    print(model.score(dog_not_cat) < model.score(adapter.create_subclass_of(dog, cat)))

.. testoutput::

    True
    True

Guarantees and limitations, including why Helly-type limits of convex regions do not apply
to an order embedding, are listed on the card ``cate`` of the :doc:`method catalogue
<catalogue>`.
