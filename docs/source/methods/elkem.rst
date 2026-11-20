ELK-Em
======

ELK-Em [singh2026]_ embeds a normalised :math:`\mathcal{EL}^+_\bot` ontology so that an
embedding with zero loss is a model of the ontology and satisfies everything the inference
rules of the ELK reasoner derive from it. The catalogue key is ``elkem``; its guarantees,
with sources and the tests that check them, are on the card in the
:doc:`method catalogue <catalogue>`.

Geometry
--------

* A concept name is an axis-parallel box :math:`B_A = [l_A, u_A]`. :math:`\top` is the full
  box and :math:`\bot` the empty box; both are fixed, not trained.
* An individual is a point :math:`e_a`.
* A role :math:`r` is a triple :math:`\langle R_r, E_r, T_r\rangle`: a range box
  :math:`R_r`, an error box :math:`E_r` and a diagonal affine map
  :math:`T_r(x) = m_r x + c_r` with :math:`m_r > 0`. It relates
  :math:`r^{\mathcal{E}} = \{(x, y) \mid x \in T_r(y) \oplus E_r,\ y \in R_r\}`.
* Hence :math:`(\exists r.C)^{\mathcal{E}} = T_r(C^{\mathcal{E}} \cap R_r) \oplus E_r`, again a
  box, and every concept description is a box.

Losses
------

With centres :math:`c`, half-widths :math:`o` and the separation
:math:`d(B_x, B_y) = |c_x - c_y| - o_x - o_y`, the paper uses three primitives:
inclusion :math:`\ell_\subseteq = \frac1n\lVert\mathrm{ReLU}(d + 2o_x + \gamma)\rVert_1`,
overlap :math:`\ell_{\cap+} = \frac1n\lVert\mathrm{ReLU}(d + \gamma)\rVert_1` and
disjointness :math:`\ell_{\cap-} = \mathrm{ReLU}(\gamma - \max_i d_i)`.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Axiom
     - Loss (Table 2 of the paper)
   * - :math:`C \sqsubseteq D`
     - :math:`\ell_\subseteq(B_C, B_D)`
   * - :math:`C \sqsubseteq \bot`
     - none: :math:`B_C` is fixed to the empty box
   * - :math:`C_1 \sqcap C_2 \sqsubseteq D`
     - :math:`\ell_\subseteq(B_{C_1} \cap B_{C_2}, B_D) + \ell_{\cap+}(B_{C_1}, B_{C_2})`
   * - :math:`C_1 \sqcap C_2 \sqsubseteq \bot`
     - :math:`\ell_{\cap-}(B_{C_1}, B_{C_2})`
   * - :math:`C \sqsubseteq \exists r.D`
     - :math:`\ell_\subseteq(B_C, B_{\exists r.D})`
   * - :math:`\exists r.C \sqsubseteq D`
     - :math:`\ell_\subseteq(B_{\exists r.C}, B_D) + \ell_{\cap+}(R_r, B_C)`
   * - :math:`\exists r.C \sqsubseteq \bot`
     - :math:`\ell_{\cap-}(R_r, B_C)`: :math:`C` and the range are disjoint, :math:`C` itself
       need not be empty
   * - :math:`r \sqsubseteq s`
     - :math:`\ell_\subseteq(R_r, R_s)` plus :math:`\ell_\subseteq(T_r(y) \oplus E_r,
       T_s(y) \oplus E_s)` at the two corners :math:`y \in \{l^R_r, u^R_r\}`
   * - :math:`r_1 \circ r_2 \sqsubseteq s`
     - the role inclusion loss of :math:`\max(r_1 \circ r_2) = \langle R_{r_2}, m_{r_1}
       E_{r_2} \oplus E_{r_1}, T_{r_1} \circ T_{r_2}\rangle` against :math:`s`, a sufficient
       condition for the chain
   * - :math:`C(a)`, :math:`r(a, b)`
     - :math:`\ell_\subseteq([e_a, e_a], B_C)`;
       :math:`\ell_\subseteq([e_a, e_a], T_r([e_b, e_b]) \oplus E_r) +
       \ell_\subseteq([e_b, e_b], R_r)`

An axiom is scored by :math:`s(\alpha) = \exp(-\ell^0(\alpha))`, the loss with margin 0.
:class:`~mowl.nn.el.elkem.module.ELKEmModule` returns :math:`\ell^0` in evaluation mode, so
mOWL's evaluators (which rank by ascending value) rank exactly as the paper does.

Training (:class:`~mowl.models.elkem.model.ELKEm`): one mini-batch per normal form and epoch,
the mean losses of all normal forms, and, when there are no assertions, a Pearson
correlation loss between the scores of positive ``gci0``/``gci2`` axioms and of ``k_neg``
random corruptions each. Corrupted assertions are trained with
:math:`\mathrm{ReLU}(-\log\tau - \ell)`. An optional lexical regulariser aligns the
distances of individuals with the cosine similarity of external representations (for
example protein language model embeddings) given as a matrix. Adam with a step decay of the
learning rate; with a validation ontology, the checkpoint with the best validation median
rank of the ``gci0`` subsumptions is kept.

With a :attr:`~mowl.base_models.elmodel.EmbeddingELModel.training_regime` attached, the
generic regime-aware loop of
:meth:`~mowl.base_models.elmodel.EmbeddingELModel.train` runs instead of the built-in one.
The regime ``elkem-code-training``
(:class:`~mowl.training.reference.ELKEmCodeTraining`) is the design of the authors'
ranking code, and the recipes of the ELK-Em paper
(:mod:`mowl.benchmarks.recipe_sets.singh2026`) use it together with the evaluation regime
:class:`~mowl.evaluation.normal_form_ranking.ELKEmSubsumptionRankingRegime` of the
``elkem-*-subsumption`` benchmarks.

Differences from the paper and the authors' code are listed in the class docstring and in
the card's ``notes``. The most important ones: per-axiom losses are not squared (as in the
paper, unlike the authors' code), ``gci2`` has no overlap term (as in Table 2, unlike the
code), and the box of :math:`\exists r.C` is kept empty when :math:`C \cap R_r` is empty.

Example
-------

.. testcode::

   import torch as th
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models.elkem.model import ELKEm
   from mowl.nn.el.elkem.module import ELKEmModule

   # Training. Use load_role_axioms=True for ontologies with role inclusions and chains,
   # and pass a validation ontology to select the checkpoint by median rank.
   model = ELKEm(FamilyDataset(), embed_dim=10, k_neg=3)
   model.train(epochs=5)

   # ∃r.C ⊑ ⊥ holds when C misses the range of r; C itself stays non-empty.
   module = ELKEmModule(nb_ont_classes=1, nb_rels=1, embed_dim=2, margin=0.1)
   module.class_center.weight.data[0] = th.tensor([3.0, 3.0])
   module.range_center.weight.data[0] = th.tensor([0.0, 0.0])
   module.range_offset.weight.data[0] = th.tensor([1.0, 1.0])
   print(module.gci3_bot_loss(th.tensor([[0, 0, 0]])).item())
   print(module.score(th.tensor([[0, 0]]), "gci0").item())

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...0.0
   1.0
