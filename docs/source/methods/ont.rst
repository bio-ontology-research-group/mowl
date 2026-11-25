OnT (Ontology Transformer)
==========================

OnT [yang2025ont]_ extends :doc:`HiT <hit>` from class hierarchies to
:math:`\mathcal{EL}`. The catalogue key is ``ont``; the model is
:class:`mowl.models.ont.OnT`, a combination of parts (the function ``ont`` in
:mod:`mowl.representation.library.hierarchy_methods`: HiT's class model and measure, a rotation
per role as the relation model, the existential and conjunction losses of the geometry); the
module is :class:`mowl.nn.lm.ont.OnTModule` and the verbaliser is in :mod:`mowl.text.verbalise`.
The earlier direct implementation is kept in :mod:`mowl.baselines.models.ont` as the reference.

Concepts, roles and axioms
--------------------------

* **Concepts.** Every concept, atomic or complex, is the encoding of its verbalisation,
  a point of the Poincaré ball of curvature :math:`-1/d` (as in HiT). The verbalisation
  rules of Section 4.1 are :math:`V(C \sqcap D)` = "*V(C)* and *V(D)*" and
  :math:`V(\exists r.C)` = "something that *V(r)* some *V(C)*". For example
  :math:`\mathit{Person} \sqcap \exists \mathit{teach}.\mathit{Class}` becomes "person and
  something that teaches some class". The published data use "*V(r)* some *V(C)*"
  (``verbalisation_style="release"``); DeepOnto's rules, which OnT's normaliser uses to name
  new concepts, are ``verbalisation_style="deeponto"``.
* **Roles.** A role :math:`r` is the map :math:`f_r(v) = k_r \otimes R(\Theta_r) v`: a
  rotation of the ball by :math:`d/2` planar rotations, then a Möbius scaling (Eq. 4).
  :math:`\Theta_r` and :math:`k_r` are a linear function of the encoding of the role label.
  Rotations preserve hyperbolic distances and norms (Proposition 1). As in the reference
  code, :math:`k_r` is not applied by default (``role_scaling=True`` applies it).
* **Axioms.** :math:`C \sqsubseteq D` is trained with HiT's loss :math:`L_\prec(x_C \prec
  x_D)` and one random negative. Every :math:`\exists r.D` in the ontology adds
  :math:`\frac{1}{2}(L_\prec(x_{\exists r.D} \prec f_r(x_D)) + L_\prec(f_r(x_D) \prec
  x_{\exists r.D}))` (Eq. 6), and every :math:`C \sqcap D` adds :math:`\frac{1}{2}(L_\prec(
  x_{C \sqcap D} \prec x_C) + L_\prec(x_{C \sqcap D} \prec x_D))` (Eq. 7).
  ``use_roles=False`` drops both, which is the ablation OnT(w/o r).

A conjunction is a point placed below its conjuncts, not an intersection of regions, and
:math:`\bot` and disjointness are not represented. The paper proves no soundness or
faithfulness result. The monotonicity :math:`A \sqsubseteq B \Rightarrow \exists r.A
\sqsubseteq \exists r.B` holds for the role images :math:`f_r(x_A), f_r(x_B)`, not for the
text encodings of :math:`\exists r.A` and :math:`\exists r.B` that are scored at test time
(``existential_mode="role"`` scores with the role images instead).

Subsumptions are scored with Eq. 8, :math:`-(d(x_C, x_D) + \lambda(\lVert x_D\rVert -
\lVert x_C\rVert))`, with :math:`\lambda` chosen on the validation subsumptions. The
paper states the grid :math:`\{0, 0.1, \dots, 1\}`; the reference code searches
:math:`\{0.0, \dots, 1.9\}`, which the recipes use (``centri_weight_grid``).

Differences from the paper are listed on the card; the main ones are that OnT is trained
on the axioms of the given ontology (mOWL does not reimplement OnT's name-preserving
normaliser) and that the losses are averaged, not summed, as in the reference code.

Example
-------

With the stand-in encoder, so that it runs without ``sentence-transformers``:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models.ont import OnT
   from mowl.representation.library.text_encoders import BagOfWordsEncoder
   from mowl.owlapi import OWLAPIAdapter

   dataset = FamilyDataset()
   model = OnT(dataset, encoder=BagOfWordsEncoder(dim=16), label_fallback="camel_case",
               batch_size=16, learning_rate=0.01, warmup_steps=0)
   model.train(epochs=2)

   adapter = OWLAPIAdapter()
   ns = "http://"
   has_child = adapter.create_object_property(ns + "hasChild")
   parent = adapter.create_object_some_values_from(has_child,
                                                   adapter.create_class(ns + "Person"))
   axiom = adapter.create_subclass_of(adapter.create_class(ns + "Father"), parent)
   print(model.text(parent))
   score = model.score(axiom)                # higher: more likely

.. testoutput::

   something that has child some person

With ``sentence-transformers`` installed (downloads all-MiniLM-L12-v2):

.. testcode::
   :skipif: __import__("importlib").util.find_spec("sentence_transformers") is None

   model = OnT(dataset, label_fallback="camel_case")
   model.train(epochs=1)

Benchmarks
----------

OnT's own benchmarks are ``ont-{galen,go,anatomy}-{prediction,inference}``, with the data
of the authors' release (:class:`mowl.datasets.builtin.yang2025ont.Yang2025OnTDataset`,
labels included) and the ranking regimes of :mod:`mowl.evaluation.ont_ranking`. The
published Tables 2-4 are recorded for OnT, HiT and the baselines the OnT authors re-ran
under their normalisation; those numbers are not comparable with the ``box2el-*``
benchmarks. The recipes of the paper (``--recipe`` with methods ``ont`` and ``hit`` on
these benchmarks) train with the named design ``ont-code-training``
(:mod:`mowl.training.lm_reference`, cards in :mod:`mowl.catalog.regimes.lm_code_designs`):
1 epoch, batch size 64, 1 negative per axiom, margins :math:`\alpha = 3.0`,
:math:`\beta = 0.5`, learning rate :math:`10^{-5}`, 500 warm-up steps, and the code's
:math:`\lambda` grid; the HiT baseline is restricted to the atomic axioms. The ``box2el-*`` datasets themselves have no ``rdfs:label`` annotations: GALEN's
IRI fragments are readable CamelCase names (``label_fallback="camel_case"``), while GO and
ANATOMY have opaque identifiers, so text-based methods should not be run on them.

The benchmark runner does not know the language-model family by default;
:func:`mowl.benchmarks.protocols.ont.main` registers it and then works as the benchmark
command line interface. A run on a GPU, and the comparison with the published numbers:

.. code-block:: bash

   python -c "import mowl; mowl.init_jvm('16g'); \
   from mowl.benchmarks.protocols.ont import main; main()" \
       run ont-galen-prediction --method ont --recipe --device cuda \
       --out ont-galen.json
   python -m mowl.benchmarks compare ont-galen.json
