Embedding the ALC language
============================================
.. |alc| replace:: :math:`\mathcal{ALC}`

The :math:`\mathcal{ALC}` description logic extends :math:`\mathcal{EL}` with
negation, disjunction and universal restrictions. mOWL provides
:class:`FALCON <mowl.models.FALCONModel>` [falcon2022]_, a fuzzy-:math:`\mathcal{ALC}`
neural reasoner: every class expression is interpreted as a *fuzzy set* over a
collection of named and sampled anonymous entities, and axioms are scored through
fuzzy logical operators. The training objective combines a TBox term and two ABox
terms (concept and relation assertions), weighted by ``alpha`` and ``beta``
(:math:`\mathcal{L} = \alpha\,\mathcal{L}_T + \beta\,\mathcal{L}_{A_1} +
(1-\alpha-\beta)\,\mathcal{L}_{A_2}`).

FALCON, DF-ALC (:class:`~mowl.models.DFALCModel`) and Al-cone
(:class:`~mowl.models.ALConeModel`) are combinations of parts, as is any other model that
:class:`~mowl.models.ComposedALC` puts together from a class model, a relation model, a
measure and an objective that read axioms with class expressions of any depth (see
:doc:`/representation/combine`).

Embed2Reason (:class:`~mowl.models.E2R`) reads its axioms as rows of one kind per form of the
quantum logic, not as class expressions of any depth, so it is trained by
:class:`~mowl.models.ComposedSubspace`, the general model for axis-parallel subspace
geometries; ``E2R`` is that model with ``method="e2r"``, and its parts (the measure, the
individual model, extra regularisers) can be replaced like those of the other composed models.

.. testcode::

   from mowl.models import ComposedSubspace, E2R

   assert issubclass(E2R, ComposedSubspace)

Reproducing the Family experiment
---------------------------------

We reproduce Figure 1 of the FALCON paper. We build a small family ontology — a
class hierarchy (``SubClassOf``), defining intersections (e.g.
``Parent ⊓ Male ⊑ Father``), disjointness axioms, and one individual asserted per
concept — and train FALCON on it.

.. testcode::

   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.models import FALCONModel
   from mowl.owlapi.adapter import OWLAPIAdapter

   adapter = OWLAPIAdapter()
   ontology = adapter.create_ontology("http://mowl/family")
   cls = lambda name: adapter.create_class("http://" + name)

   concepts = ["Male", "Female", "Parent", "Child", "Person",
               "Father", "Mother", "Boy", "Girl", "Grandma"]
   c = {name: cls(name) for name in concepts}
   axioms = HashSet()

   for sub, sup in [("Male", "Person"), ("Female", "Person"), ("Parent", "Person"),
                    ("Child", "Person"), ("Father", "Male"), ("Father", "Parent"),
                    ("Mother", "Female"), ("Mother", "Parent"), ("Boy", "Male"),
                    ("Boy", "Child"), ("Girl", "Female"), ("Girl", "Child"),
                    ("Grandma", "Mother")]:
       axioms.add(adapter.create_subclass_of(c[sub], c[sup]))

   for (left, right), target in [(("Parent", "Male"), "Father"),
                                 (("Parent", "Female"), "Mother"),
                                 (("Child", "Male"), "Boy"),
                                 (("Child", "Female"), "Girl")]:
       axioms.add(adapter.create_subclass_of(
           adapter.create_object_intersection_of(c[left], c[right]), c[target]))

   for a, b in [("Male", "Female"), ("Parent", "Child"),
                ("Boy", "Girl"), ("Father", "Mother")]:
       axioms.add(adapter.create_disjoint_classes(c[a], c[b]))

   for name in concepts:
       individual = adapter.create_individual("http://" + name.lower() + "_0")
       axioms.add(adapter.create_class_assertion(c[name], individual))

   adapter.owl_manager.addAxioms(ontology, axioms)
   dataset = Dataset(ontology, validation=ontology, testing=ontology)

   model = FALCONModel(dataset, embed_dim=50, anon_e=4, alpha=0.5, beta=0.5,
                       learning_rate=0.01, num_negs=8)
   model.train(epochs=300, validate_every=100)

The learned fuzzy model can be inspected with
:class:`FALCONVisualizer <mowl.visualization.FALCONVisualizer>`, which computes the
membership degree :math:`\mu(C, e)` of every concept ``C`` over the named individuals
and a number of sampled anonymous entities, and renders it as a heatmap:

.. testcode::

   from mowl.visualization import FALCONVisualizer

   visualizer = FALCONVisualizer(model)
   matrix, concepts, entities = visualizer.membership_matrix(n_anon=2)

.. figure:: falcon_family_heatmap.png
   :alt: FALCON concept-by-entity membership heatmap on the Family ontology
   :align: center

   FALCON membership degrees on the Family ontology (dark = membership close to 1).
   Each individual is a member of its asserted concept and of every super-concept
   entailed by the TBox — for example, ``boy_0`` belongs to ``Boy``, ``Child``,
   ``Male`` and ``Person`` — while ``Nothing`` is empty and ``Thing`` is universal.

You can also access the learned embeddings:

.. testcode::

   class_embeddings = model.class_embeddings
   object_property_embeddings = model.object_property_embeddings

How the ALC models are trained
------------------------------

The ALC models have no training loop of their own: they run the shared loop of
:mod:`mowl.training.loop` (one optimisation step per epoch over all batches of all axiom
patterns, with validation and checkpointing every ``validate_every`` epochs), and what is
particular to a method is a part:

* the combination of the axiom losses is the objective part
  :class:`~mowl.representation.library.logic_alc.AxiomKindWeighting`
  (``model.module.objective.weighting``; ``alpha`` and ``beta`` are its weights of the
  terminological axioms and the class assertions, the role assertions get
  ``1 - alpha - beta``). Another weighting is a swap of the objective;
* the anonymous entities of FALCON are drawn by
  :class:`~mowl.representation.library.logic_alc.AnonymousElements`;
* the annealed relaxation of Al-cone
  (:class:`~mowl.representation.library.logic_alc.AnnealedRelaxation`) is a training regime
  of the geometry, which the loop consults like any other regime. A regime that changes the
  course of training (a learning-rate schedule, early stopping) can be assigned; one that
  changes the loss or the negatives is refused, since the objective reads the axioms itself.

.. testcode::

   from mowl.models import ComposedALC
   from mowl.representation.library.logic_alc import AxiomKindWeighting
   from mowl.training import LinearDecay

   assert model.module.objective.weighting.tbox == 0.5
   tbox_only = ComposedALC(dataset, method="dfalc", embed_dim=10,
                           objective_options={"weighting": AxiomKindWeighting(1.0, 0.0)})
   tbox_only.training_regime = LinearDecay()
   tbox_only.train(epochs=2)

A combination of parts is checked when the model is built. A replaced part that cannot
evaluate the axioms of the ontology, or that no axiom reads, is refused with the reason, for
example the truth tables of DF-ALC as the classes of FALCON, whose relation network evaluates
class expressions on the element vectors and sampled anonymous entities:

.. testcode::

   from mowl.representation import ConstructorUnavailable

   try:
       ComposedALC(dataset, method="falcon", embed_dim=10, anon_e=2,
                   classes="fuzzy set (product)")
   except ConstructorUnavailable as error:
       print("refused")

.. testoutput::

   refused

Three loops of mOWL are not the shared loop on purpose: the fit of the scoring head of
vector-only models (:class:`~mowl.evaluation.model_scoring.PairClassifierHead`, an inner fit
of the evaluation on fixed vectors) and the gradient solver of the solved methods
(:class:`~mowl.representation.library.solved_solver.GradientFit`, which minimises one
program) are inner solvers; the regimes of the canonical mixture
(:mod:`mowl.representation.library.mixture_training`) are a training loop of their own that
accumulates gradients chunk by chunk, which the shared loop cannot do yet, so that model
refuses every training regime.
