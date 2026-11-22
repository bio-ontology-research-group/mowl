Adding a method to mOWL
=======================

Every method in mOWL is a *combination of parts*: a class model, a relation model, a measure
of "inside", an objective, regularisers, a training regime and an evaluation regime. A new
method is therefore a short function that names its parts, possibly with one or two new parts,
plus the pieces that show it is right: a catalogue card, a direct implementation kept as a
baseline, and a *Pair* that compares the two. This page is the checklist. A method that cannot
be implemented (for example because the publication gives no learning procedure, or needs
proprietary software) still gets the first piece, the catalogue card.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Piece
     - Location
     - Purpose
   * - Catalogue card
     - ``mowl/catalog/entries/<key>.py``
     - Classification, logic, guarantees with sources, limitations, references.
   * - The method as parts
     - ``mowl/representation/library/<family>_methods.py`` (a function registered with
       ``register_method``)
     - Names the class model, relation model, measure and objective of the method.
   * - New parts, if needed
     - ``mowl/representation/library/``, ``mowl/training/``, ``mowl/evaluation/``
     - A part is general: another method must be able to use it.
   * - Baseline
     - ``mowl/baselines/models/<key>/``, ``mowl/baselines/nn/<...>/``
     - The direct implementation of the paper, frozen, with its own copies of any
       computation. Nothing outside ``mowl/baselines`` imports it.
   * - Pair
     - ``tests/baselines/pairs_<group>.py``
     - Same losses, same scores and same training run as the baseline, from the same
       parameters.
   * - Tests
     - ``tests/representation/``, ``tests/models/``
     - Each new part on its own, and in a combination outside the paper.
   * - Documentation
     - ``docs/source/methods/<key>.rst``
     - Short description and a runnable ``.. testcode::`` example.
   * - Published results
     - ``mowl/benchmarks/published/<source>.py``
     - Numbers printed in the paper, for the benchmarks in :mod:`mowl.benchmarks`.

``<key>`` is the lower-case catalogue key of the method (``elem``, ``box2el``, ``elkem``).

Representation, training and evaluation are separate
----------------------------------------------------

A publication usually bundles three things: a representation (the method), a way to train it
and a way to evaluate it. In mOWL they are catalogued and implemented separately, so that any
training or evaluation procedure can be combined with any compatible method, and so that it
is explicit under which procedures a published number was obtained.

* The representation is the *method*: a catalogue card and a model, as described below.
* A procedure for training that is not specific to one representation (a negative sampling
  scheme, a way to combine positive and negative scores, a schedule, a model selection rule)
  is a *training regime*: a :class:`mowl.training.TrainingRegime` subclass in
  :mod:`mowl.training` with a :class:`mowl.catalog.RegimeCard` in
  ``mowl/catalog/regimes/<key>.py``.
* A procedure for testing (how the test set is built, what is ranked against what, what is
  filtered, which metrics) is an *evaluation regime*: a
  :class:`mowl.evaluation.EvaluationRegime` subclass in :mod:`mowl.evaluation`, again with a
  ``RegimeCard``.

A regime is catalogued whether or not it was ever published on its own. When a new paper
trains or tests in a way that mOWL does not have yet, add the regime; when it uses existing
ones, name them in the ``training`` and ``evaluation`` fields of the method card. See
:doc:`training_and_evaluation` for how the literature trains and tests.

1. The catalogue card
---------------------

Create ``mowl/catalog/entries/<key>.py`` with a module-level ``CARD``
(:class:`mowl.catalog.MethodCard`), or ``CARDS`` for several closely related methods. The
module is discovered automatically.

Rules:

* **Every guarantee names its source** (publication and theorem, proposition or section) and
  the **condition** under which it holds. The usual conditions are ``"zero loss"`` (a learned
  embedding is a model only if the loss reaches zero), ``"by construction"`` and
  ``"always"``. Do not record a claim you have not read in the source.
* Negative results are guarantees of kind ``LIMITATION`` (for example the Helly property of
  box semantics).
* ``origin`` says how the code relates to the paper: ``REIMPLEMENTATION`` (follows the
  published losses), ``ADAPTATION`` (the paper gives a geometry or construction, the training
  objective is ours) or ``ORIGINAL``. An adaptation must say in ``notes`` what was added.
* A method that is not implemented has ``status=Status.CATALOGUED`` and a
  ``not_implemented_reason``.

2. The method as a combination of parts
---------------------------------------

Write the method as a function that returns the geometry (a
:class:`~mowl.representation.ComposedGeometry`) and register it under the key of its card.
Reuse the parts that exist (:data:`mowl.representation.CLASS_MODELS`,
:data:`~mowl.representation.RELATION_MODELS`, :data:`~mowl.representation.MEASURES`, the
objectives and regularisers of :mod:`mowl.representation`) and give the options of the paper
as options of the function. The function below is a method nobody has published, built the
same way as the published ones; ``ComposedEL(dataset, method=...)`` trains it:

.. testcode::

   import logging
   import torch as th
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ComposedEL
   from mowl.representation import (BoxClasses, ComposedGeometry, MEASURES, METHODS,
                                    METHOD_KINDS, Objective, Translation, register_method)

   logging.disable(logging.CRITICAL)
   th.manual_seed(0)

   @register_method("shifted_boxes")
   def shifted_boxes(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1):
       return ComposedGeometry(BoxClasses(nb_classes, embed_dim),
                               Translation(nb_relations, embed_dim),
                               MEASURES["bounds per coordinate"](), Objective(margin))

   model = ComposedEL(FamilyDataset(), method="shifted_boxes", embed_dim=8,
                      learning_rate=0.01)
   model.train(epochs=2)
   print(model.as_geometry.parts["classes"], "with", model.as_geometry.parts["relations"])

   del METHODS["shifted_boxes"], METHOD_KINDS["shifted_boxes"]   # leave the registry as it was

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

The class users import (``from mowl.models import X``) is a thin class that names this
combination: it wires parts and hands rows to them, and does no tensor arithmetic of its own
(``tests/baselines/test_equivalence.py`` checks this). Methods trained on queries,
sequences, ALC expressions or by construction register with ``trained_on="queries"``,
``"sequences"``, ``"expressions"`` or ``"construction"``; see
:func:`~mowl.representation.register_method`.

If an existing part almost does what the paper needs, add an *option* to it before writing
a new part. If no part does, write a new one and register it (next section). Quirks of the
paper's code are options of a part, selected by the function that builds the published
method, never defaults: see :doc:`/representation/defaults`.

A new part
~~~~~~~~~~

A part is named for what it computes, not for the method it came from, and is registered in
the registry of its kind (``register_class_model``, ``register_relation_model``,
``register_measure``; training regimes and evaluation regimes have their own registries and
cards). Its docstring says what it computes. It has a unit test that exercises it alone and a
test that uses it in a combination other than the method it came from (a swap). A worked
example, a new relation model, is on :doc:`/representation/extending`.

A baseline and its Pair
~~~~~~~~~~~~~~~~~~~~~~~

The direct implementation of the paper goes to ``mowl/baselines/models/<key>/`` (the model)
and ``mowl/baselines/nn/<...>/`` (the module). It is *frozen*: it owns its copies of every
loss, score and geometry function and does not import computing code from the library (data
loading, the OWL API layer, datasets and the base ``Model`` plumbing are fine). Nothing
outside ``mowl/baselines`` imports it.

The comparison is a :class:`Pair` in ``tests/baselines/pairs_<group>.py``. The harness builds
both sides from the same seed, copies every parameter of the baseline into the combination,
and checks that the losses of every axiom form (positives and negatives), the scores that the
evaluation regimes rank, and the parameters, scores and losses after a short training run all
agree. Compare more than one number: the positive and negative scores separately, every form
and every ranking score, not only a total. A check that cannot hold is listed in ``skip``
with its reason; a method with no direct implementation to compare with (a general model, a
source part) is listed in ``EXEMPT`` with its reason, which is printed in the documentation.

.. testcode::

   from tests.baselines.harness import Pair

   def build_new(dataset):
       from mowl.models import ComposedEL
       return ComposedEL(dataset, method="elbe", embed_dim=4)

   def build_old(dataset):
       from mowl.baselines.models.elbe.model import ELBE
       return ELBE(dataset, embed_dim=4)

   def copy(old, new):
       """Copy every parameter and buffer of the baseline into the combination."""

   def scores(model):
       """What the evaluation regime ranks, on a fixed input (a tensor or a dict of them)."""

   pair = Pair(key="elbe/example", new=build_new, old=build_old, copy=copy, scores=scores)
   print(pair.method, pair.atol)

.. testoutput::

   elbe 1e-05

A Pair fails when the combination is not the method. Do not loosen the tolerance or change a
formula of the baseline to make it pass; find the difference, and if it is a defect, fix the
combination (the correct behaviour is the default, the old one an option for the recipe).
Where a tolerance larger than ``atol=1e-5`` is unavoidable, the exact numerical reason sits
next to it, and a test shows the reason holds (see
``tests/baselines/test_concept2box_bias_noise.py``).

Methods that are not trained by gradient descent (exact constructions, convex programmes)
register with ``trained_on="construction"`` and build their model from a source (the
canonical model, a least model, a solver) and a reading of it as regions
(:mod:`mowl.representation.construction`); the construction itself lives in
:mod:`mowl.geometric` or in a part under ``mowl/representation/library/``. Their Pair compares
what the construction produces: the regions element by element, the truth values of axioms of
every form, and the guarantees the construction promises (see
``tests/baselines/pairs_construct.py``).

Training that is specific to the representation is still not a method of the model class.
Anything that could be used with another representation is a training regime (see above);
the generic training loop (:mod:`mowl.training.loop`) consults the regime attached to
``model.training_regime``.

Every model says how the rank-based evaluation regimes obtain axiom scores from it, with
the class attribute ``axiom_scoring`` (see :mod:`mowl.evaluation.model_scoring`):
``"module"`` if ``module(rows, gci_name)`` is the cost of a candidate axiom (inherited by
the :math:`\mathcal{EL}^{++}` models), ``"method"`` with an
``axiom_costs(gci_name, rows)`` method otherwise, and ``"vectors"`` if the model gives
class vectors only. ``scored_forms`` lists the normal forms it can score. With this the
benchmark runner, every ranking regime and the joint ``common-*`` benchmarks
(:ref:`joint-benchmarks`) work for the model without model-specific code.

Optional dependencies (a language model, a conic solver) are imported inside the functions
that need them and raise an :class:`ImportError` that names the package. They are listed in
``optional_dependencies`` of the card and are never added to ``install_requires``.

3. Tests
--------

* **Semantics.** For hand-built configurations, the loss of an axiom is zero when the
  geometric condition holds and positive when it does not. This is the test that catches a
  wrong sign or a swapped argument.
* **Guarantees.** Each ``Guarantee`` of kind ``SOUNDNESS`` on the card has a test: build or
  train a zero-loss embedding of a small ontology and check, with code that is independent
  of the loss, that it satisfies every axiom. Name the test in ``verified_in_mowl``.
* **Same as the baseline.** The Pair of section 2: losses, scores and a training run.
* **End to end.** Train for a few epochs on a small dataset and check that the loss
  decreases (``assertLess(last, first)`` on a fixed batch) and that scoring and evaluation
  run.
* **A combination outside the paper.** Each new part is used by another method's parts in a
  ``ComposedEL`` or ``build_method`` swap, and the swap trains.
* Tests that train for longer than a few seconds are marked ``@pytest.mark.slow``.

4. Documentation
----------------

``docs/source/methods/<key>.rst`` gives a short description (geometry, how each axiom type
is interpreted, differences from the paper) and one runnable example in a ``.. testcode::``
block. The table of guarantees is generated from the card and must not be repeated by hand.

5. Published results
--------------------

If the paper reports results on a benchmark of :mod:`mowl.benchmarks`, add them to
``mowl/benchmarks/published/<source>.py`` as :class:`mowl.benchmarks.PublishedResult`
objects, with the table number as ``locator`` and the stated hyperparameters as
``settings``. Record only numbers read from the publication itself. If the paper uses a
benchmark mOWL does not have yet, add it to ``mowl/benchmarks/entries/``.

6. Recipe and reproduction
--------------------------

A published number depends on how the paper's *code* trained and tested, which is often
not what the paper's text or mOWL's defaults do. If the method has published results:

* Read the reference code and write down its training design: meaning of the margin,
  negatives (which columns, how many, redrawn or fixed), what a step is, schedule, model
  selection. Read its evaluation: score, candidates, filtering, ties.
* Express each difference from mOWL's defaults as a training regime, an evaluation regime,
  or an argument of one. Do not add a flag to the model for it. A design that belongs to
  one code base is a named composition in :mod:`mowl.training.reference` (for example
  :class:`mowl.training.Box2ELCodeTraining`), with a ``RegimeCard``.
* Add a :class:`mowl.benchmarks.Recipe` in ``mowl/benchmarks/recipe_sets/<source>.py``:
  model arguments, training regime with its arguments, and epochs. Modules of that package
  are discovered automatically and expose ``RECIPES``.
* If the reference code has a defect that the published number depends on, mOWL's default
  is the correct behaviour and the recipe names the option that gives the published one.
  Say so in the recipe's ``notes`` and on the card.
* Run the recipe (``python -m mowl.benchmarks run <benchmark> --method <key> --recipe``)
  with three seeds and record the outcome in ``docs/source/benchmarks/reproduction.rst``.
  Never tune on the test set, and state the number of seeds.
