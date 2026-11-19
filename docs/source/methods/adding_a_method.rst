Adding a method to mOWL
=======================

Every embedding method in mOWL consists of the same pieces, in the same places. This page
is the checklist. A method that cannot be implemented (for example because the publication
gives no learning procedure, or needs proprietary software) still gets the first piece, the
catalogue card.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Piece
     - Location
     - Purpose
   * - Catalogue card
     - ``mowl/catalog/entries/<key>.py``
     - Classification, logic, guarantees with sources, limitations, references.
   * - Neural module
     - ``mowl/nn/<logic>/<key>/`` (``module.py``, ``losses.py``)
     - The geometry and the per-axiom losses, with no I/O.
   * - Model
     - ``mowl/models/<key>/model.py``
     - Dataset handling, training loop, scoring; subclasses a base model.
   * - Tests
     - ``tests/nn/test_<key>_*.py``, ``tests/models/test_<key>_*.py``
     - Loss semantics, guarantees, a small end-to-end run.
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

2. The implementation
---------------------

For :math:`\mathcal{EL}^{++}` methods:

* The module subclasses :class:`mowl.nn.ELModule` and implements the per-normal-form losses
  (``gci0_loss`` ... ``gci3_bot_loss``, and ``role_inclusion_loss`` / ``role_chain_loss`` if
  the method covers role axioms). Each loss takes a batch of index tensors and ``neg`` and
  returns one non-negative value per row. Declare ``neg_capable_gcis`` and
  ``role_axiom_capable`` truthfully.
* The model subclasses :class:`mowl.base_models.EmbeddingELModel` and overrides
  ``init_module`` and, where needed, the training hooks (``get_negative_sampling_config``,
  ``compute_loss``, ``get_regularization_loss``, ``on_epoch_start``,
  ``additional_training_loss``). Reuse the generic training loop unless the method needs a
  different optimisation procedure.

For :math:`\mathcal{ALC}` methods, subclass :class:`mowl.base_models.EmbeddingALCModel` and
use :mod:`mowl.datasets.alc`.

Methods that are not trained by gradient descent (exact constructions, convex programmes)
subclass :class:`mowl.base_models.Model` directly, or live in :mod:`mowl.geometric` if they
construct a geometric interpretation from the canonical model.

Training that is specific to the representation stays in the model. Anything that could be
used with another representation is a training regime (see above); the generic training loop
of :class:`mowl.base_models.EmbeddingELModel` consults the regime attached to
``model.training_regime`` and falls back to the model's own behaviour for every hook the
regime does not implement.

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
* **End to end.** Train for a few epochs on a small dataset and check that the loss
  decreases and that scoring and evaluation run.
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
