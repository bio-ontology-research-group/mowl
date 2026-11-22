Training regimes and the DELE regime
=====================================

A paper on ontology embeddings usually bundles a *representation* (balls, boxes, ...), a
*training regime* (which negatives are sampled, which axioms count as positives, how the
losses are combined) and an *evaluation regime*. mOWL keeps them apart: the representation
is the model, and a training regime is a :class:`~mowl.training.TrainingRegime` object
that is assigned to any :class:`~mowl.base_models.EmbeddingELModel`:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ELEmbeddings
   from mowl.training import dele_regime

   dataset = FamilyDataset()
   model = ELEmbeddings(dataset, embed_dim=10)
   model.training_regime = dele_regime()   # ELEm+l+n of DELE
   model.train(epochs=2)

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

Without a regime (``model.training_regime = None``, the default) training is exactly as
before. The generic training loop of :meth:`EmbeddingELModel.train
<mowl.base_models.EmbeddingELModel.train>` consults the regime's hooks
(``training_datasets``, ``negatives``, ``filter_negatives``, ``loss``, ``on_epoch_start``,
``additional_loss``, ``optimizer``) and falls back to the model whenever a hook returns
its neutral default. Several regimes are combined with
:class:`~mowl.training.ComposedRegime`. Models that override ``train`` with a loop of their
own may ignore regimes.

A model with a training loop of its own can still be driven by regimes:
:class:`~mowl.models.GroundedEL` calls the same hooks from its loop, and its default
training design is the composition :class:`~mowl.training.GroundedELTraining` of
:class:`~mowl.training.TemperatureAnnealing`, a learning-rate decay,
:class:`~mowl.training.IdleComponentReseeding` and
:class:`~mowl.training.LagrangianDualWeights`. Each of them states the small protocol a
model must provide (a temperature, ``reseed_components``, ``dual_weights``) and refuses
other models when it is attached. See :doc:`../methods/grounded`.

The readings of :class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.MultiELBE` other
than ``"axiom"`` are regimes of :mod:`mowl.training.structural`:
:class:`~mowl.training.structural.StructuralTables` (the definition table and the targets of
the structural reading, compiled when training starts),
:class:`~mowl.training.structural.ExtraPenalties` (its existence, size and witness penalties)
and :class:`~mowl.training.structural.SampledPointLoss` (the point loss on a sample of the
canonical elements). The models attach them (``model.method_regimes()``); a regime assigned
to ``training_regime`` is added to them, and one of the same class replaces one of them:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import MultiBoxEL
   from mowl.training import IdleComponentReseeding
   from mowl.training.structural import ExtraPenalties, StructuralTables

   model = MultiBoxEL(FamilyDataset(), embed_dim=4, objective="structural")
   model.training_regime = IdleComponentReseeding(every=1)
   kinds = {type(r) for r in model.training_regime.regimes}
   assert {StructuralTables, ExtraPenalties, IdleComponentReseeding} <= kinds

Models whose data are not the EL normal forms (the edges of a graph, the morphisms of a
lattice, the typed rows of E2R, verbalised triplets) use the same hooks through
:class:`~mowl.training.RegimeTrained`: :class:`~mowl.models.E2R`,
:class:`~mowl.models.BinaryCodeBox`, :class:`~mowl.models.CatE`, :class:`~mowl.models.HiT`,
:class:`~mowl.models.OnT` and :class:`~mowl.models.FaithEL` have no loop of their own. Each names
its training as a combination of regimes (read ``model.training_regime``): batching
(:class:`~mowl.training.EpochBatches`, :class:`~mowl.training.PerNormalFormMiniBatches`,
:class:`~mowl.training.AlternatingMinibatches`), negatives
(:class:`~mowl.training.EdgeCorruption` with :class:`~mowl.training.KnownRowMask`), the loss
(:class:`~mowl.training.SelfAdversarialBCE`, :class:`~mowl.training.GroupedMarginLoss`,
:class:`~mowl.training.SumLoss`, :class:`~mowl.training.SampledLogicalLosses`), the schedule
(:class:`~mowl.training.CyclicLearningRate`, :class:`~mowl.training.LinearWarmupDecay`),
selection and stopping (:class:`~mowl.training.ValidationPatience`) and the projection of the
parameters (:class:`~mowl.training.Regularization` with an
:class:`~mowl.training.IntervalProjection` or a :class:`~mowl.training.ModuleProjection`).
Every one of them can be used by the other methods and replaced in these.

Strict and slack margins
------------------------

The margin :math:`\gamma` of an inclusion loss has two meanings in the literature, and
implementations of the same method disagree on which one they use.

* *Strict*: the inner box must lie inside the outer one with a gap of :math:`\gamma`.
  Zero loss implies that the axiom holds.
* *Slack*: the inner box may stick out by up to :math:`\gamma`. Zero loss does not imply
  that the axiom holds, so the embedding need not be a model of the ontology.

Each module declares the meaning of its ``margin`` parameter in
:attr:`mowl.nn.ELModule.margin_semantics` (ELBE and TransBox: strict;
ELEmbeddings, EmEL++ and Box\ :sup:`2`\ EL: slack). The regimes
:class:`~mowl.training.StrictMargin` and :class:`~mowl.training.SlackMargin` set the
margin by meaning, whatever the module's convention:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ELBE
   from mowl.training import SlackMargin, margin_semantics

   model = ELBE(FamilyDataset(), embed_dim=4)
   print(margin_semantics(model))
   model.training_regime = SlackMargin(0.05)
   print(model.module.margin)

.. testoutput::
   :options: +ELLIPSIS

   ...strict
   -0.05

The published Box\ :sup:`2`\ EL-benchmark numbers of ELBE, Box\ :sup:`2`\ EL and
TransBox were obtained with a slack margin; see :doc:`../benchmarks/index`.

DELE
----

DELE [mashkova2024dele]_ (and its NeSy 2024 predecessor [mashkova2024nesy]_) is not an
embedding method but a way to train and evaluate any geometric
:math:`\mathcal{EL}^{++}` embedding with the help of the deductive closure. Its training
part is made of two composable regimes:

:class:`~mowl.training.AllNormalFormNegatives` ("+l" in DELE's tables)
   Negatives are sampled for all seven normal forms, not only :math:`C \sqsubseteq
   \exists R.D`, and trained with the negative losses of DELE, Section 4. The losses are
   implemented in :class:`~mowl.nn.ELEmModule`, :class:`~mowl.nn.ELBEModule` and
   :class:`~mowl.nn.BoxSquaredELModule` and switched on with ``all_negative_losses=True``
   (the regime does this for you); by default these modules are unchanged.

:class:`~mowl.training.ClosureFilteredNegatives` ("+n")
   A negative that is entailed by the training ontology is never used: it is re-sampled,
   or dropped if no non-entailed replacement is found. The filter applies to negatives
   from any sampler, so it can be combined with the regimes of other papers.

``dele_regime()`` composes both (the "+l+n" rows of DELE's Tables 3-6);
``dele_regime(filter_negatives=False)`` gives "+l". Use the regime when the ontology has
many non-trivial entailments (deep hierarchies, conjunctions, disjointness) and when the
model is evaluated on entailed knowledge as well as on novel axioms.

:class:`~mowl.training.ClosurePositives` adds entailed axioms as training positives. It is
**not** part of the papers and is provided as an extension.

Regularisation
--------------

Where the parameters of a model may lie is a choice separate from the representation and
from the loss. Publications mean three different things by it:

- a **penalty** added to the loss, such as the :math:`|\lVert c \rVert - 1|` of
  ELEmbeddings, which holds only approximately;
- a **projection** after every optimisation step, such as the normalisation of the entity
  vectors in TransE, which holds exactly;
- a **parametrisation** that cannot be violated, such as a box size stored as a logarithm.
  This is part of the model and stays there, and so do the constraints a geometry needs to
  be well defined (points inside the Poincaré ball, boxes inside the unit cube).

Penalties and projections are components (:mod:`mowl.training.regularization`) that the
regime :class:`~mowl.training.Regularization` applies to any model trained by the generic
loop, in addition to the regularisation term of the model's module or, with
``keep_default=False``, instead of it. ELEmbeddings with its centres exactly on the unit
sphere instead of pulled towards it:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ELEmbeddings, ELBE
   from mowl.training import (Regularization, SphereProjection, NormPenalty,
                              embedding_tables)

   model = ELEmbeddings(FamilyDataset(), embed_dim=10)
   model.training_regime = Regularization(SphereProjection(), keep_default=False)
   model.train(epochs=2)
   norms = model.module.class_embed.weight.norm(dim=1)
   print(bool((norms - 1).abs().max() < 1e-5))

.. testoutput::
   :options: +ELLIPSIS

   ...True

``Regularization(keep_default=False)`` alone trains without the penalty. The same components
work on other methods; ``embedding_tables`` lists what can be constrained:

.. testcode::

   boxes = ELBE(FamilyDataset(), embed_dim=10)
   print(embedding_tables(boxes))
   boxes.training_regime = Regularization(NormPenalty(parameters="class_embed", weight=0.1))
   boxes.train(epochs=2)

.. testoutput::
   :options: +ELLIPSIS

   ('class_embed', ...)...

The penalties are :class:`~mowl.training.NormPenalty` (norm near a target),
:class:`~mowl.training.MaxNormPenalty` (norm below a bound) and
:class:`~mowl.training.PowerPenalty` (L1, L2, N3); the projections are
:class:`~mowl.training.SphereProjection`, :class:`~mowl.training.BallProjection` and
:class:`~mowl.training.IntervalProjection`. A term a module adds inside the loss of every
axiom (EmEL++) is part of that loss and is not switched off.

The reference code
------------------

The published numbers of DELE's tables were not produced with the losses and the
training loop described above, but with the reference implementation
(bio-ontology-research-group/DELE), which differs from mOWL's models in ways the paper
does not mention. Two regimes make those differences explicit instead of hiding them in
the modules:

:class:`~mowl.training.DELECodeLosses`
   The reference's loss functions. It corrects the GCI3 losses of ELEmbeddings and ELBE
   (the filler's radius / offset is added, not subtracted) and gives ELBE slack margins
   and a GCI2 negative loss that penalises overlap instead of separation. For
   Box\ :sup:`2`\ EL the positive losses return the unsquared scores of the reference
   module, so that the validation loss is the reference's.

:class:`~mowl.training.DELECodeLoss`
   How the reference training loops combine the per-axiom losses: squares for every
   normal form of ELBE and Box\ :sup:`2`\ EL, for GCI2 with negatives only in
   ELEmbeddings. Without it mOWL's ELBE combination (squared means, negatives pulled to 1)
   trains the DELE negatives to be violated.

:class:`~mowl.training.DELECodeTraining`
   The reference's training design: cyclic mini-batches of 32,768 axioms per normal
   form (an epoch is :math:`\max_t \lceil n_t / b \rceil` steps), negatives corrupted
   with a class of the same kind on the GO & STRING data (``kind_prefixes``), entailed
   negatives dropped for the "+n" rows, and ReduceLROnPlateau with early stopping acting
   once per paper epoch. It composes ``DELECodeLosses`` and ``DELECodeLoss`` with the
   regimes above; the recipes of the ``dele-*`` benchmarks use it. ``negative_seed=0``
   restarts the negative draws from seed 0 at every step, as the reference does, so that
   every positive keeps one fixed negative.

.. testcode::

   from mowl.models import ELEmbeddings
   from mowl.training import DELECodeTraining

   model.training_regime = DELECodeTraining(   # baseline: GCI2 negatives only
       negatives="gci2", kind_prefixes=("4932.",))

The deductive closure
---------------------

:class:`~mowl.training.DeductiveClosure` computes DELE's approximation of the deductive
closure with respect to the normal forms: the full class hierarchy (computed by
mOWL's pure-Python EL++ completion or by ELK), and sound but incomplete syntactic rules
for the other normal forms (DELE, Algorithms 1 and 2). It can be queried, enumerated and
cached:

.. testcode::

   from mowl.training import DeductiveClosure

   closure = DeductiveClosure.from_model(model, reasoner="canonical")
   gci0 = closure.axioms("gci0")          # entailed A ⊑ B, as model indices
   train_gci0 = model.training_datasets["gci0"][:]
   assert closure.contains("gci0", train_gci0).all()   # training axioms are entailed

See :mod:`mowl.training.closure` for exactly which rules are applied, and
:doc:`/evaluation/deductive` for the matching evaluation regime.
