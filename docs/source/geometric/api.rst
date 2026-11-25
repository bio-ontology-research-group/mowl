mowl.geometric API reference
============================

Normal forms and normalisation (TBox, RBox, and ABox)
-----------------------------------------------------

.. automodule:: mowl.geometric.normal_forms
   :members:
   :show-inheritance:


Completion and canonical model (TBox, RBox, and ABox)
-----------------------------------------------------

.. automodule:: mowl.geometric.completion
   :members:
   :show-inheritance:


Geometric region types
----------------------

.. automodule:: mowl.geometric.regions
   :members:
   :show-inheritance:


Geometric model checker
-----------------------

.. automodule:: mowl.geometric.checker
   :members:
   :show-inheritance:


Convex polyhedral model
-----------------------

.. automodule:: mowl.geometric.polyhedral
   :members:
   :show-inheritance:


Convex model of a finite model
------------------------------

.. automodule:: mowl.geometric.convex_regions
   :members:
   :show-inheritance:


Strong faithfulness
-------------------

.. automodule:: mowl.geometric.faithfulness
   :members:
   :show-inheritance:


Interval-union model
--------------------

.. automodule:: mowl.geometric.intervals
   :members:
   :show-inheritance:


Helly-satisfiability check
--------------------------

.. automodule:: mowl.geometric.helly
   :members:
   :show-inheritance:


Axis-parallel box model builder
-------------------------------

.. automodule:: mowl.geometric.boxes
   :members:
   :show-inheritance:


Embedding exports
-----------------

.. automodule:: mowl.geometric.export
   :members:
   :show-inheritance:


Exact-model box initialisation and anchor
-----------------------------------------

.. automodule:: mowl.geometric.init
   :members:
   :show-inheritance:


Shared multi-box utilities
--------------------------

.. automodule:: mowl.nn.el.multibox_utils
   :members:
   :show-inheritance:


MultiBoxEL neural module and losses
-----------------------------------

The module is the combination of parts that MultiBoxEL is
(:func:`~mowl.representation.library.multibox_methods.multiboxel`). The loss functions below are
those of the direct implementation (:mod:`mowl.baselines`), which the combination is tested against.

.. automodule:: mowl.nn.el.multiboxel.module
   :members:
   :show-inheritance:

.. automodule:: mowl.baselines.nn.el.multiboxel.losses
   :members:
   :show-inheritance:


MultiBoxEL model
----------------

.. automodule:: mowl.models.multiboxel.model
   :members:
   :show-inheritance:


MultiELBE neural module and losses
----------------------------------

The module is the combination of parts that MultiELBE is
(:func:`~mowl.representation.library.multibox_methods.multielbe`). The loss functions below are
those of the direct implementation (:mod:`mowl.baselines`).

.. automodule:: mowl.nn.el.multielbe.module
   :members:
   :show-inheritance:

.. automodule:: mowl.baselines.nn.el.multielbe.losses
   :members:
   :show-inheritance:


MultiELBE model
---------------

.. automodule:: mowl.models.multielbe.model
   :members:
   :show-inheritance:


Prism neural module and losses
------------------------------

.. automodule:: mowl.baselines.nn.el.prism.module
   :members:
   :show-inheritance:

.. automodule:: mowl.baselines.nn.el.prism.losses
   :members:
   :show-inheritance:


PrismEL model
-------------

.. automodule:: mowl.models.prism.model
   :members:
   :show-inheritance:

The neural module that :class:`~mowl.models.PrismEL` trains is
:class:`~mowl.nn.PrismModule`, the geometry of the method ``prism`` built from parts
(:func:`~mowl.representation.library.shapes_methods.prism`). The page above and the module
and losses of the previous section describe the earlier, hand-written implementation, which is
kept for comparison in :mod:`mowl.baselines`.


Canonical membership structure
------------------------------

.. automodule:: mowl.representation.library.membership_structure
   :members:
   :show-inheritance:


CanonicalMixture neural module
------------------------------

.. automodule:: mowl.nn.el.canonical_mixture.module
   :members:
   :show-inheritance:


CanonicalMixture model
----------------------

.. automodule:: mowl.models.canonical_mixture.model
   :members:
   :show-inheritance:


Soft component aggregation and diversity regularisation
-------------------------------------------------------

The parts: :func:`~mowl.representation.library.multibox_parts.soft_minimum` and the
aggregation of :class:`~mowl.representation.library.multibox_parts.UnionMeasure`, and the
regulariser :class:`~mowl.representation.library.multibox_objective.DiversityPenalty`. The
direct implementation, which the comparison tests read:

.. automodule:: mowl.baselines.nn.el.multiboxel.softagg
   :members:
   :show-inheritance:


Canonical-element point supervision and losses
----------------------------------------------

.. automodule:: mowl.datasets.el.points
   :members:
   :show-inheritance:

The parts: :mod:`mowl.representation.library.structural_points`. The direct
implementation:

.. automodule:: mowl.baselines.nn.el.multiboxel.pointloss
   :members:
   :show-inheritance:


Structural region algebra, definition tables, and losses
--------------------------------------------------------

.. automodule:: mowl.representation.library.structural_tables
   :members:
   :show-inheritance:

The losses are computed from the parts
(:class:`~mowl.representation.library.structural_reading.StructuralReading` and the slot
algebra of :mod:`mowl.representation.library.structural_regions`). The direct implementation:

.. automodule:: mowl.baselines.nn.el.multiboxel.structural_losses
   :members:
   :show-inheritance:


Exceptions
----------

.. automodule:: mowl.geometric.errors
   :members:
   :show-inheritance:
