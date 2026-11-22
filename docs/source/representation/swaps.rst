Swapping a part of a method: what works
=======================================

For every method of :data:`mowl.representation.METHODS` and every part it has, this table
says what happens when the part is replaced *by name* through the constructor of the general
model of the method (:doc:`combine`): ``works`` (some replacement trains one epoch and gives
finite numbers), ``unavailable`` (every replacement tried is refused with
:class:`~mowl.representation.ConstructorUnavailable`: the combination cannot exist, and the
message names both sides), ``not supported`` (the method has no such part or no general model
trains it; the reason is below). The table is written by
``python -m tests.representation.test_user_level_swaps`` and checked by the tests.

Query methods (QueryEmbeddingModel)
-----------------------------------

.. list-table::
   :header-rows: 1

   * - method
   * - ``acone``
   * - ``betae``
   * - ``cone_qe``
   * - ``query2box``

Axiom methods (ComposedEL, ComposedALC)
---------------------------------------

.. list-table::
   :header-rows: 1

   * - method
     - classes
     - relations
     - individuals
     - measure
     - objective
     - objective_options
     - regularizers
   * - ``alcone``
     - unavailable
     - unavailable
     - works
     - unavailable
     - works
     - works
     - works
   * - ``bcbox``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``box2el``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``box_lattice``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``boxe``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``boxel``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``boxlite``
     - unavailable
     - works
     - works
     - works
     - works
     - works
     - works
   * - ``boxsel``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``cate``
     - works
     - works
     - works
     - works
     - works
     - works
     - works
   * - ``concept2box``
     - unavailable
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``dfalc``
     - unavailable
     - works
     - works
     - unavailable
     - works
     - works
     - works
   * - ``disk_embeddings``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``eike``
     - unavailable
     - works
     - unavailable
     - unavailable
     - works
     - works
     - works
   * - ``elbe``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``elem``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``elkem``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``emelpp``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``emelvar``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``expressive``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``faithel``
     - not supported
     - not supported
     - 
     - not supported
     - not supported
     - not supported
     - not supported
   * - ``falcon``
     - unavailable
     - unavailable
     - works
     - unavailable
     - works
     - works
     - works
   * - ``graph_kge``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``grounded``
     - not supported
     - not supported
     - 
     - not supported
     - not supported
     - not supported
     - not supported
   * - ``gumbel_box``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``hit``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``hyperbolic_cones``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``mcmultibox``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``multiboxel``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``multielbe``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``octagon``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``ont``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``order_embeddings``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``prism``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``regd``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``regd_elem``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``shadow_cones``
     - unavailable
     - works
     - 
     - works
     - works
     - works
     - works
   * - ``transbox``
     - works
     - works
     - 
     - works
     - works
     - works
     - works

Constructed methods (ConstructedModel)
--------------------------------------

.. list-table::
   :header-rows: 1

   * - method
   * - ``alcone-exact``
   * - ``attributed_dl``
   * - ``box_milp``
   * - ``canonical_model``
   * - ``convex_polyhedral``
   * - ``convex_regions``
   * - ``interval_union``
   * - ``least_model``
   * - ``temporal_cones``

Sequence methods (ComposedSequenceModel)
----------------------------------------

.. list-table::
   :header-rows: 1

   * - method
   * - ``dl2vec``
   * - ``onto2vec``
   * - ``opa2vec``
   * - ``owl2vec_star``
   * - ``taxonomy_walks``

Expression methods
------------------

.. list-table::
   :header-rows: 1

   * - method
     - classes
     - relations
     - measure
     - objective
     - objective_options
     - regularizers
   * - ``e2r``
     - not supported
     - not supported
     - not supported
     - not supported
     - not supported
     - not supported

Reasons
-------

* ``alcone``, classes, unavailable. Last replacement tried: octagon product: 'alcone' cannot take ['classes', 'individuals']: ConeMeasure compares ['FuzzySet'] regions; OctagonClasses gives OctagonProduct regions. (the method's other parts: {'classes': 'axis-aligned cone', 'relations': 'fuzzy relation', 'individuals': 'region', 'measure': 'fuzzy implication'
* ``alcone``, relations, unavailable. Last replacement tried: fuzzy relation: TruthMatrix relates the 6 elements of its domain; the region is a LogitSet over 12 elements. A class expression with an existential restriction cannot be evaluated by ExpressionGeometry.
* ``alcone``, measure, unavailable. Last replacement tried: masked bounds: 'alcone' cannot take ['measure']: MaskedBounds compares ['Box'] regions; HalfAxisClasses gives LogitSet regions. (the method's other parts: {'classes': 'axis-aligned cone', 'relations': 'fuzzy relation', 'individuals': 'region', 'measure': 'fuzzy implication', 'conjunction': 'fuzzy op
* ``boxlite``, classes, unavailable. Last replacement tried: sector cone: 'boxlite' cannot take ['classes']: SignedDistance compares ['Box', 'Point'] regions; ConeClasses gives SectorCone regions. (the method's other parts: {'classes': 'box', 'relations': 'head and tail boxes', 'individuals': 'point with bump', 'measure': 'exact', 'conjunction': 'exact', 'neg
* ``concept2box``, classes, unavailable. Last replacement tried: sector cone: 'concept2box' cannot take ['classes']: ByRegion compares ['Box', 'Point'] regions; ConeClasses gives SectorCone regions. (the method's other parts: {'classes': 'soft box', 'relations': 'neural network', 'individuals': 'none', 'measure': 'smoothed volume ratio', 'conjunction': 'exact'}).
* ``dfalc``, classes, unavailable. Last replacement tried: octagon product: 'dfalc' cannot take ['classes', 'individuals']: MeanViolation compares ['FuzzySet'] regions; OctagonClasses gives OctagonProduct regions. (the method's other parts: {'classes': 'fuzzy set', 'relations': 'fuzzy relation', 'individuals': 'region', 'measure': 'fuzzy implication', 'conj
* ``dfalc``, measure, unavailable. Last replacement tried: masked bounds: 'dfalc' cannot take ['measure']: MaskedBounds compares ['Box'] regions; TruthTableClasses gives FuzzySet regions. (the method's other parts: {'classes': 'fuzzy set', 'relations': 'fuzzy relation', 'individuals': 'region', 'measure': 'fuzzy implication', 'conjunction': 'fuzzy operator'
* ``eike``, classes, unavailable. Last replacement tried: sector cone: 'eike' cannot take ['classes']: LinkedMeasure compares ['Linked'] regions; ConeClasses gives SectorCone regions. (the method's other parts: {'classes': 'ellipsoid', 'relations': 'translation', 'individuals': 'region', 'measure': 'centre distance', 'conjunction': 'approximate'}).
* ``eike``, individuals, unavailable. Last replacement tried: sector cone: 'eike' cannot take ['individuals']: LinkedMeasure compares ['Linked'] regions; the individual model ConeClasses gives SectorCone regions. (the method's other parts: {'classes': 'ellipsoid', 'relations': 'translation', 'individuals': 'region', 'measure': 'centre distance', 'conjunction':
* ``eike``, measure, unavailable. Last replacement tried: point distance: 'eike' cannot take ['measure']: PointDistance compares ['Point'] regions; LinkedClasses gives Linked regions. (the method's other parts: {'classes': 'ellipsoid', 'relations': 'translation', 'individuals': 'region', 'measure': 'centre distance', 'conjunction': 'approximate'}).
* ``faithel``, not supported: its TBox is constructed by a reasoner from the ontology and its classes are anchored to it; ComposedEL builds from sizes. Train it with FaithEL.
* ``falcon``, classes, unavailable. Last replacement tried: octagon product: 'falcon' cannot take ['classes', 'individuals']: ConflictMeasure compares ['FuzzySet'] regions; OctagonClasses gives OctagonProduct regions. (the method's other parts: {'classes': 'fuzzy set', 'relations': 'neural network', 'individuals': 'region', 'measure': 'fuzzy implication', 'c
* ``falcon``, relations, unavailable. Last replacement tried: fuzzy relation: 'falcon' cannot take ['relations']: the combination {'classes': 'fuzzy set', 'relations': 'fuzzy relation', 'individuals': 'region', 'measure': 'fuzzy implication', 'conjunction': 'fuzzy operator', 'negation': 'fuzzy operator', 'disjunction': 'fuzzy operator', 'family': 'Product'} do
* ``falcon``, measure, unavailable. Last replacement tried: masked bounds: 'falcon' cannot take ['measure']: MaskedBounds compares ['Box'] regions; NetworkClasses gives FuzzySet regions. (the method's other parts: {'classes': 'fuzzy set', 'relations': 'neural network', 'individuals': 'region', 'measure': 'fuzzy implication', 'conjunction': 'fuzzy operator', 
* ``grounded``, not supported: GroundedELModule is a grounded program over a shared element domain, not an ELModule: ComposedEL refuses it; train it with GroundedEL.
* ``shadow_cones``, classes, unavailable. Last replacement tried: sector cone: 'shadow_cones' cannot take ['classes']: ShadowCone compares ['Point'] regions; ConeClasses gives SectorCone regions. (the method's other parts: {'classes': 'point', 'relations': 'none', 'individuals': 'none', 'measure': 'angle', 'conjunction': 'none'}).
* ``e2r``, not supported: its geometry reads ALC axioms through subspaces (a SubspaceGeometry, not an ExpressionGeometry): ComposedALC refuses it, ComposedEL refuses a method trained on expressions. Train it with E2R.
