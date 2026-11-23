Swapping a part of a method: what works
=======================================

For every method of :data:`mowl.representation.METHODS` and every part it has, this table
says what happens when the part is replaced *by name* through the constructor of the general
model of the method (:doc:`combine`: ``ComposedEL``, ``ComposedALC``, ``ComposedSubspace``,
``QueryEmbeddingModel``, ``ComposedSequenceModel``, ``ConstructedModel``, and
``GraphKGEModel`` for the knowledge graph models, whose columns are marked ``(graph)`` or
are the loss and the sampler of triples). The cells are

``works``
   at least one replacement is accepted, is the part of the model, **changes the losses (or
   the parameters that receive gradient) compared with the method as it is**, and trains two
   epochs to finite numbers;
``unavailable``
   every replacement tried is refused at build time with
   :class:`~mowl.representation.ConstructorUnavailable`, naming both sides. The reason that
   this is right is written below and in the test (``REFUSES_EVERY``): a method that refuses
   every candidate of a part without a written reason fails the test;
``not supported``
   the method has no such part, or no general model trains it (the reason is below).

A replacement that is accepted but would change nothing (a relation model that no trained
form reads, an objective option that no loss reads, an objective of a triple model that is
trained with a loss) is refused at build time, never ignored. The ALC methods ``falcon``,
``dfalc`` and ``alcone`` are checked with expected failures until their objective is
validated at build time. The table is written by
``python -m tests.representation.test_user_level_swaps`` and checked by the tests.

Query methods (QueryEmbeddingModel)
-----------------------------------

.. list-table::
   :header-rows: 1

   * - method
     - relations
     - intersection
     - distance
     - negation
   * - ``acone``
     - not supported
     - not supported
     - not supported
     - works
   * - ``betae``
     - not supported
     - works
     - works
     - works
   * - ``cone_qe``
     - works
     - not supported
     - not supported
     - works
   * - ``query2box``
     - not supported
     - works
     - works
     - unavailable

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
     - classes (graph)
     - relations (graph)
     - loss
     - sampler
     - regularizers (graph)
   * - ``alcone``
     - works
     - works
     - works
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``bcbox``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``box2el``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``box_lattice``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``boxe``
     - works
     - works
     - 
     - works
     - unavailable
     - unavailable
     - works
     - works
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
     - 
     - 
     - 
     - 
     - 
   * - ``boxlite``
     - unavailable
     - works
     - works
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``boxsel``
     - works
     - works
     - 
     - works
     - works
     - unavailable
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``canonical_mixture``
     - unavailable
     - unavailable
     - unavailable
     - unavailable
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``cate``
     - works
     - works
     - works
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``concept2box``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``dfalc``
     - works
     - works
     - works
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``disk_embeddings``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``eike``
     - unavailable
     - works
     - unavailable
     - unavailable
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``elbe``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``elem``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``elkem``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``emelpp``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``emelvar``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``expressive``
     - works
     - works
     - 
     - works
     - unavailable
     - unavailable
     - works
     - works
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
     - 
     - 
     - 
     - 
     - 
   * - ``falcon``
     - unavailable
     - unavailable
     - works
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``gda_projection``
     - works
     - works
     - 
     - works
     - unavailable
     - unavailable
     - works
     - works
     - works
     - works
     - works
     - works
   * - ``graph_kge``
     - works
     - works
     - 
     - works
     - unavailable
     - unavailable
     - works
     - works
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
     - 
     - 
     - 
     - 
     - 
   * - ``gumbel_box``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``hit``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``hyperbolic_cones``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``mcmultibox``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``multiboxel``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``multielbe``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``octagon``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``ont``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``order_embeddings``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``prism``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``regd``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``regd_elem``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``reshuffle``
     - not supported
     - not supported
     - 
     - not supported
     - not supported
     - not supported
     - not supported
     - not supported
     - not supported
     - works
     - works
     - works
   * - ``shadow_cones``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 
   * - ``transbox``
     - works
     - works
     - 
     - works
     - works
     - works
     - works
     - 
     - 
     - 
     - 
     - 

Constructed methods (ConstructedModel)
--------------------------------------

.. list-table::
   :header-rows: 1

   * - method
     - source
     - reading
   * - ``alcone-exact``
     - works
     - works
   * - ``attributed_dl``
     - unavailable
     - works
   * - ``box_milp``
     - unavailable
     - unavailable
   * - ``canonical_model``
     - works
     - works
   * - ``convex_polyhedral``
     - works
     - works
   * - ``convex_regions``
     - works
     - works
   * - ``interval_union``
     - works
     - works
   * - ``least_model``
     - works
     - works
   * - ``temporal_cones``
     - unavailable
     - works

Sequence methods (ComposedSequenceModel)
----------------------------------------

.. list-table::
   :header-rows: 1

   * - method
     - measure
     - view
     - walker
     - source
     - encoder
     - reading
     - scorer
   * - ``dl2vec``
     - works
     - works
     - works
     - works
     - works
     - not supported
     - works
   * - ``onto2vec``
     - works
     - works
     - not supported
     - works
     - works
     - not supported
     - works
   * - ``opa2vec``
     - works
     - works
     - not supported
     - works
     - works
     - not supported
     - works
   * - ``owl2vec_star``
     - works
     - works
     - works
     - works
     - works
     - not supported
     - works
   * - ``taxonomy_walks``
     - works
     - works
     - works
     - works
     - works
     - not supported
     - works

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
     - works
     - unavailable
     - unavailable
     - unavailable
     - works
     - works

Reasons
-------

* ``acone``, not supported: acone has no relations option: it is fixed by the method
* ``acone``, not supported: the builder knows one intersection of cones (attention) and takes any other as an instance
* ``acone``, not supported: the builder knows one distance of cones and takes any other as an instance
* ``betae``, not supported: betae has no relations option: it is fixed by the method
* ``cone_qe``, not supported: the builder knows one intersection of cones (attention) and takes any other as an instance
* ``cone_qe``, not supported: the builder knows one distance of cones and takes any other as an instance
* ``query2box``, not supported: query2box has no relations option: it is fixed by the method
* ``query2box``, negation, unavailable: the library has no negation of boxes (Query2Box has none); the negations of BetaE and ConE act on other region families and are refused with both sides named
* ``boxe``, objective, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``boxe``, objective_options, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``boxlite``, classes, unavailable: the variables of the BoxLitE programme are the lower and upper corner tables of its own BoundsBoxes; no other class model has corner tables, and the geometry refuses one without them
* ``boxsel``, objective_options, unavailable: the only option it has is the margin, and its volume measure ignores a margin (BoxEL and BoxSEL have none)
* ``canonical_mixture``, classes, unavailable: the measure reads parts of its own region kind (SparseElementSets), which no other entry of the registry provides; every other class model is refused with the two sides named
* ``canonical_mixture``, relations, unavailable: its relations are canonical roles over element sets; geometric relation models map geometric regions and refuse sparse sets
* ``canonical_mixture``, individuals, unavailable: the measure reads parts of its own region kind (SparseElementSets), which no other entry of the registry provides; every other class model is refused with the two sides named
* ``canonical_mixture``, measure, unavailable: ProbabilityMass is the only measure of sparse element sets (the regions are sets of finite-model elements, not geometric regions)
* ``eike``, classes, unavailable: the measure reads parts of its own region kind (an ellipsoid and a vector, Linked), which no other entry of the registry provides; every other class model is refused with the two sides named
* ``eike``, individuals, unavailable: the measure reads parts of its own region kind (Linked), which no other entry of the registry provides; every other class model is refused with the two sides named
* ``eike``, measure, unavailable: LinkedMeasure is the only measure that compares Linked regions; every other measure compares one region kind
* ``expressive``, objective, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``expressive``, objective_options, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``faithel``, not supported: its TBox is constructed by a reasoner from the ontology and its classes are anchored to it; ComposedEL builds from sizes. Train it with FaithEL.
* ``falcon``, classes, unavailable: FALCON evaluates class expressions on an open domain of element vectors (Domain, completed by sampled anonymous elements): only its own network class model evaluates on that domain; every other class model is refused with the reason (build time)
* ``falcon``, relations, unavailable: FALCON's relations are networks over pairs of element vectors of the domain and its sampled ABox objective needs a relation model: every geometric relation model, and none, is refused (build time)
* ``gda_projection``, objective, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``gda_projection``, objective_options, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``graph_kge``, objective, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``graph_kge``, objective_options, unavailable: a model of triples is trained with a TripleLoss (loss= of GraphKGEModel), no cost of its geometry reads an Objective: replacing it would change nothing
* ``grounded``, not supported: GroundedELModule is a grounded program over a shared element domain, not an ELModule: ComposedEL refuses it; train it with GroundedEL.
* ``reshuffle``, not supported: it is defined on a graph (its entities are encoded by message passing over the edges): ComposedEL has no edges to give it. Swap its parts through GraphKGEModel.
* ``reshuffle``, not supported: reshuffle encodes its entities with a graph network that uses the matrices of its own relations
* ``attributed_dl``, source, unavailable: the data of the method is an attributed ontology, which no other source reads (each source reads one kind of data and refuses another with the two sides named)
* ``box_milp``, source, unavailable: the only reading of a box model is 'boxes'; a finite model cannot in general be read as single boxes (boxes have the Helly property), so every other source is refused with both sides named
* ``box_milp``, reading, unavailable: a box model is a set of boxes, it has no domain elements for the cell readings to place: only the 'boxes' reading reads it
* ``temporal_cones``, source, unavailable: the data of the method is a temporal ontology, which no other source reads (each source reads one kind of data and refuses another with the two sides named)
* ``dl2vec``, not supported: one reading is registered (the vector of the entity's token): it is the method's own
* ``onto2vec``, not supported: onto2vec reads sentences, there are no walks to walk: give a view too (view=...), then walker= works
* ``onto2vec``, not supported: one reading is registered (the vector of the entity's token): it is the method's own
* ``opa2vec``, not supported: opa2vec reads sentences, there are no walks to walk: give a view too (view=...), then walker= works
* ``opa2vec``, not supported: one reading is registered (the vector of the entity's token): it is the method's own
* ``owl2vec_star``, not supported: one reading is registered (the vector of the entity's token): it is the method's own
* ``taxonomy_walks``, not supported: one reading is registered (the vector of the entity's token): it is the method's own
* ``e2r``, relations, unavailable: E2R's relations map pairs of vectors (a relation is a vector of the pair space, dimension 2d); the registered 'subspace' relation model is the method's own, and every other registered relation model works in the dimension of one class region and does not evaluate against the pair equations (refused 
* ``e2r``, measure, unavailable: the measure of E2R is the residual of a point against a subspace; every other measure compares another region kind or has no residual (refused when the model is built)
* ``e2r``, objective, unavailable: the losses of E2R are the equations of its axiom kinds (SubspaceObjective.kind_cost): a general Objective has none, and a SubspaceObjective is the method's own

Cross-method swaps among the region-based EL methods
----------------------------------------------------

The table above says whether *some* replacement of a part works. This one counts every
cross-method swap among the region-based EL methods (balls, boxes, octagons, prisms,
ellipsoids, unions of boxes, cones and orders: ``REGION_METHODS`` of
``tests/representation/cross_method_swaps.py``): method A built through ``ComposedEL`` with
the class model, relation model, measure or objective of method B, once per distinct
replacement. ``works``: accepted, the losses change, two epochs train to finite numbers;
``refused``: :class:`~mowl.representation.ConstructorUnavailable` at build time with the
reason of the part that cannot do it; ``same losses``: an objective or measure that is read
and gives the costs of the method's own on these forms; ``no effect`` and ``bug`` (an
accepted part that changes nothing, any other error) must stay at zero. Before the parts
declared their operations (:mod:`mowl.representation.operations`), 361 of the 1628 cases
worked, and 289 of the refusals were errors of parts that the general model wrapped as
refusals. The counts are written by the generator of this page:

.. list-table::
   :header-rows: 1

   * - part replaced
     - works
     - refused
     - same losses
     - no effect
     - bug
     - cases
   * - classes
     - 110
     - 349
     - 0
     - 0
     - 0
     - 459
   * - relations
     - 55
     - 169
     - 0
     - 0
     - 0
     - 224
   * - measure
     - 109
     - 347
     - 3
     - 0
     - 0
     - 459
   * - objective
     - 359
     - 99
     - 28
     - 0
     - 0
     - 486
   * - all
     - 633
     - 964
     - 31
     - 0
     - 0
     - 1628

Most refusals are of a class model or measure of another region family: a measure of
boxes cannot compare balls, and no exact view of one family as the other exists (exact views
are used: a point is a box of no width, a box is an octagon product). The hierarchy methods
train only ``C ⊑ D``, so no relation model they are given would be read. By category:

.. list-table::
   :header-rows: 1

   * - why refused
     - classes
     - relations
     - measure
     - objective
   * - the measure compares another region family
     - 336
     - 9
     - 322
     - 0
   * - no trained form reads a relation model
     - 0
     - 96
     - 0
     - 0
   * - the relation model does not map these regions
     - 1
     - 58
     - 6
     - 12
   * - an operation the part does not provide
     - 0
     - 5
     - 8
     - 47
   * - a margin the measure cannot read
     - 0
     - 0
     - 7
     - 35
   * - the method's training regime or solver needs its own part
     - 7
     - 0
     - 0
     - 4
   * - other
     - 5
     - 1
     - 4
     - 1
