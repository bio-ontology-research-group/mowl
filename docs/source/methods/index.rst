Ontology embedding methods
==========================

mOWL covers every way of turning an OWL ontology into vectors that a learning method can
use, not only geometric ones. This page explains how the methods are classified, what the
labels and guarantees in the :doc:`method catalogue <catalogue>` mean, and how to query
the catalogue from Python. The catalogue lists the methods mOWL implements and also
relevant published methods it does not implement, with the reason.

.. toctree::
   :maxdepth: 1

   catalogue
   regimes
   training_and_evaluation
   adding_a_method

.. toctree::
   :maxdepth: 1
   :caption: Methods added with the catalogue

   elkem
   emelpp
   regd
   mcmultibox
   octagon
   bcbox
   alcone
   e2r
   hit
   ont
   boxlite
   cate
   grounded
   kg_region


Representation, training and evaluation
---------------------------------------

A paper on an embedding method usually bundles three things. mOWL keeps them apart:

* the **representation** (the *method*): what concepts, roles and individuals are mapped
  to, and when an axiom counts as true in the embedding. This is what a
  :class:`~mowl.catalog.MethodCard` describes;
* the **training regime**: how the embedding is fitted, for example which negatives are
  sampled, which axioms are used as positives, how losses are combined, how the model is
  initialised and selected;
* the **evaluation regime**: how it is tested, for example how test axioms are chosen,
  which candidates are ranked, what is filtered and which metrics are reported.

Training and evaluation regimes have their own cards (:class:`~mowl.catalog.RegimeCard`,
listed on the :doc:`regimes` page). A method card names the regimes its publication used
in ``training`` and ``evaluation``. Keeping them apart lets any regime be combined with
any compatible method, and makes clear which regime a published number was obtained
with.


Families
--------

Every method belongs to one family (:class:`mowl.catalog.Family`):

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Family
     - What is embedded and how
     - Examples
   * - ``graph``
     - The ontology is projected to a graph, which is embedded with random walks and
       Word2Vec or with a knowledge graph embedding model.
     - OWL2Vec*, DL2Vec, projection + PyKEEN
   * - ``syntactic``
     - Axioms (and annotations) are sentences of a corpus; a word embedding is trained on
       it.
     - Onto2Vec, OPA2Vec
   * - ``geometric``
     - Concepts are regions of a vector space and axioms are relations between regions
       (inclusion, disjointness), learned by minimising a loss.
     - ELEmbeddings, ELBE, Box²EL, TransBox, MultiBoxEL, PrismEL
   * - ``constructive``
     - A model of the ontology is constructed, usually from its canonical model, and is
       exact by construction. No training, or training only of what the construction
       leaves open.
     - convex polyhedral model, interval-union model, CanonicalMixture
   * - ``fuzzy``
     - Concepts are fuzzy sets over (named and sampled) individuals, combined with fuzzy
       logical operators.
     - FALCON, DF-ALC
   * - ``order``
     - Only the subsumption order (or a lattice) is embedded.
     - order embeddings, hyperbolic cones, box lattices, CatE
   * - ``language_model``
     - Concept embeddings come from a (fine-tuned) language model applied to labels and
       definitions.
     - HiT, OnT
   * - ``algebraic``
     - Concepts are subspaces, distributions or other objects that are not regions.
     - E2R, BetaE

Geometric methods by region shape
---------------------------------

Inside the geometric family the ``region`` field gives the shape of a concept. The shape
decides which axioms can be represented exactly.

* **Balls** (ELEmbeddings, EmEL++). The intersection of two balls is not a ball, so
  a concept defined as a conjunction cannot be represented exactly.
* **Axis-parallel boxes** (ELBE, BoxEL, Box²EL, TransBox, BoxE). Closed under
  intersection, and existential restrictions with translations map boxes to boxes. Boxes
  have *Helly number 2* in every dimension: if every two boxes of a family intersect, all
  of them do. So a satisfiable ontology such as
  :math:`\{A \equiv X_1 \sqcap X_2, B \equiv X_2 \sqcap X_3, C \equiv X_1 \sqcap X_3\}`
  with :math:`A, B, C` pairwise disjoint has no box model in any dimension.
* **Unions of boxes** (MultiBoxEL, MultiELBE). Non-convex; every satisfiable
  :math:`\mathcal{ELHO}(\circ)` ontology has a model of this kind, given enough boxes.
* **Prisms** (PrismEL): an equilateral triangle times a box. Closed under
  intersection, with Helly number 3, so the three-concept obstruction above disappears.
* **Cones** (axis-aligned cones for :math:`\mathcal{ALC}`, ConE, AConE, hyperbolic
  entailment cones). Cones allow a geometric negation (polar or complement cone).
* **Octagons, convex polytopes, general convex regions.** General convex sets in
  :math:`\mathbb{R}^d` have Helly number :math:`d + 1`, so they are limited only when
  more than :math:`d + 1` concepts have to intersect in a particular way; the dimension
  needed can grow with the ontology.
* **Other shapes**: ellipsoids (EIKE), hyper-parallelograms over entity pairs
  (ExpressivE), intervals on a line (the interval-union model).

Methods also differ in the **logic** they cover (``logic`` field): most region methods
are defined for normalised :math:`\mathcal{EL}^{++}`; FALCON and the cone methods for
:math:`\mathcal{ALC}`; some theory papers for :math:`\mathcal{ELH}`, DL-Lite or
existential rules; knowledge graph methods only for triples and rule patterns; order
embeddings only for a hierarchy. Use ``list_methods(logic=...)`` to filter.


Status and origin
-----------------

``status`` (:class:`mowl.catalog.Status`) says what mOWL provides:

* ``learnable`` -- a trainable implementation exists;
* ``constructive`` -- an exact construction (no training) is implemented;
* ``catalogued`` -- documented only; ``not_implemented_reason`` says why (for example:
  defined only for knowledge graphs, no learning procedure published, logic not supported
  by mOWL's datasets).

``origin`` (:class:`mowl.catalog.Origin`) says how the code relates to the publication:
``reimplementation`` follows the published losses, ``adaptation`` means the paper gives a
geometry or construction and the training objective is mOWL's (published numbers are
then not comparable), and ``original`` means the method was developed in mOWL.


Guarantees and their conditions
-------------------------------

Each card lists theoretical statements (:class:`mowl.catalog.Guarantee`), each with its
source (paper and theorem) and the condition under which it holds. The kinds follow the
terminology of Bourgaux et al. (KR 2024):

* **soundness** -- the embedding (or one with zero loss) is a model of the ontology, so
  the ontology is satisfiable;
* **completeness** -- every satisfiable ontology of the fragment has an embedding of this
  kind;
* **weak faithfulness** -- everything that holds in the embedding is consistent with the
  ontology;
* **strong faithfulness** -- an axiom holds in the embedding if and only if the ontology
  entails it;
* **closure** -- the region family is closed under a constructor (intersection,
  existential restriction, role composition);
* **expressivity** -- another positive result;
* **limitation** -- a proven negative result, such as the Helly obstruction for boxes.

The condition matters as much as the statement:

* ``zero loss`` -- the statement is about an embedding whose training loss is exactly
  zero. Training rarely reaches zero, so it says what the loss is designed to achieve,
  not what a trained model satisfies;
* ``by construction`` -- it holds for every output of the construction, whatever the
  data or parameters;
* ``always`` -- it is a property of the representation itself.

A guarantee is recorded only if it was read in the cited source. Where mOWL checks it,
the card names the test (``verified_in_mowl``). The fields ``intersection_closed`` and
``helly`` summarise the two properties that most often decide whether conjunctions and
disjointness can be represented.


Querying the catalogue
----------------------

The catalogue is plain data and does not need the JVM, except to load an implementation.

.. testcode::

   from mowl.catalog import Family, get_method, list_methods

   card = get_method("elbe")
   print(card.name, card.family.value, card.region)

   for g in card.guarantees:
       assert g.source  # every statement names its source

   trainable_boxes = [c.key for c in list_methods(family=Family.GEOMETRIC, implemented=True)
                      if c.region == "box"]
   assert "elbe" in trainable_boxes

.. testoutput::

   ELBE geometric box

:func:`~mowl.catalog.load_method` (or :meth:`MethodCard.load
<mowl.catalog.MethodCard.load>`) imports the implementing class; for a catalogued-only
method it raises :class:`NotImplementedError` with the reason.
