"""The parts of a method, as a vocabulary.

A publication on ontology embeddings fixes several things at once. Training and evaluation
are described by regimes (:mod:`mowl.training`, :mod:`mowl.evaluation`); what is left, the
*representation*, still consists of parts that the literature varies one at a time:

* the :class:`View` of the ontology the method is given,
* the :class:`Space` in which things live,
* what a class is (:class:`ClassModel`), what a relation is (:class:`RelationModel`) and
  what an individual is (:class:`IndividualModel`),
* how compound classes are obtained (:class:`Construction` of conjunction, existential
  restriction, negation and disjunction),
* how "inside" is scored (:class:`Measure`), and
* how the embedding of a name is produced (:class:`Parametrisation`).

A :class:`Representation` states these for one method. The catalogue has one for every
method (:attr:`mowl.catalog.MethodCard.representation`); the parts that exist as code of
their own are in :mod:`mowl.representation`.
"""

from dataclasses import dataclass
from enum import Enum


class _Part(str, Enum):
    def __str__(self):
        return self.value


class View(_Part):
    """What the method is given: the form in which it sees the ontology."""

    #: The axioms of the ontology in normal form (the seven EL normal forms, role axioms).
    NORMAL_FORMS = "normal-form axioms"
    #: Axioms with arbitrarily nested class expressions.
    CLASS_EXPRESSIONS = "class expressions"
    #: A graph obtained by a projection (taxonomy, DL2Vec, OWL2Vec*, categorical, ...).
    GRAPH = "graph"
    #: Sequences of tokens: random walks on a graph, or axioms written as sentences.
    CORPUS = "corpus"
    #: Labels and verbalised axioms, read by a language model.
    TEXT = "text"
    #: Queries (trees of projection, intersection, negation) with their answers.
    QUERIES = "queries"
    #: A canonical model, saturation or grounding computed from the ontology.
    CANONICAL_MODEL = "canonical model"
    #: The ontology as a whole, as constraints of a construction or optimisation problem.
    ONTOLOGY = "whole ontology"


class Space(_Part):
    """Where the embeddings live."""

    EUCLIDEAN = "euclidean"
    UNIT_CUBE = "unit cube"
    HYPERBOLIC = "hyperbolic"
    SPHERE = "sphere"
    #: Angles (a torus): cone sectors, rotations.
    ANGLES = "angles"
    #: Parameters of probability distributions.
    DISTRIBUTIONS = "distributions"
    #: A finite set of elements; classes are (fuzzy) subsets of it.
    FINITE_DOMAIN = "finite domain"
    #: The simplex spanned by the elements of a canonical model.
    SIMPLEX = "simplex"
    #: A (complex) vector space whose subspaces are the classes.
    SUBSPACES = "subspace lattice"


class ClassModel(_Part):
    """What a class is."""

    #: A point; only similarity is defined.
    POINT = "point"
    #: A point with the corner cone above it (order embeddings).
    ORDER_CONE = "order cone"
    BALL = "ball"
    BOX = "box"
    #: A box with smoothed (Gumbel or softplus) boundaries.
    SOFT_BOX = "soft box"
    MULTI_BOX = "union of boxes"
    OCTAGON = "octagon product"
    PRISM = "triangle x box"
    ELLIPSOID = "ellipsoid"
    #: A point of a curved space with the cone it spans (entailment and shadow cones).
    GEODESIC_CONE = "geodesic cone"
    #: A sector cone per coordinate (axis and aperture angles).
    SECTOR_CONE = "sector cone"
    #: An axis-aligned cone given by its half-axes.
    AXIS_CONE = "axis-aligned cone"
    DISK = "disk"
    SUBSPACE = "subspace"
    DISTRIBUTION = "distribution"
    #: Degrees of membership of the elements of a finite domain.
    FUZZY_SET = "fuzzy set"
    INTERVAL_UNION = "union of intervals"
    CONVEX = "convex region"
    #: Not learned: fixed by reasoning (a set of canonical elements).
    FIXED = "fixed by reasoning"


class RelationModel(_Part):
    """What a relation is: how the set of pairs it holds for is represented."""

    NONE = "none"
    #: A point or token; no set of pairs is defined.
    TOKEN = "token"
    #: One shift vector: ``y = x + r``.
    TRANSLATION = "translation"
    #: A shift vector with a learned tolerance.
    TRANSLATION_TOLERANCE = "translation with tolerance"
    #: Scaling and shift per coordinate.
    AFFINE = "affine map"
    #: A box of shift vectors: ``y - x`` lies in a box.
    TRANSLATION_BOX = "box of translations"
    #: A head box and a tail box, with bump vectors of the arguments.
    HEAD_TAIL_BOXES = "head and tail boxes"
    #: A range box, an error box and an affine map (ELK-Em).
    AFFINE_BOXES = "affine map with range and error box"
    #: A region of the space of pairs (parallelogram, union of boxes, product region).
    PAIR_REGION = "region over pairs"
    ROTATION = "rotation"
    MATRIX = "matrix"
    NETWORK = "neural network"
    #: Degrees of membership of the pairs of a finite domain.
    FUZZY_RELATION = "fuzzy relation"
    SUBSPACE = "subspace"
    FIXED = "fixed by reasoning"


class IndividualModel(_Part):
    """What an individual is, and with it what the domain is."""

    NONE = "none"
    POINT = "point"
    #: A point with a bump vector (BoxE, BoxLitE).
    POINT_WITH_BUMP = "point with bump"
    #: A (small) region of the same kind as the classes.
    REGION = "region"
    #: A class with one element (a nominal), treated as any other class.
    NOMINAL = "one-element class"
    DISTRIBUTION = "distribution"
    #: An element of a finite domain that is sampled or enumerated.
    DOMAIN_ELEMENT = "element of a finite domain"
    #: Elements of a canonical model, or a distribution over them.
    CANONICAL_ELEMENT = "canonical element"


class Construction(_Part):
    """How the region of a compound class is obtained from the regions of its parts."""

    #: The class model is closed under the constructor: the result is computed exactly.
    EXACT = "exact"
    #: The result is replaced by a region of the class model that is not the exact one, or
    #: the constructor only appears inside a loss.
    APPROXIMATE = "approximate"
    #: A trained network produces the result.
    LEARNED = "learned"
    #: A fuzzy logic operator on degrees of membership.
    FUZZY = "fuzzy operator"
    #: The compound class is a name of its own (a lattice element, a text).
    REIFIED = "own embedding"
    NONE = "none"


class Measure(_Part):
    """How "inside" (and with it disjointness and membership) is scored."""

    #: Distance of centres against sizes (balls).
    CENTRE_DISTANCE = "centre distance"
    #: Comparison of the bounds in every coordinate (boxes, octagons, prisms).
    BOUNDS = "bounds per coordinate"
    VOLUME_RATIO = "volume ratio"
    SOFT_VOLUME_RATIO = "smoothed volume ratio"
    SAMPLED_VOLUME = "sampled volume"
    #: Distance to the boundary plus difference of depth (RegD).
    BOUNDARY_DEPTH = "boundary and depth"
    ORDER_VIOLATION = "order violation"
    #: Angle against aperture (cones).
    ANGLE = "angle"
    #: Distance of centres against radii on a manifold (disks).
    PROTRUSION = "protrusion"
    #: Distance and depth of points (HiT).
    DISTANCE_DEPTH = "distance and depth"
    DIVERGENCE = "divergence"
    FUZZY_IMPLICATION = "fuzzy implication"
    #: Distance of a point to a region.
    POINT_TO_REGION = "point-to-region distance"
    #: The score of a knowledge graph model for a triple.
    TRIPLE_SCORE = "triple score"
    SIMILARITY = "similarity"
    #: The statement holds or does not: an exact check.
    EXACT = "exact"


class Parametrisation(_Part):
    """How the embedding of a name is produced."""

    #: One row of a table per name.
    TABLE = "lookup table"
    #: Computed from the label or description by a language model.
    TEXT_ENCODER = "text encoder"
    #: Computed from the neighbourhood in a graph.
    GRAPH_ENCODER = "graph encoder"
    #: The solution of an optimisation problem or the output of a construction.
    SOLVER = "construction"
    #: Read off a reasoner's result.
    REASONER = "reasoner"


class Composable(_Part):
    """Whether the method is a combination of independent parts."""

    YES = "yes"
    #: Yes, except for a parameter shared between two parts.
    MOSTLY = "mostly"
    PARTLY = "partly"
    NO = "no"


def _tuple(value, kind):
    values = (value,) if isinstance(value, (str, Enum)) else tuple(value)
    return tuple(kind(v) for v in values)


@dataclass(frozen=True)
class Representation:
    """The parts of the representation of one method.

    Every field takes one member of its vocabulary or several (a method that comes with
    several class models, or measures inclusion in two ways).

    :param view: :class:`View`.
    :param space: :class:`Space`.
    :param classes: :class:`ClassModel`.
    :param relations: :class:`RelationModel`.
    :param individuals: :class:`IndividualModel`.
    :param measure: :class:`Measure`.
    :param conjunction: :class:`Construction` of :math:`C \\sqcap D`.
    :param existential: :class:`Construction` of :math:`\\exists r.C`.
    :param negation: :class:`Construction` of :math:`\\neg C`.
    :param disjunction: :class:`Construction` of :math:`C \\sqcup D`.
    :param parametrisation: :class:`Parametrisation`.
    :param composable: :class:`Composable`.
    :param notes: What does not fit the vocabulary.
    """

    view: tuple
    space: tuple
    classes: tuple
    relations: tuple
    individuals: tuple
    measure: tuple
    conjunction: Construction = Construction.NONE
    existential: Construction = Construction.NONE
    negation: Construction = Construction.NONE
    disjunction: Construction = Construction.NONE
    parametrisation: tuple = (Parametrisation.TABLE,)
    composable: Composable = Composable.NO
    notes: str = ""

    def __post_init__(self):
        for name, kind in _MULTI.items():
            object.__setattr__(self, name, _tuple(getattr(self, name), kind))
        for name in CONSTRUCTORS:
            object.__setattr__(self, name, Construction(getattr(self, name)))
        object.__setattr__(self, "composable", Composable(self.composable))

    @property
    def inductive(self):
        """Whether names that were not seen in training get an embedding (the embedding
        is computed from text or from the graph, not looked up)."""
        return any(p in (Parametrisation.TEXT_ENCODER, Parametrisation.GRAPH_ENCODER)
                   for p in self.parametrisation)

    @property
    def constructors(self):
        """Mapping from constructor name to its :class:`Construction`."""
        return {name: getattr(self, name) for name in CONSTRUCTORS}

    def has(self, part):
        """Whether ``part`` (a member of one of the vocabularies, or its value) is one of
        the parts of this representation."""
        for name in tuple(_MULTI) + CONSTRUCTORS + ("composable",):
            values = getattr(self, name)
            values = values if isinstance(values, tuple) else (values,)
            for v in values:
                if v is part or (not isinstance(part, Enum) and v.value == part):
                    return True
        return False

    def rows(self):
        """``(label, text)`` pairs for display."""
        def join(values):
            return ", ".join(v.value for v in values)
        rows = [("View of the ontology", join(self.view)), ("Space", join(self.space)),
                ("A class is", join(self.classes)), ("A relation is", join(self.relations)),
                ("An individual is", join(self.individuals)),
                ('"Inside" is scored by', join(self.measure)),
                ("Compound classes", "; ".join(
                    f"{label}: {getattr(self, name).value}" for name, label in _LABELS.items())),
                ("Embeddings come from", join(self.parametrisation)),
                ("Combination of independent parts", self.composable.value)]
        if self.notes:
            rows.append(("Remarks on the parts", self.notes))
        return rows


#: The four constructors a :class:`Representation` describes.
CONSTRUCTORS = ("conjunction", "existential", "negation", "disjunction")
_LABELS = {"conjunction": "and", "existential": "exists", "negation": "not",
           "disjunction": "or"}
_MULTI = {"view": View, "space": Space, "classes": ClassModel, "relations": RelationModel,
          "individuals": IndividualModel, "measure": Measure,
          "parametrisation": Parametrisation}
