r"""BoxE on a projected ontology."""
from pykeen.models import BoxE as PyKEENBoxE

from mowl.baselines.models.graph_kge.region_kge import RegionKGEModel


class BoxE(RegionKGEModel):
    r"""BoxE [abboud2020]_ trained on the triples of a projection of the ontology.

    This is PyKEEN's :class:`pykeen.models.BoxE` used through
    :class:`~mowl.baselines.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel` (no
    reimplementation). Each relation has one box per argument position, each entity a point
    and a translational bump; ``r(h, t)`` holds when ``h`` bumped by ``t`` lies in the head
    box of ``r`` and ``t`` bumped by ``h`` in its tail box.

    Ontology link (an adaptation: BoxE is a knowledge graph method). The projector turns the
    ontology into triples; with the default
    :class:`~mowl.projection.OWL2VecStarProjector` these are ``C subclassof D``,
    ``C r D`` for ``C ⊑ ∃r.D``, ``a type C`` for ``C(a)`` and ``a r b`` for ``r(a, b)``.
    The paper's unary relations (types as boxes) become binary ``type`` edges, and class
    names are entities, so the guarantees of the paper (rule patterns) hold for the
    projected triples, not for the description logic semantics. Scores are available for
    ``gci0`` (``C ⊑ D``, the edge ``subclassof``) and ``gci2`` (``C ⊑ ∃r.D``).

    :param dataset: mOWL dataset.
    :param embed_dim: Entity dimension ``d``; relation boxes have dimension ``d`` per
        position.
    :param kge_kwargs: Further arguments of :class:`pykeen.models.BoxE` (``p``,
        ``power_norm``, ``tanh_map``, ``loss``, ...). Its default loss is the self-adversarial
        negative sampling loss with margin 3 and temperature 2.
    """

    KGE_CLASS = PyKEENBoxE
