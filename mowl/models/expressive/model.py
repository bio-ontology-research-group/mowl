r"""ExpressivE on a projected ontology."""
from mowl.models.graph_kge.triple_model import GraphKGEModel


class ExpressivE(GraphKGEModel):
    r"""ExpressivE [pavlovic2023]_ trained on the triples of a projection of the ontology.

    The named combination ``expressive`` of
    :func:`mowl.representation.library.other_graph.expressive`: **entities** are points, a
    **relation** is a hyper-parallelogram in the space of entity pairs
    :math:`\mathbb{R}^{2d}` (slopes, centres and widths for the head and the tail position),
    the **interaction** is the distance of Equation 3 of the paper with the constraints of the
    authors' code (:mod:`mowl.representation.library.graph_interactions`), the **objective** the
    self-adversarial negative sampling loss (margin 3, temperature 2), the **sampler** the
    uniform corruption of heads and tails, the **projection** the
    :class:`~mowl.projection.OWL2VecStarProjector`. ``model.module.parts`` names them and
    every one can be swapped (see :class:`~mowl.models.GraphKGEModel`).

    Ontology link (an adaptation: ExpressivE is a knowledge graph method). As for
    :class:`~mowl.models.BoxE`, the triples are those of a projection of the ontology
    (``subclassof``, existential, type and role assertion edges). Composition patterns of
    the paper hold for the projected triples only; an ontology's role chains are not read.
    Scores exist for ``gci0``, ``gci2`` and the assertion forms.

    :param min_denom: ``Dmin`` of the paper, lower bound of :math:`|1 - s^h s^t|` (default 0.5;
        with ``relations=`` another relation model only options that are given are passed on).
    :param p: Norm of the score.
    :param tanh_map: Map entities, centres and widths into ``[-1, 1]``.
    """

    METHOD = "expressive"

    def __init__(self, dataset, embed_dim=50, min_denom=None, p=None, tanh_map=None,
                 kge_kwargs=None, projector=None, learning_rate=1e-3, batch_size=1024,
                 num_negs_per_pos=None, seed=0, model_filepath=None, device="cpu", **parts):
        options = dict(kge_kwargs or {})
        unknown = set(options) - {"min_denom", "p", "tanh_map"}
        if unknown:
            raise TypeError(f"Unknown ExpressivE options {sorted(unknown)}.")
        given = dict(min_denom=min_denom, p=p, tanh_map=tanh_map)
        if "relations" not in parts:        # the defaults are those of its own relation model
            given = {k: v if v is not None else dict(min_denom=0.5, p=2, tanh_map=True)[k]
                     for k, v in given.items()}
        # with another relation model only the options that were given are passed on (and
        # refused if that relation model does not take them)
        options.update({k: v for k, v in given.items() if v is not None})
        options.update(parts)
        super().__init__(dataset, embed_dim=embed_dim, projector=projector,
                         learning_rate=learning_rate, batch_size=batch_size,
                         num_negs_per_pos=num_negs_per_pos, seed=seed,
                         model_filepath=model_filepath, device=device, **options)
