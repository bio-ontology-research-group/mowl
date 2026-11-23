r"""BoxE on a projected ontology."""
from mowl.models.graph_kge.triple_model import GraphKGEModel


class BoxE(GraphKGEModel):
    r"""BoxE [abboud2020]_ trained on the triples of a projection of the ontology.

    The named combination ``boxe`` of :func:`mowl.representation.library.other_graph.boxe`:
    **entities** are points with a translational bump, a **relation** has one box per argument
    position (``r(h, t)`` holds when ``h`` bumped by ``t`` lies in the head box of ``r`` and
    ``t`` bumped by ``h`` in its tail box), the **interaction** is the distance of BoxE, the
    **objective** the self-adversarial negative sampling loss (margin 3, temperature 2), the
    **sampler** the uniform corruption of heads and tails, the **projection** the
    :class:`~mowl.projection.OWL2VecStarProjector`. ``model.module.parts`` names them, and
    every one can be swapped (``relations=``, ``classes=``, ``loss=``, ``sampler=``,
    ``projector=``, ``regularizers=``; see :class:`~mowl.models.GraphKGEModel`).

    Ontology link (an adaptation: BoxE is a knowledge graph method). The projector turns the
    ontology into triples; with the default projection these are ``C subclassof D``,
    ``C r D`` for ``C ⊑ ∃r.D``, ``a type C`` for ``C(a)`` and ``a r b`` for ``r(a, b)``.
    The paper's unary relations (types as boxes) become binary ``type`` edges, and class
    names are entities, so the guarantees of the paper (rule patterns) hold for the
    projected triples, not for the description logic semantics. Scores are available for
    ``gci0`` (``C ⊑ D``, the edge ``subclassof``), ``gci2`` (``C ⊑ ∃r.D``) and the two
    assertion forms.

    :param dataset: mOWL dataset.
    :param embed_dim: Entity dimension ``d``; relation boxes have dimension ``d`` per
        position.
    :param projector: The projection (default: OWL2Vec* projector).
    :param kge_kwargs: Options of the interaction: ``p``, ``power_norm``, ``tanh_map``; and,
        in the form of the published configurations, the loss: ``loss`` (``"nssa"``) and
        ``loss_kwargs`` (``margin``, ``adversarial_temperature``, ``reduction``).
    """

    METHOD = "boxe"

    def __init__(self, dataset, embed_dim=50, projector=None, learning_rate=1e-3,
                 batch_size=1024, num_negs_per_pos=None, seed=0, model_filepath=None,
                 device="cpu", kge_kwargs=None, **parts):
        options = dict(kge_kwargs or {})
        from mowl.representation.library.other_graph import configured_loss
        # the form of the published configurations: the loss by name and its options
        loss = configured_loss(options.pop("loss", None), options.pop("loss_kwargs", None),
                               "nssa", parts)
        if loss is not None:
            parts["loss"] = loss
        unknown = set(options) - {"p", "power_norm", "tanh_map"}
        if unknown:
            raise TypeError(f"Unknown BoxE options {sorted(unknown)}; the interaction takes "
                            f"p, power_norm and tanh_map (swap a part with classes=, "
                            f"relations=, loss=, sampler=).")
        options.update(parts)
        super().__init__(dataset, embed_dim=embed_dim, projector=projector,
                         learning_rate=learning_rate, batch_size=batch_size,
                         num_negs_per_pos=num_negs_per_pos, seed=seed,
                         model_filepath=model_filepath, device=device, **options)
