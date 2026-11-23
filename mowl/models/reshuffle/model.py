r"""RESHUFFLE on a projected ontology."""
from mowl.evaluation.inductive_triples import InductiveTripleCosts
from mowl.models.graph_kge.triple_model import GraphKGEModel, HitsEarlyStopping


class Reshuffle(GraphKGEModel):
    r"""RESHUFFLE [pavlovic2025]_ trained on the triples of a projection of the ontology.

    The named combination ``reshuffle`` of
    :func:`mowl.representation.library.other_graph.reshuffle`: **entities** are encoded by a
    graph network from fixed random binary feature matrices
    (:class:`~mowl.representation.library.other_graph.GraphEncodedPoints`), a **relation** is
    a soft coordinate reshuffling, a matrix
    (:class:`~mowl.representation.library.other_graph.ReshuffleRelations`), the
    **interaction** is :math:`-\lVert \mathrm{ReLU}(x_h R_r - x_t) \rVert_2`, the
    **objective** the margin ranking loss with margin 2, the **sampler** the uniform
    corruption of heads and tails (100 per positive), the **projection** the
    :class:`~mowl.projection.OWL2VecStarProjector`. The default hyper-parameters are those of
    the best configurations in the authors' repository (``l=25``, ``k=80``, four layers,
    Adam with learning rate 0.005).

    Ontology link (an adaptation: RESHUFFLE is a knowledge graph method for closed path
    rules). The graph is a projection of the ontology (``subclassof``, existential, type
    and role assertion edges), and the model is trained on all of its edges, transductively.
    Note that the graph is encoded in full at every training step; this is meant for small
    and medium graphs. Scores exist for ``gci0``, ``gci2`` and the assertion forms.

    **Inductive link prediction** (the setting of the paper). :meth:`inductive_costs` scores
    triples over a graph whose entities were not seen in training, encoding that graph
    with fresh random features; it is the interface of
    :class:`mowl.evaluation.inductive_ranking.InductiveSampledRankingRegime`, so the model
    runs the ``kg-inductive-*`` benchmarks. The paper trains on the training graph, stops
    early on Hits@10 of the validation triples of that graph and tests on the test graph:
    ``early_stopping={"frequency": 10, "patience": 10, "relative_delta": 0.01}`` is the
    authors' setting (every 10 epochs, stop when 10 evaluations did not improve Hits@10 by
    1%, then restore the best weights); it is the training regime
    :class:`~mowl.training.triples.HitsEarlyStopping`.

    :param l: Number of coordinates.
    :param k: Rows of an entity feature matrix.
    :param num_layers: Message passing layers.
    :param early_stopping: ``None`` (train for all epochs) or a dict with ``frequency``,
        ``patience`` and ``relative_delta`` of :class:`HitsEarlyStopping`, evaluated on the
        validation triples of the dataset (projected like the training ontology; edges with
        an entity or relation unknown to the training graph are dropped), filtered, all
        entities as candidates, Hits@10 over both sides.
    :param inverse_triples: Train as the authors' code does (the setting of all published
        results; the default): the inverse of every triple is an extra positive with its
        own relation matrix, and a head is predicted with the inverse matrix. ``False`` trains
        on the forward triples only and scores heads with the forward matrix, which predicts
        heads poorly.
    :param kge_kwargs: Further options: ``aggregation`` (``"max_self_noLoop"`` is RESHUFFLE
        without loop), ``row_function`` (``"square"`` is RESHUFFLE2), ``density``,
        ``feature_seed``, and ``loss_kwargs`` (``margin`` and ``reduction`` of the margin
        ranking loss, as in the authors' configuration files; the reduction is ``"mean"``
        where they do not give it).
    """

    METHOD = "reshuffle"

    #: ``GraphKGEModel(dataset, method="reshuffle", ...)`` builds the same combination; this
    #: class names it with the authors' defaults, the early stopping of the paper and the
    #: inductive scores.

    def __init__(self, dataset, l=25,  # noqa: E741
                 k=80, num_layers=4, learning_rate=5e-3, num_negs_per_pos=100,
                 early_stopping=None, inverse_triples=True, kge_kwargs=None, batch_size=1024,
                 projector=None, seed=0, model_filepath=None, device="cpu", **parts):
        self.create_inverse_triples = bool(inverse_triples)
        self._inductive = None
        options = dict(kge_kwargs or {})
        from mowl.representation.library.other_graph import configured_loss
        # the form of the authors' configuration files: the options of the margin ranking loss
        loss = configured_loss(options.pop("loss", None), options.pop("loss_kwargs", None),
                               "marginranking", parts)
        if loss is not None:
            parts["loss"] = loss
        unknown = set(options) - {"aggregation", "row_function", "density", "feature_seed"}
        if unknown:
            raise TypeError(f"Unknown Reshuffle options {sorted(unknown)}.")
        parts.pop("embed_dim", None)
        options.update(l=l, k=k, num_layers=num_layers, inverse_triples=bool(inverse_triples))
        options.update(parts)
        super().__init__(dataset, projector=projector, learning_rate=learning_rate,
                         batch_size=batch_size, num_negs_per_pos=num_negs_per_pos, seed=seed,
                         model_filepath=model_filepath, device=device, **options)
        self.early_stopping = None if early_stopping is None else dict(early_stopping)
        if self.early_stopping is not None:
            self.training_regime = HitsEarlyStopping(**self.early_stopping)

    def train(self, epochs=0):
        self._inductive = None
        return super().train(epochs=epochs)

    # -- inductive link prediction ------------------------------------------------------
    def inductive_costs(self, graph, rows, seed=0, target="tail"):
        """Cost of triples over a graph the model was not trained on (lower is more
        plausible): the norm :math:`\\lVert \\mathrm{ReLU}(x_h R_r - x_t)\\rVert_2` of the
        entity matrices after message passing over ``graph``.

        The graph's entities get fresh random features (drawn once per ``seed`` and
        cached), the learned relation matrices are those of training; relations are
        matched by IRI. Facts with a relation the model has not seen are not used for
        message passing, and a row with such a relation costs ``inf``. With
        ``inverse_triples=True`` and ``target="head"`` (the head of the rows is the
        corrupted entity), the triple is scored from the tail with the inverse matrix,
        :math:`\\lVert \\mathrm{ReLU}(x_t R_r^{-1} - x_h)\\rVert_2`, as is usual for head
        prediction with inverse triples. (The authors' code draws new random features at
        every forward pass.)

        :param graph: :class:`mowl.datasets.builtin.inductive_kg.InductiveGraph`.
        :param rows: ``(n, 3)`` ``[head, relation, tail]`` indices into the graph.
        :param seed: Seed of the random features.
        :param target: ``"tail"`` or ``"head"``: which entity of the rows is the candidate.
        """
        if self._inductive is None or self._inductive.module is not self.module:
            self._inductive = InductiveTripleCosts(
                self.module, self.triples_factory.relation_to_id, self.create_inverse_triples)
        return self._inductive(graph, rows, seed, target)
