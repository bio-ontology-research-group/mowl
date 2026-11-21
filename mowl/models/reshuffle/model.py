r"""RESHUFFLE on a projected ontology."""
import os
import pathlib

import torch as th

from mowl.models.graph_kge.region_kge import RegionKGEModel
from mowl.nn.graph.reshuffle import Reshuffle as ReshuffleKGE


class Reshuffle(RegionKGEModel):
    r"""RESHUFFLE [pavlovic2025]_ trained on the triples of a projection of the ontology.

    Entities carry random binary feature matrices, relations are soft coordinate
    reshuffling matrices, and a GNN propagates the features along the graph so that closed
    path rules are captured (see :mod:`mowl.nn.graph.reshuffle`). The default
    hyper-parameters are those of the best configurations in the authors' repository
    (``l=25``, ``k=80``, four layers, margin ranking loss with margin 2, 100 negatives,
    Adam with learning rate 0.005).

    Ontology link (an adaptation: RESHUFFLE is a knowledge graph method for closed path
    rules). The graph is a projection of the ontology (``subclassof``, existential, type
    and role assertion edges), and the model is trained on all of its edges, transductively.
    Note that the graph is encoded in full at every training step; this is meant for small
    and medium graphs. Scores exist for ``gci0`` and ``gci2``.

    **Inductive link prediction** (the setting of the paper). :meth:`inductive_costs` scores
    triples over a graph whose entities were not seen in training, encoding that graph
    with fresh random features; it is the interface of
    :class:`mowl.evaluation.inductive_ranking.InductiveSampledRankingRegime`, so the model
    runs the ``kg-inductive-*`` benchmarks. The paper trains on the training graph, stops
    early on Hits@10 of the validation triples of that graph and tests on the test graph:
    ``early_stopping={"frequency": 10, "patience": 10, "relative_delta": 0.01}`` is the
    authors' setting (every 10 epochs, stop when 10 evaluations did not improve Hits@10 by
    1%, then restore the best weights), see :meth:`_make_stopper`.

    :param l: Number of coordinates.
    :param k: Rows of an entity feature matrix.
    :param num_layers: Message passing layers.
    :param early_stopping: ``None`` (train for all epochs) or a dict with ``frequency``,
        ``patience`` and ``relative_delta`` of PyKEEN's ``EarlyStopper``, evaluated on the
        validation triples of the dataset (projected like the training ontology; edges with
        an entity or relation unknown to the training graph are dropped), filtered, all
        entities as candidates, Hits@10 over both sides.
    :param inverse_triples: Train as the authors' code does (the setting of all published
        results; the default): the inverse of every triple is an extra positive with its
        own relation matrix, and a head is predicted with the inverse matrix
        (``create_inverse_triples=True``). ``False`` trains on the forward triples only and
        scores heads with the forward matrix, which predicts heads poorly.
    :param kge_kwargs: Further arguments of the PyKEEN model, e.g. ``aggregation``
        (``"max_self_noLoop"`` is RESHUFFLE without loop), ``row_function``
        (``"square"`` is RESHUFFLE2), ``density``.
    """

    KGE_CLASS = ReshuffleKGE

    def __init__(self, dataset, l=25,  # noqa: E741
                 k=80, num_layers=4, learning_rate=5e-3,
                 num_negs_per_pos=100, early_stopping=None, inverse_triples=True,
                 kge_kwargs=None, **kwargs):
        self.create_inverse_triples = bool(inverse_triples)
        self.early_stopping = None if early_stopping is None else dict(early_stopping)
        self._inductive_cache = None
        self._best_weights = None
        kge_kwargs = dict(kge_kwargs or {})
        kge_kwargs.update(l=l, k=k, num_layers=num_layers)
        kwargs.pop("embed_dim", None)
        super().__init__(dataset, learning_rate=learning_rate,
                         num_negs_per_pos=num_negs_per_pos, kge_kwargs=kge_kwargs, **kwargs)

    @staticmethod
    def _default_kge_kwargs(embed_dim):
        return {}

    # -- early stopping (the paper's setting) ------------------------------------------
    def _make_stopper(self):
        if self.early_stopping is None:
            return None
        if self.dataset.validation is None:
            raise ValueError("early_stopping needs validation triples (dataset.validation).")
        from pykeen.evaluation import RankBasedEvaluator
        from pykeen.stoppers import EarlyStopper
        from pykeen.triples import TriplesFactory
        nodes, relations = self.graph_node_to_id, self.graph_relation_to_id
        edges = [e for e in self.projector.project(self.dataset.validation)
                 if e.src in nodes and e.dst in nodes and e.rel in relations]
        if not edges:
            raise ValueError("The validation ontology gives no edge over the training graph.")
        triples = th.tensor([[nodes[e.src], relations[e.rel], nodes[e.dst]] for e in edges])
        validation = TriplesFactory(triples, entity_to_id=self.triples_factory.entity_to_id,
                                    relation_to_id=self.triples_factory.relation_to_id,
                                    create_inverse_triples=self.create_inverse_triples)
        self._best_weights = f"{self.model_filepath}.best"
        return EarlyStopper(model=self._kge_method, evaluator=RankBasedEvaluator(),
                            training_triples_factory=self.triples_factory,
                            evaluation_triples_factory=validation,
                            best_model_path=pathlib.Path(self._best_weights),
                            **self.early_stopping)

    def train(self, epochs=0):
        self._inductive_cache = None
        try:
            return super().train(epochs=epochs)
        finally:
            if self._best_weights is not None and os.path.exists(self._best_weights):
                os.remove(self._best_weights)

    # -- inductive link prediction ------------------------------------------------------
    @th.no_grad()
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
        :math:`\\lVert \\mathrm{ReLU}(x_t R_r^{-1} - x_h)\\rVert_2`, as PyKEEN does for head
        prediction with inverse triples. (The authors' code draws new random features at
        every forward pass.)

        :param graph: :class:`mowl.datasets.builtin.inductive_kg.InductiveGraph`.
        :param rows: ``(n, 3)`` ``[head, relation, tail]`` indices into the graph.
        :param seed: Seed of the random features.
        :param target: ``"tail"`` or ``"head"``: which entity of the rows is the candidate.
        """
        if self._kge_method is None:
            raise AttributeError("The model is not trained or loaded.")
        if target not in ("head", "tail"):
            raise ValueError("target must be 'head' or 'tail'.")
        inverse = self.create_inverse_triples
        known = self.triples_factory.relation_to_id
        to_model = th.tensor([known.get(r, -1) for r in graph.relations], dtype=th.long)
        key = (id(graph), int(seed), len(graph.triples))
        if self._inductive_cache is None or self._inductive_cache[0] != key:
            self._kge_method.eval()
            facts = graph.triples
            keep = to_model[facts[:, 1]] >= 0
            facts = th.stack([facts[keep, 0], to_model[facts[keep, 1]], facts[keep, 2]], dim=1)
            x, rel = self._kge_method.encode_graph(facts, len(graph.entities), seed=seed,
                                                   all_relations=inverse)
            self._inductive_cache = (key, x, rel)
        _, x, rel = self._inductive_cache
        rows = rows.cpu()
        relation = to_model[rows[:, 1]]
        out = th.full((len(rows),), float("inf"))
        ok = relation >= 0
        if ok.any():
            from mowl.nn.graph.reshuffle.module import reshuffle_interaction
            device = x.device
            h, r, t = rows[ok, 0].to(device), relation[ok].to(device), rows[ok, 2].to(device)
            if inverse and target == "head":
                score = reshuffle_interaction(x[t], rel[2 * r + 1], x[h])
            else:
                score = reshuffle_interaction(x[h], rel[2 * r if inverse else r], x[t])
            out[ok] = -score.cpu()
        return out
