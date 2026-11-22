r"""RESHUFFLE [pavlovic2025]_: a GNN over the graph whose messages reshuffle coordinates.

Every entity ``a`` has a feature matrix :math:`x_a \in \{0, 1\}^{k \times l}` (random
binary, fixed). A relation ``r`` is a matrix :math:`R_r \in [0, 1]^{l \times l}` obtained
from a raw :math:`(l + 1) \times l` parameter by a column-wise softmax whose last row is
dropped, so every column has total weight at most 1: a (soft) reshuffling of coordinates
in which a coordinate can also be dropped. A message passing layer updates

.. math::

    x_b \leftarrow \max\big(x_b,\; x_b R_{loop},\; \max_{(a, r, b)} x_a R_r,\;
    \max_{(b, r, a)} x_a R_{r^-}\big),

with the maximum taken coordinate-wise (the ``max_self_and_selfLoop`` aggregation of the
authors' code; ``max_self_noLoop`` drops the loop term). The same layer is applied
``num_layers`` times. A triple is scored by how far :math:`x_h R_r` is above :math:`x_t`:

.. math::

    s(h, r, t) = -\lVert \mathrm{ReLU}(x_h R_r - x_t) \rVert_2 ,

which is 0 exactly when :math:`x_h R_r \le x_t` coordinate-wise, the ordering constraint
of the region embedding. A closed path rule is captured when the matrix of the head
relation is the product of the matrices of the body relations (and the GNN propagates
along paths of the graph).

Differences from the authors' code (https://github.com/AleksVap/RESHUFFLE), which is
written for inductive link prediction on ILP-teru datasets:

* the paper's text initialises the features with probability 0.5 of a 1; the code (and
  the default here) uses 0.3085; ``density`` selects;
* the random features are drawn once per model (seeded) and kept, not redrawn at every
  forward pass;
* the model works on the graph it is given (transductive); the authors' inductive setting
  is available through :meth:`Reshuffle.score_on_graph`, which encodes any other graph
  over the same relations with fresh random features;
* inverse relations are separate parameters and are used only for message passing.
"""

import torch as th
from pykeen.losses import MarginRankingLoss
from pykeen.models import ERModel
from pykeen.nn import Embedding
from mowl.nn.graph.kge_interactions import (  # noqa: F401
    ReshuffleInteraction, propagate, random_features, relation_matrices,
    reshuffle_interaction)


class _FixedFeatures:
    """Initialiser that writes given feature matrices (a class, so that the model can be
    pickled)."""

    def __init__(self, features):
        self.features = features

    def __call__(self, x):
        return self.features.to(x.device, x.dtype)


class Reshuffle(ERModel):
    """RESHUFFLE as a PyKEEN model, trained by ``GraphPlusPyKEENModel``.

    :param triples_factory: The graph. With ``create_inverse_triples=True`` the model
        works as the authors' code does: PyKEEN trains the inverse triples
        :math:`(t, r^{-1}, h)` as positives, so the inverse matrices are relations of their
        own (relation ``r`` of the factory is matrix ``2r``, its inverse ``2r + 1``), and a
        head is predicted from the tail with the inverse matrix. Without, the inverse
        matrices only carry messages and both sides are scored with the forward matrix.
    :param l: Number of coordinates (relation matrices are ``l x l``).
    :param k: Number of rows of an entity feature matrix.
    :param num_layers: Message passing layers.
    :param aggregation: ``"max_self_and_selfLoop"`` or ``"max_self_noLoop"``.
    :param density: Probability of a 1 in the random features. The authors' code draws
        ``N(0, 1) > 0.5`` (0.3085, the default); the paper's text says 0.5.
    :param row_function: ``"softmax"`` (RESHUFFLE) or ``"square"`` (the RESHUFFLE2
        ablation of the paper).
    :param feature_seed: Seed of the random features.
    """

    loss_default = MarginRankingLoss
    loss_default_kwargs = dict(margin=2.0, reduction="mean")

    def __init__(self, *, triples_factory, l=25, k=80, num_layers=4,
                 aggregation="max_self_and_selfLoop", density=0.3085, feature_seed=0,
                 row_function="softmax", **kwargs):
        self.row_function = row_function
        self.l, self.k, self.num_layers = l, k, num_layers
        self.aggregation = aggregation
        self.density = density
        generator = th.Generator().manual_seed(feature_seed)
        features = random_features(triples_factory.num_entities, k, l, density, generator)

        super().__init__(
            triples_factory=triples_factory,
            interaction=ReshuffleInteraction(),
            entity_representations=Embedding,
            # the features are random and fixed: only the relation matrices are learned
            entity_representations_kwargs=dict(shape=(k, l), initializer=_FixedFeatures(features),
                                               trainable=False),
            relation_representations=Embedding(
                max_id=2 * triples_factory.real_num_relations, shape=(l + 1, l)),
            **kwargs)
        self.loop_raw = th.nn.Parameter(th.randn(l + 1, l) * 0.1)
        self.register_buffer("edge_index", th.stack(
            [triples_factory.mapped_triples[:, 0], triples_factory.mapped_triples[:, 2]]))
        self.register_buffer("edge_type", triples_factory.mapped_triples[:, 1].clone())

    def reset_parameters_(self):
        super().reset_parameters_()
        if getattr(self, "loop_raw", None) is not None:
            th.nn.init.normal_(self.loop_raw, std=0.1)

    def encode(self, features=None, edge_index=None, edge_type=None, all_relations=False):
        """Entity matrices ``(n, k, l)`` after message passing, and the relation matrices
        (``all_relations``: the ``2R`` matrices, relation ``i`` at ``2i`` and its inverse at
        ``2i + 1``; otherwise the ``R`` forward ones). Without arguments: the model's own
        features and graph."""
        features = self.entity_representations[0]() if features is None else features
        edge_index = self.edge_index if edge_index is None else edge_index
        edge_type = self.edge_type if edge_type is None else edge_type
        rel = relation_matrices(self.relation_representations[0](), self.row_function)
        loop = relation_matrices(self.loop_raw, self.row_function)
        x = propagate(features, rel, loop, edge_index, edge_type, self.num_layers,
                      self.aggregation)
        return x, (rel if all_relations else rel[0::2])

    def _get_representations(self, h, r, t, *, mode=None):
        x, rel = self.encode(all_relations=self.use_inverse_triples)
        return (x if h is None else x[h],
                rel if r is None else rel[r],
                x if t is None else x[t])

    def encode_graph(self, graph_triples, num_entities, seed=0, all_relations=False):
        """Entity matrices ``(n, k, l)`` and relation matrices of another graph over the
        same relations, with fresh random features (drawn with ``seed``): the inductive
        setting of the paper, where the test entities are not seen in training.

        :param graph_triples: ``(E, 3)`` ids of the facts, entities ``0 .. num_entities - 1``.
        """
        generator = th.Generator().manual_seed(seed)
        features = random_features(num_entities, self.k, self.l, self.density, generator)
        features = features.to(self.edge_index.device)
        graph_triples = graph_triples.to(self.edge_index.device)
        return self.encode(features, th.stack([graph_triples[:, 0], graph_triples[:, 2]]),
                           graph_triples[:, 1], all_relations=all_relations)

    def score_on_graph(self, triples, graph_triples, num_entities, seed=0):
        """Score ``triples`` (``(m, 3)`` ids) on another graph over the same relations.

        The graph ``graph_triples`` (``(E, 3)`` ids, entities ``0 .. num_entities - 1``)
        is encoded with fresh random features (:meth:`encode_graph`).
        """
        x, rel = self.encode_graph(graph_triples, num_entities, seed)
        triples = triples.to(self.edge_index.device)
        return reshuffle_interaction(x[triples[:, 0]], rel[triples[:, 1]], x[triples[:, 2]])
