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
from pykeen.nn.modules import Interaction


def relation_matrices(raw):
    """Soft reshuffling matrices from raw ``(..., l + 1, l)`` parameters: softmax over the
    rows of every column, then drop the last row. Shape ``(..., l, l)``."""
    return th.softmax(raw, dim=-2)[..., :-1, :]


def random_features(num_entities, k, l,  # noqa: E741
                    density=0.3085, generator=None):
    """Random binary features of shape ``(n, k, l)``. The default density is the
    probability that a standard normal exceeds 0.5, which is how the authors' code draws
    them."""
    return (th.rand(num_entities, k, l, generator=generator) < density).float()


def propagate(x, rel, loop, edge_index, edge_type, num_layers, aggregation="max_self_and_selfLoop"):
    """Run ``num_layers`` message passing layers.

    :param x: Features ``(n, k, l)``.
    :param rel: Relation matrices ``(2R, l, l)``; relation ``i`` is ``rel[2i]`` and its
        inverse ``rel[2i + 1]``.
    :param loop: The loop matrix ``(l, l)``.
    :param edge_index: ``(2, E)`` source and target of every triple.
    :param edge_type: ``(E,)`` relation index of every triple (0 .. R-1).
    """
    if aggregation not in ("max_self_and_selfLoop", "max_self_noLoop"):
        raise ValueError("aggregation must be 'max_self_and_selfLoop' or 'max_self_noLoop'.")
    src, dst = edge_index
    for _ in range(num_layers):
        def message(source, target, matrices):
            m = th.bmm(x[source], matrices)
            index = target.view(-1, 1, 1).expand_as(m)
            return th.zeros_like(x).scatter_reduce(0, index, m, reduce="amax")

        new = th.maximum(x, th.maximum(
            message(src, dst, rel[2 * edge_type]),
            message(dst, src, rel[2 * edge_type + 1])))
        if aggregation == "max_self_and_selfLoop":
            new = th.maximum(new, x @ loop)
        x = new
    return x


def reshuffle_interaction(h, r, t):
    """:math:`-\\lVert \\mathrm{ReLU}(h R - t) \\rVert_2` over the ``k x l`` entries."""
    return -th.norm(th.relu(h @ r - t), dim=(-2, -1), p=2)


class ReshuffleInteraction(Interaction):
    """The scoring function of RESHUFFLE. Relations arrive as softmaxed matrices."""

    entity_shape = ("kl",)
    relation_shape = ("dl",)

    def forward(self, h, r, t):
        return reshuffle_interaction(h, r, t)


class _FixedFeatures:
    """Initialiser that writes given feature matrices (a class, so that the model can be
    pickled)."""

    def __init__(self, features):
        self.features = features

    def __call__(self, x):
        return self.features.to(x.device, x.dtype)


class Reshuffle(ERModel):
    """RESHUFFLE as a PyKEEN model, trained by ``GraphPlusPyKEENModel``.

    :param triples_factory: The graph (without inverse triples).
    :param l: Number of coordinates (relation matrices are ``l x l``).
    :param k: Number of rows of an entity feature matrix.
    :param num_layers: Message passing layers.
    :param aggregation: ``"max_self_and_selfLoop"`` or ``"max_self_noLoop"``.
    :param density: Probability of a 1 in the random features.
    :param feature_seed: Seed of the random features.
    """

    loss_default = MarginRankingLoss
    loss_default_kwargs = dict(margin=2.0, reduction="mean")

    def __init__(self, *, triples_factory, l=25, k=80, num_layers=4,
                 aggregation="max_self_and_selfLoop", density=0.3085, feature_seed=0,
                 **kwargs):
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
                max_id=2 * triples_factory.num_relations, shape=(l + 1, l)),
            **kwargs)
        self.loop_raw = th.nn.Parameter(th.randn(l + 1, l) * 0.1)
        self.register_buffer("edge_index", th.stack(
            [triples_factory.mapped_triples[:, 0], triples_factory.mapped_triples[:, 2]]))
        self.register_buffer("edge_type", triples_factory.mapped_triples[:, 1].clone())

    def reset_parameters_(self):
        super().reset_parameters_()
        if getattr(self, "loop_raw", None) is not None:
            th.nn.init.normal_(self.loop_raw, std=0.1)

    def encode(self, features=None, edge_index=None, edge_type=None):
        """Entity matrices ``(n, k, l)`` after message passing, and the relation matrices.
        Without arguments: the model's own features and graph."""
        features = self.entity_representations[0]() if features is None else features
        edge_index = self.edge_index if edge_index is None else edge_index
        edge_type = self.edge_type if edge_type is None else edge_type
        rel = relation_matrices(self.relation_representations[0]())
        loop = relation_matrices(self.loop_raw)
        x = propagate(features, rel, loop, edge_index, edge_type, self.num_layers,
                      self.aggregation)
        return x, rel[0::2]

    def _get_representations(self, h, r, t, *, mode=None):
        x, rel = self.encode()
        return (x if h is None else x[h],
                rel if r is None else rel[r],
                x if t is None else x[t])

    def score_on_graph(self, triples, graph_triples, num_entities, seed=0):
        """Score ``triples`` (``(m, 3)`` ids) on another graph over the same relations.

        The graph ``graph_triples`` (``(E, 3)`` ids, entities ``0 .. num_entities - 1``)
        is encoded with fresh random features: this is the inductive setting of the paper,
        where the test entities are not seen in training.
        """
        generator = th.Generator().manual_seed(seed)
        features = random_features(num_entities, self.k, self.l, self.density, generator)
        features = features.to(self.edge_index.device)
        graph_triples = graph_triples.to(self.edge_index.device)
        triples = triples.to(self.edge_index.device)
        x, rel = self.encode(features, th.stack([graph_triples[:, 0], graph_triples[:, 2]]),
                             graph_triples[:, 1])
        return reshuffle_interaction(x[triples[:, 0]], rel[triples[:, 1]], x[triples[:, 2]])
