import torch as th
import torch.nn as nn
import torch.nn.functional as F

import mowl.baselines.nn.el.concept2box.losses as L
from mowl.baselines.nn.el.elmodule import ELModule


class Concept2BoxModule(ELModule):
    r"""Concept2Box [huang2023]_ for an ontology viewed as a two-view knowledge graph.

    * **Ontology view** (classes): a triple is evaluated by the conditional probability
      :math:`\phi(h, r, t) = E[Vol(f_r(h) \cap f_r(t))] / E[Vol(f_r(t))]` of Gumbel boxes.
      :math:`\phi` is near 1 when the box of the *tail* lies inside the box of the *head*.
      ``gci0`` (``C ⊑ D``) is the triple ``(D, subclass, C)``. ``gci2`` (``C ⊑ ∃r.D``)
      is, by default (``gci2_reading="axiom"``), the triple ``(D, r, C)``: in the context of
      ``r`` the box of ``C`` lies inside the box of ``D``, as for ``gci0``. The paper's own
      triples ``(h, r, t)`` such as ``(Artist, lives_in, City)`` put the tail inside the head
      (:math:`\phi` is the probability of the head given the tail); ``gci2_reading="triple"``
      reads ``C ⊑ ∃r.D`` as the paper's triple ``(C, r, D)``, the box of ``D`` inside the
      box of ``C``, which is the reverse of the axiom (the reading of earlier mOWL versions
      and of the recipes that reproduce the paper).
    * **Instance view** (individuals): role assertions ``r(a, b)`` are scored by a
      translational model, :math:`\lVert x_a + v_r - x_b \rVert`.
    * **Cross view**: a class assertion ``C(a)`` is scored by the vector-to-box distance
      from :math:`x_a` to the box of ``C``.

    The paper encodes ``f_r(Box(c))`` with BERT over the texts of the concept and the
    relation followed by a two-layer MLP. Here the text encoder is replaced by learned
    embeddings of the concept and of the relation (concatenated and passed through the same
    kind of two-layer MLP, split into centre and offset); so concept and relation interact
    in the box, as in the paper, but no text is used. The box of a concept in the cross view
    is the box in the context of a dedicated *null* relation.

    :param nb_classes: Number of classes (named and auxiliary).
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals (``None``: no ABox).
    :param embed_dim: Dimension of boxes and entity vectors.
    :param beta: Gumbel scale :math:`\beta`.
    :param alpha: Weight :math:`\alpha` of the inside distance (``0 < alpha < 1``).
    :param kge_margin: Margin :math:`\gamma^{KG}` of the instance view (paper: 0.3).
    :param cross_margin: Margin :math:`\gamma^{Cross}` of the cross view (paper: 0.15).
    :param hidden_dim: Hidden size of the MLP (default ``2 * embed_dim``).
    :param instance_weight: :math:`\lambda_1` of the overall loss.
    :param cross_weight: :math:`\lambda_2` of the overall loss.
    :param gci2_reading: ``"axiom"`` (default) or ``"triple"``, see above.

    The cost of a point inside a box in the cross view is not zero: the paper's distance
    (Equation 8) adds :math:`\alpha(c)` times the distance to the centre (the inside
    distance) to the outside distance, so that entities of a small concept are pulled to its
    centre. This is the published behaviour and is kept.
    """

    neg_capable_gcis = frozenset({"gci0", "gci2", "class_assertion",
                                  "object_property_assertion"})
    role_axiom_capable = False

    def __init__(self, nb_classes, nb_rels, nb_inds=None, embed_dim=32, beta=1.0,
                 alpha=0.5, kge_margin=0.3, cross_margin=0.15, hidden_dim=None,
                 instance_weight=1.0, cross_weight=1.0, gci2_reading="axiom"):
        super().__init__()
        if gci2_reading not in ("axiom", "triple"):
            raise ValueError("gci2_reading must be 'axiom' or 'triple'.")
        self.gci2_reading = gci2_reading
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        self.nb_classes, self.nb_rels, self.nb_inds = nb_classes, nb_rels, nb_inds
        self.embed_dim = embed_dim
        self.beta, self.alpha = beta, alpha
        self.kge_margin, self.cross_margin = kge_margin, cross_margin
        self.instance_weight, self.cross_weight = instance_weight, cross_weight
        hidden_dim = hidden_dim or 2 * embed_dim

        #: Index of the subclass relation and of the null relation in the relation
        #: embedding of the ontology view (after the object properties).
        self.subclass_id = nb_rels
        self.null_id = nb_rels + 1
        self.class_embed = nn.Embedding(nb_classes, embed_dim)
        self.ontology_rel_embed = nn.Embedding(nb_rels + 2, embed_dim)
        self.project = nn.Sequential(nn.Linear(2 * embed_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 2 * embed_dim))
        self.rel_embed = nn.Embedding(max(nb_rels, 1), embed_dim)
        self.ind_embed = nn.Embedding(nb_inds, embed_dim) if nb_inds else None
        for emb in (self.class_embed, self.ontology_rel_embed, self.rel_embed,
                    self.ind_embed):
            if emb is not None:
                nn.init.normal_(emb.weight, std=0.5)

    # geometry ---------------------------------------------------------------------

    def box(self, classes, relations):
        """Centre and offset of ``f_r(Box(c))`` for index tensors of classes and of
        relations of the ontology view (object properties, ``subclass_id``, ``null_id``)."""
        h = self.project(th.cat([self.class_embed(classes),
                                 self.ontology_rel_embed(relations)], dim=-1))
        center, raw = h.chunk(2, dim=-1)
        return center, F.softplus(raw)

    def _corners(self, classes, relations):
        center, offset = self.box(classes, relations)
        return L.box_corners(center, offset)

    def triple_probability(self, head, relation, tail):
        """:math:`\\phi(h, r, t)` (Equation 4) for class indices and a relation of the
        ontology view."""
        r = th.full_like(head, relation) if isinstance(relation, int) else relation
        min_h, max_h = self._corners(head, r)
        min_t, max_t = self._corners(tail, r)
        return L.conditional_probability(min_h, max_h, min_t, max_t, self.beta)

    def cross_distance(self, individuals, classes):
        """:math:`f_d(x_a, \\mathrm{Box}(C))` (Equations 8 and 9)."""
        null = th.full_like(classes, self.null_id)
        center, offset = self.box(classes, null)
        return L.vector_to_box_distance(self.ind_embed(individuals), center, offset,
                                        self.alpha, self.beta)

    def instance_distance(self, a, relation, b):
        return (self.ind_embed(a) + self.rel_embed(relation)
                - self.ind_embed(b)).norm(dim=-1)

    # losses -----------------------------------------------------------------------

    def _need_inds(self):
        if self.ind_embed is None:
            raise ValueError("The number of individuals must be specified to use this "
                             "loss function.")

    def gci0_loss(self, data, neg=False):
        # C ⊑ D : the box of C inside the box of D, triple (D, subclass, C)
        phi = self.triple_probability(data[:, 1], self.subclass_id, data[:, 0])
        return L.ontology_view_loss(phi, neg)

    def gci2_loss(self, data, neg=False):
        if self.gci2_reading == "axiom":     # C ⊑ ∃r.D: the box of C inside that of D
            phi = self.triple_probability(data[:, 2], data[:, 1], data[:, 0])
        else:                                # the paper's triple (C, r, D)
            phi = self.triple_probability(data[:, 0], data[:, 1], data[:, 2])
        return L.ontology_view_loss(phi, neg)

    def class_assertion_loss(self, data, neg=False):
        self._need_inds()
        distance = self.cross_distance(data[:, 0], data[:, 1])
        return self.cross_weight * L.cross_view_loss(distance, neg, self.cross_margin)

    def object_property_assertion_loss(self, data, neg=False):
        self._need_inds()
        distance = self.instance_distance(data[:, 0], data[:, 1], data[:, 2])
        return self.instance_weight * L.instance_view_loss(distance, neg, self.kge_margin)
