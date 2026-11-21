"""EIKE (Wang et al., 2024): extensional (ellipsoid / point) and intensional (vector) spaces.

The losses are the scores :math:`f` of Section 4.2 of the paper; the margin-based ranking
loss of Section 4.3 is built by the model from the scores of positive and negative axioms.
"""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.nn.el.elmodule import ELModule

_EPS = 1e-6


class EIKEModule(ELModule):
    """EIKE module.

    Extensional space: concept :math:`c` is the ellipsoid
    :math:`\\{x \\mid \\sum_j ((x_j - c_j)/b_j)^2 \\le r_c^2\\}` (centre ``c``, positive
    semi-axes ``b``, positive scale ``r_c``), instances are points, relations are vectors.
    Intensional space: concepts are vectors (optionally initialised from pre-computed
    language model vectors), instances are the virtual vectors :math:`M\\,i^e`.

    Per-axiom scores (lower is better):

    * ``class_assertion`` (InstanceOf): :math:`f^{ex} + \\alpha f^{in}` with
      :math:`f^{ex} = [\\sum_j ((i_j - c_j)/b_j)^2 - r_c^2]_+`, :math:`f^{in} = 1 - \\cos(Mi, c^{i})`.
    * ``gci0`` (SubClassOf): :math:`f^{ex} + \\alpha f^{in}` with
      :math:`f^{ex} = [\\sum_p (c_{ip}/b_{ip} - c_{jp}/b_{jp})^2 + r_{c_i}^2 - r_{c_j}^2]_+`
      (as printed in the paper) and :math:`f^{in} = 1 - \\cos(c^i_i, c^i_j) +
      (\\|c^i_i\\| - \\|c^i_j\\|)`.
    * ``object_property_assertion`` (relational triple): TransE, :math:`\\|h + r - t\\|_2^2`.

    A negative call (``neg=True``) returns the score minus the margin of the axiom type, so
    that the ranking loss :math:`[\\gamma + f(\\delta) - f(\\delta')]_+` of the paper is
    ``relu(pos - neg)`` (see :meth:`mowl.models.eike.model.EIKE.compute_loss`).

    :param nb_classes: Number of classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals.
    :param embed_dim: Dimension :math:`d` of both spaces.
    :param alpha: Weight of the intensional part (the paper's :math:`\\alpha`).
    :param margins: ``(gamma_sub, gamma_ins, gamma_rel)``.
    :param instance_map: ``"eye"`` (fixed identity, EYE in the paper) or ``"matrix"``
        (learnable :math:`M`, MAT).
    :param intensional_init: Optional array ``(nb_classes, embed_dim)`` of pre-computed
        vectors for the intensional concept embeddings (rows of classes without text may be
        NaN: they keep the random initialisation).
    :param freeze_intensional: Keep the intensional concept vectors fixed.
    """

    neg_capable_gcis = frozenset({"gci0", "class_assertion", "object_property_assertion"})

    def __init__(self, nb_classes, nb_rels, nb_inds, embed_dim=100, alpha=1.0,
                 margins=(1.0, 1.0, 1.0), instance_map="eye", intensional_init=None,
                 freeze_intensional=False):
        super().__init__()
        if instance_map not in ("eye", "matrix"):
            raise ValueError("instance_map must be 'eye' or 'matrix'.")
        if nb_inds < 1:
            nb_inds = 1
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.gamma_sub, self.gamma_ins, self.gamma_rel = margins
        self.instance_map = instance_map

        self.class_center = self._emb(nb_classes, embed_dim)
        # b = softplus(raw) > 0 and r = softplus(raw) > 0; raw 0.5413 gives 1.
        self.class_axes_raw = nn.Embedding(nb_classes, embed_dim)
        self.class_radius_raw = nn.Embedding(nb_classes, 1)
        nn.init.constant_(self.class_axes_raw.weight, 0.5413)
        nn.init.constant_(self.class_radius_raw.weight, 0.5413)
        self.ind_embed = self._emb(nb_inds, embed_dim)
        self.rel_embed = self._emb(max(nb_rels, 1), embed_dim)
        self.class_intensional = self._emb(nb_classes, embed_dim)
        if intensional_init is not None:
            init = th.as_tensor(np.asarray(intensional_init, dtype="float32"))
            if init.shape != (nb_classes, embed_dim):
                raise ValueError(f"intensional_init must have shape ({nb_classes}, "
                                 f"{embed_dim}), got {tuple(init.shape)}.")
            keep = ~th.isnan(init).any(dim=1)
            with th.no_grad():
                self.class_intensional.weight[keep] = init[keep]
        if freeze_intensional:
            self.class_intensional.weight.requires_grad_(False)
        if instance_map == "matrix":
            self.M = nn.Parameter(th.eye(embed_dim) + 0.01 * th.randn(embed_dim, embed_dim))
        else:
            self.register_buffer("M", th.eye(embed_dim))

    @staticmethod
    def _emb(n, d):
        e = nn.Embedding(n, d)
        nn.init.uniform_(e.weight, -0.5, 0.5)
        return e

    # -- parameters ----------------------------------------------------------------------
    def axes(self, idx):
        return F.softplus(self.class_axes_raw(idx)) + _EPS

    def radius(self, idx):
        return F.softplus(self.class_radius_raw(idx))

    def virtual_instance(self, idx):
        """Intensional vector of an instance, :math:`M i^e`."""
        return self.ind_embed(idx) @ self.M.T

    # -- scores --------------------------------------------------------------------------
    def class_assertion_loss(self, data, neg=False):
        x = self.ind_embed(data[:, 0])
        c = self.class_center(data[:, 1])
        b = self.axes(data[:, 1])
        r = self.radius(data[:, 1]).squeeze(-1)
        f_ex = F.relu((((x - c) / b) ** 2).sum(-1) - r ** 2)
        f_in = 1 - F.cosine_similarity(self.virtual_instance(data[:, 0]),
                                       self.class_intensional(data[:, 1]), dim=-1)
        f = f_ex + self.alpha * f_in
        return f - self.gamma_ins if neg else f

    def gci0_loss(self, data, neg=False):
        ci, cj = self.class_center(data[:, 0]), self.class_center(data[:, 1])
        bi, bj = self.axes(data[:, 0]), self.axes(data[:, 1])
        ri, rj = self.radius(data[:, 0]).squeeze(-1), self.radius(data[:, 1]).squeeze(-1)
        f_ex = F.relu(((ci / bi - cj / bj) ** 2).sum(-1) + ri ** 2 - rj ** 2)
        vi, vj = self.class_intensional(data[:, 0]), self.class_intensional(data[:, 1])
        f_in = 1 - F.cosine_similarity(vi, vj, dim=-1) + (vi.norm(dim=-1) - vj.norm(dim=-1))
        f = f_ex + self.alpha * f_in
        return f - self.gamma_sub if neg else f

    def object_property_assertion_loss(self, data, neg=False):
        h, r, t = self.ind_embed(data[:, 0]), self.rel_embed(data[:, 1]), \
            self.ind_embed(data[:, 2])
        f = ((h + r - t) ** 2).sum(-1)
        return f - self.gamma_rel if neg else f

    def regularization_loss(self):
        return 0
