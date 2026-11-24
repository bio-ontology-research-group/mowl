"""Neural module for binary code box embeddings [zhang2022bcbox]_."""

import torch as th
import torch.nn as nn

from mowl.nn.graph.bcbox.losses import (
    code_lasso,
    gbc_energy,
    gbox_energy,
    gumbel_box_log_score,
    vbc_energy,
)

VARIANTS = ("gbox", "gbc", "vbc")


class BinaryCodeBoxModule(nn.Module):
    """Gumbel boxes with learned binary codes for directed graphs (Zhang et al., NeurIPS 2022).

    Every vertex ``u`` is a box :math:`\\prod_i [x^-_i(u), x^+_i(u)] \\subseteq
    \\mathbb{R}^d`. The energy :math:`E(u, v)` of the directed edge :math:`(u, v)` is low when
    the box of ``v`` lies inside the box of ``u`` (Eq. 3 normalises by the volume of
    ``v``). Three variants are available:

    * ``"gbox"``: the G-Box model of Section 2.2 (Eq. 2 with Eq. 3), no codes. By Theorem 3 /
      Corollary 4 of the paper it cannot model a chordless directed cycle on more than two
      vertices.
    * ``"gbc"``: global binary codes, Eq. (8). ``num_codes`` (:math:`k`) code vectors
      :math:`B_{\\cdot, j} \\in [0, 1]^d` each select a set of dimensions; the energy is the
      minimum over the :math:`k` codes, so the edge set is a union over code words of
      intersections of per-dimension interval graphs (Eq. 6).
    * ``"vbc"``: per-vertex binary codes, Eq. (9). Every vertex has a code
      :math:`u^\\diamond \\in [0, 1]^d`; dimension ``i`` contributes
      :math:`u^\\diamond_i v^\\diamond_i \\cdot (-\\log F_i)` to :math:`E(u, v)` (Eq. 7).

    ``code_dims`` is :math:`d_{bin}` of Section 4.5: only the last ``code_dims`` dimensions
    carry trainable codes, the first :math:`d - d_{bin}` codes are fixed to 1.

    **Parametrisation of the codes (not specified in the paper).** The paper only states
    :math:`B \\in [0, 1]^{d \\times k}` and :math:`u^\\diamond \\in [0, 1]^d`; neither the
    paper, its appendix nor the public reference repository (which does not contain the
    binary-code models) says how the constraint is enforced. Here a code is the sigmoid of
    an unconstrained logit, initialised from :math:`\\mathcal{N}(\\mu, 1)` with
    :math:`\\mu` = ``code_init``.

    :param num_nodes: Number of vertices.
    :type num_nodes: int
    :param embed_dim: Box dimension :math:`d`.
    :type embed_dim: int
    :param variant: ``"gbox"``, ``"gbc"`` or ``"vbc"``.
    :type variant: str
    :param num_codes: Number :math:`k` of global codes (``"gbc"`` only).
    :type num_codes: int
    :param code_dims: :math:`d_{bin}`, number of dimensions with trainable codes. ``None``
        means all :math:`d`.
    :type code_dims: int, optional
    :param intersection_temp: Intersection temperature :math:`\\tau \\ge 0`.
    :type intersection_temp: float
    :param volume_temp: Volume temperature :math:`\\nu \\ge 0`.
    :type volume_temp: float
    :param code_init: Mean :math:`\\mu` of the initial code logits.
    :type code_init: float
    """

    def __init__(self, num_nodes, embed_dim, variant="vbc", num_codes=2, code_dims=None,
                 intersection_temp=0.01, volume_temp=1.0, code_init=0.0):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}.")
        if code_dims is None:
            code_dims = embed_dim
        if not 0 <= code_dims <= embed_dim:
            raise ValueError("code_dims must be between 0 and embed_dim.")
        if variant == "gbc" and num_codes < 1:
            raise ValueError("num_codes must be at least 1.")
        if intersection_temp < 0 or volume_temp < 0:
            raise ValueError("Temperatures must be non-negative.")

        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.variant = variant
        self.num_codes = num_codes if variant == "gbc" else 0
        self.code_dims = code_dims if variant != "gbox" else 0
        self.intersection_temp = float(intersection_temp)
        self.volume_temp = float(volume_temp)

        corners = th.sort(th.randn(num_nodes, embed_dim, 2), dim=-1).values
        self.box_min = nn.Parameter(corners[..., 0].contiguous())
        self.box_max = nn.Parameter(corners[..., 1].contiguous())

        if variant == "gbc":
            self.code_logits = nn.Parameter(
                th.randn(self.code_dims, num_codes) + code_init)
        elif variant == "vbc":
            self.code_logits = nn.Parameter(
                th.randn(num_nodes, self.code_dims) + code_init)
        else:
            self.register_parameter("code_logits", None)

    # ------------------------------------------------------------------ codes
    def trainable_codes(self):
        """The trainable codes in :math:`[0, 1]` (``None`` for ``"gbox"``).

        Shape ``(code_dims, k)`` for ``"gbc"`` and ``(num_nodes, code_dims)`` for ``"vbc"``.
        """
        if self.code_logits is None:
            return None
        return th.sigmoid(self.code_logits)

    def global_codes(self):
        """The full global code matrix :math:`B \\in [0, 1]^{d \\times k}` (``"gbc"`` only).

        :rtype: :class:`torch.Tensor`
        """
        if self.variant != "gbc":
            raise ValueError("global_codes is only defined for variant 'gbc'.")
        fixed = th.ones(self.embed_dim - self.code_dims, self.num_codes,
                        device=self.box_min.device, dtype=self.box_min.dtype)
        return th.cat((fixed, self.trainable_codes()), dim=0)

    def vertex_codes(self, idx):
        """Per-vertex codes :math:`u^\\diamond \\in [0, 1]^d` (``"vbc"`` only).

        :param idx: Vertex indices, any shape.
        :type idx: :class:`torch.Tensor`
        :returns: Codes of shape ``idx.shape + (d,)``.
        :rtype: :class:`torch.Tensor`
        """
        if self.variant != "vbc":
            raise ValueError("vertex_codes is only defined for variant 'vbc'.")
        learned = th.sigmoid(self.code_logits[idx])
        fixed = th.ones(*idx.shape, self.embed_dim - self.code_dims,
                        device=learned.device, dtype=learned.dtype)
        return th.cat((fixed, learned), dim=-1)

    def code_regularization(self):
        """:math:`\\lVert \\mathbf{1} - B \\rVert_{\\ell_1}` of the trainable codes (Sec. 4.5).

        For ``"vbc"`` the sum runs over the codes of all vertices. It is 0 for ``"gbox"``.

        :rtype: :class:`torch.Tensor`
        """
        codes = self.trainable_codes()
        if codes is None:
            return th.zeros((), device=self.box_min.device)
        return code_lasso(codes)

    # ----------------------------------------------------------------- energy
    def neg_log_scores(self, u, v):
        """Per-dimension :math:`-\\log F_{\\text{G-Box}}(u_i, v_i)`, shape ``u.shape + (d,)``."""
        return -gumbel_box_log_score(self.box_min[u], self.box_max[u],
                                     self.box_min[v], self.box_max[v],
                                     self.intersection_temp, self.volume_temp)

    def energy(self, u, v):
        """Energy :math:`E(u, v) = -\\log P((u, v) \\in E)` of the directed edges ``u -> v``.

        Low energy means the box of ``v`` is (softly) contained in the box of ``u``.

        :param u: Source vertex indices, any shape.
        :type u: :class:`torch.Tensor`
        :param v: Target vertex indices, same shape as ``u``.
        :type v: :class:`torch.Tensor`
        :rtype: :class:`torch.Tensor`
        """
        nls = self.neg_log_scores(u, v)
        if self.variant == "gbox":
            return gbox_energy(nls)
        if self.variant == "gbc":
            return gbc_energy(nls, self.global_codes())
        return vbc_energy(nls, self.vertex_codes(u), self.vertex_codes(v))

    def forward(self, edges):
        """Energy of a tensor of edges of shape ``(*, 2)`` (column 0 = ``u``, column 1 = ``v``).

        :rtype: :class:`torch.Tensor`
        """
        return self.energy(edges[..., 0], edges[..., 1])

    @th.no_grad()
    def set_boxes(self, box_min, box_max):
        """Overwrite the box endpoints (for hand-built configurations)."""
        self.box_min.copy_(th.as_tensor(box_min, dtype=self.box_min.dtype))
        self.box_max.copy_(th.as_tensor(box_max, dtype=self.box_max.dtype))

    @th.no_grad()
    def set_code_logits(self, logits):
        """Overwrite the code logits (for hand-built configurations)."""
        self.code_logits.copy_(th.as_tensor(logits, dtype=self.code_logits.dtype))
