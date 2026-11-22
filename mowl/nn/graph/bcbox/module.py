"""Neural module for binary code box embeddings [zhang2022bcbox]_, built from parts."""

import torch as th

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.hierarchy_methods import bcbox
from mowl.representation.library.hierarchy_modules import adopt

VARIANTS = ("gbox", "gbc", "vbc")


class BinaryCodeBoxModule(ComposedGeometry):
    """Gumbel boxes with learned binary codes for directed graphs (Zhang et al., NeurIPS 2022),
    the combination that :func:`mowl.representation.library.hierarchy_methods.bcbox` names
    (read ``module.parts``): a class model of boxes with codes
    (:class:`~mowl.representation.library.hierarchy_classes.CodedBoxes`), the smoothed volume
    ratio that reads the codes as the measure
    (:class:`~mowl.representation.library.hierarchy_measures.LogVolumeRatio`), the energy as
    the cost of a stated edge and :math:`-\\log(1 - e^{-E})` as that of a corruption, and the
    code lasso as an optional regulariser.

    Every vertex ``u`` is a box :math:`\\prod_i [x^-_i(u), x^+_i(u)] \\subseteq
    \\mathbb{R}^d`. The energy :math:`E(u, v)` of the directed edge :math:`(u, v)` is low when
    the box of ``v`` lies inside the box of ``u`` (Eq. 3 normalises by the volume of
    ``v``). Three variants are available:

    * ``"gbox"``: the G-Box model of Section 2.2, no codes. By Theorem 3 / Corollary 4 of the
      paper it cannot model a chordless directed cycle on more than two vertices.
    * ``"gbc"``: global binary codes, Eq. (8). ``num_codes`` (:math:`k`) code vectors
      :math:`B_{\\cdot, j} \\in [0, 1]^d` each select a set of dimensions; the energy is the
      minimum over the :math:`k` codes.
    * ``"vbc"``: per-vertex binary codes, Eq. (9). Every vertex has a code
      :math:`u^\\diamond \\in [0, 1]^d`; dimension ``i`` contributes
      :math:`u^\\diamond_i v^\\diamond_i \\cdot (-\\log F_i)` to :math:`E(u, v)` (Eq. 7).

    ``code_dims`` is :math:`d_{bin}` of Section 4.5: only the last ``code_dims`` dimensions
    carry trainable codes, the first :math:`d - d_{bin}` codes are fixed to 1. A code is the
    sigmoid of an unconstrained logit (the paper does not say how the constraint is
    enforced), initialised from :math:`\\mathcal{N}(\\mu, 1)`, :math:`\\mu` = ``code_init``.

    ``module(edges)`` with a tensor of edges of shape ``(*, 2)`` is the energy of the edges
    in the paper's direction (column 0 = ``u``, column 1 = ``v``); ``module(rows, "gci0")`` is
    the cost of the axioms ``rows[:, 0] ⊑ rows[:, 1]`` as for every geometry. The earlier
    hand-written module is kept for comparison as
    ``mowl.baselines.nn.graph.bcbox.BinaryCodeBoxModule``.

    :param num_nodes: Number of vertices.
    :param embed_dim: Box dimension :math:`d`.
    :param variant: ``"gbox"``, ``"gbc"`` or ``"vbc"``.
    :param num_codes: Number :math:`k` of global codes (``"gbc"`` only).
    :param code_dims: :math:`d_{bin}`, number of dimensions with trainable codes. ``None``
        means all :math:`d`.
    :param intersection_temp: Intersection temperature :math:`\\tau \\ge 0`.
    :param volume_temp: Volume temperature :math:`\\nu \\ge 0`.
    :param code_init: Mean :math:`\\mu` of the initial code logits.
    :param code_weight: Weight :math:`w_r` of the code lasso, the module's regulariser
        (``regularization_loss()``); 0: none.
    """

    def __init__(self, num_nodes, embed_dim, variant="vbc", num_codes=2, code_dims=None,
                 intersection_temp=0.01, volume_temp=1.0, code_init=0.0, code_weight=0.0):
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
        adopt(self, bcbox(num_nodes, embed_dim=embed_dim, variant=variant,
                          num_codes=num_codes, code_dims=code_dims,
                          intersection_temp=intersection_temp, volume_temp=volume_temp,
                          code_init=code_init, code_weight=code_weight))
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.variant = variant
        self.num_codes = self.class_model.num_codes
        self.code_dims = self.class_model.code_dims
        self.intersection_temp = float(intersection_temp)
        self.volume_temp = float(volume_temp)

    # ------------------------------------------------------------ parameters
    @property
    def box_min(self):
        return self.class_model.box_min

    @property
    def box_max(self):
        return self.class_model.box_max

    @property
    def code_logits(self):
        return self.class_model.code_logits

    # ------------------------------------------------------------------ codes
    def trainable_codes(self):
        """The trainable codes in :math:`[0, 1]` (``None`` for ``"gbox"``).

        Shape ``(code_dims, k)`` for ``"gbc"`` and ``(num_nodes, code_dims)`` for ``"vbc"``.
        """
        if self.code_logits is None:
            return None
        return th.sigmoid(self.code_logits)

    def global_codes(self):
        """The full global code matrix :math:`B \\in [0, 1]^{d \\times k}` (``"gbc"`` only)."""
        if self.variant != "gbc":
            raise ValueError("global_codes is only defined for variant 'gbc'.")
        return self.class_model.global_codes()

    def vertex_codes(self, idx):
        """Per-vertex codes :math:`u^\\diamond \\in [0, 1]^d` (``"vbc"`` only), of shape
        ``idx.shape + (d,)``."""
        if self.variant != "vbc":
            raise ValueError("vertex_codes is only defined for variant 'vbc'.")
        return self.class_model.vertex_codes(idx)

    def code_regularization(self):
        """:math:`\\lVert \\mathbf{1} - B \\rVert_{\\ell_1}` of the trainable codes (Sec. 4.5).
        For ``"vbc"`` the sum runs over the codes of all vertices; it is 0 for ``"gbox"``."""
        return self.class_model.code_regularization()

    # ----------------------------------------------------------------- energy
    def neg_log_scores(self, u, v):
        """Per-dimension :math:`-\\log F_{\\text{G-Box}}(u_i, v_i)`, shape ``u.shape + (d,)``."""
        return self.measure.neg_log_scores(self.class_model(v), self.class_model(u))

    def energy(self, u, v):
        """Energy :math:`E(u, v) = -\\log P((u, v) \\in E)` of the directed edges ``u -> v``
        (low when the box of ``v`` is softly contained in the box of ``u``); ``u``, ``v``
        are vertex indices of any shape, the energy has that shape."""
        return self.measure.energy(self.class_model(v), self.class_model(u)).squeeze(-1)

    def forward(self, edges, gci_name=None, neg=False):
        """Energy of a tensor of edges of shape ``(*, 2)`` (column 0 = ``u``, column 1 =
        ``v``); with a form name the cost of the axioms (see :class:`ComposedGeometry`)."""
        if gci_name is None:
            return self.energy(edges[..., 0], edges[..., 1])
        return super().forward(edges, gci_name, neg)

    @th.no_grad()
    def set_boxes(self, box_min, box_max):
        """Overwrite the box endpoints (for hand-built configurations)."""
        self.box_min.copy_(th.as_tensor(box_min, dtype=self.box_min.dtype))
        self.box_max.copy_(th.as_tensor(box_max, dtype=self.box_max.dtype))

    @th.no_grad()
    def set_code_logits(self, logits):
        """Overwrite the code logits (for hand-built configurations)."""
        self.code_logits.copy_(th.as_tensor(logits, dtype=self.code_logits.dtype))
