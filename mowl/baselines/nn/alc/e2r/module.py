r"""Neural module of Embed2Reason (E2R; Garg et al., NeurIPS 2019).

Concepts and relations are axis-parallel linear subspaces of :math:`\mathbb{C}^d`
simulated as :math:`\mathbb{R}^{2d}` (Sec. 4), given by learnable indicator vectors;
entities are learnable vectors. See :mod:`mowl.baselines.nn.alc.e2r.losses` for the loss terms.

The module also provides exact linear-algebra helpers for *arbitrary* (oblique)
subspaces given by orthogonal projectors (:func:`projector`, :func:`meet_projector`,
:func:`join_projector`, :func:`orthocomplement`, :func:`is_subspace`). They implement the
quantum-logic operations of Sec. 3 in general and are used to check the axis-parallel
case against ground truth and to exhibit non-distributivity (which the axis-parallel
restriction removes, supplementary material Theorem 11).
"""
import torch as th
import torch.nn as nn

from mowl.baselines.nn.alc.module import ALCModule
from mowl.baselines.nn.alc.e2r import losses as L

# Axiom kinds understood by :meth:`E2RModule.forward`; each maps to ``<kind>_loss``.
AXIOM_KINDS = (
    "class_assertion", "role_assertion",
    "concept_inclusion", "concept_conjunction", "concept_disjunction",
    "concept_negation", "concept_disjointness",
    "role_inclusion", "role_conjunction", "role_disjunction", "role_negation",
    "range",
)


# -- general subspaces (projector representation) --------------------------------------

def _orthonormal_range(m, tol):
    """Orthonormal basis (columns) of the column space of ``m``."""
    if m.shape[1] == 0:
        return m.new_zeros((m.shape[0], 0))
    u, s, _ = th.linalg.svd(m, full_matrices=False)
    rank = int((s > tol * max(1.0, float(s.max()) if s.numel() else 1.0)).sum())
    return u[:, :rank]


def projector(basis, tol=1e-9):
    r"""Orthogonal projector :math:`P = QQ^\top` onto the span of the columns of
    ``basis`` (``[n, k]``; linearly dependent columns are allowed)."""
    basis = th.as_tensor(basis, dtype=th.float64)
    q = _orthonormal_range(basis, tol)
    return q @ q.T


def indicator_projector(y):
    r"""Projector of the axis-parallel subspace with 0/1 indicator ``y``:
    :math:`\mathrm{diag}(y)` (supplementary material, Theorem 9)."""
    return th.diag(th.as_tensor(y, dtype=th.float64))


def rank(p, tol=1e-9):
    """Dimension of the subspace with projector ``p``."""
    return int(th.linalg.matrix_rank(p, atol=tol))


def orthocomplement(p):
    r""":math:`\neg S = S^\perp`, projector :math:`I - P` (Sec. 3)."""
    return th.eye(p.shape[0], dtype=p.dtype) - p


def join_projector(*ps, tol=1e-9):
    r""":math:`S_1 \vee S_2 = S_1 + S_2` (span of the union, Sec. 3)."""
    return projector(th.cat(list(ps), dim=1), tol)


def meet_projector(*ps, tol=1e-9):
    r""":math:`S_1 \wedge S_2 = S_1 \cap S_2` (Sec. 3), computed as the orthocomplement of
    :math:`S_1^\perp + S_2^\perp`."""
    return orthocomplement(join_projector(*[orthocomplement(p) for p in ps], tol=tol))


def is_subspace(p_sub, p_sup, tol=1e-8):
    r""":math:`S_1 \sqsubseteq S_2` iff :math:`(I - P_2) P_1 = 0` (equivalently
    :math:`P_1 P_2 = P_2 P_1 = P_1`, supplementary material, Theorem 2)."""
    return bool(th.linalg.norm(orthocomplement(p_sup) @ p_sub) < tol)


def residual_length(p, x):
    r"""Length of the residual :math:`\|(I - P)x\|` of projecting ``x`` onto :math:`S`
    (the non-fitment score of Sec. 5)."""
    x = th.as_tensor(x, dtype=th.float64)
    return float(th.linalg.norm(orthocomplement(p) @ x))


# -- the module ------------------------------------------------------------------------

class E2RModule(ALCModule):
    r"""Parameters and per-axiom losses of E2R.

    * entity :math:`i`: vector :math:`x_i \in \mathbb{R}^{2d}` (imaginary half pushed to
      ``0`` by eq. (1));
    * concept :math:`C`: indicator :math:`y_C \in [0, 1]^{2d}` (imaginary half pushed to
      ``0`` by eq. (2), values pushed to :math:`\{0, 1\}` by eq. (3));
    * relation :math:`R`: indicator :math:`z_R \in [0, 1]^{2d}`.

    ``owl:Thing`` and ``owl:Nothing`` (if their indices are given) are fixed to the
    concept universe :math:`(1_d, 0_d)` and to :math:`\{0\}`.

    The paper optimises the indicators directly (eq. (10)); it does not say how they are
    kept in :math:`[0, 1]`. Here :meth:`project_` clips them after every optimiser step
    (projected gradient descent).

    :meth:`forward` dispatches on an axiom kind (one of ``AXIOM_KINDS``) and a batch of
    index rows, returning one loss value per row. :meth:`forward_fs` of
    :class:`~mowl.nn.alc.module.ALCModule` (fuzzy-set semantics) does not apply.

    :param num_entities: Number of individuals.
    :param num_classes: Number of classes.
    :param num_relations: Number of object properties.
    :param dim: :math:`d`; vectors have :math:`2d` real coordinates. Defaults to ``100``
        (Sec. 5).
    :param top_index: Index of ``owl:Thing`` or ``None``.
    :param bottom_index: Index of ``owl:Nothing`` or ``None``.
    :param role_unit_length: Add the unit-length terms of the two entities to role
        assertion losses (see :func:`~mowl.baselines.nn.alc.e2r.losses.role_membership_loss`).
    """

    def __init__(self, num_entities, num_classes, num_relations, dim=100, top_index=None,
                 bottom_index=None, role_unit_length=True):
        super().__init__()
        self.dim = dim
        self.top_index = top_index
        self.bottom_index = bottom_index
        self.role_unit_length = role_unit_length

        self.e_embedding = nn.Embedding(max(num_entities, 1), 2 * dim)
        self.c_embedding = nn.Embedding(max(num_classes, 1), 2 * dim)
        self.r_embedding = nn.Embedding(max(num_relations, 1), 2 * dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Entities: random unit vectors in the real half. Concepts: real half
        :math:`U(0, 1)`, imaginary half ``0``. Relations: :math:`U(0, 1)` everywhere. (The
        paper does not state an initialisation.)"""
        d = self.dim
        with th.no_grad():
            x = th.randn(self.e_embedding.weight.shape[0], d)
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.e_embedding.weight.zero_()
            self.e_embedding.weight[:, :d] = x
            self.c_embedding.weight.uniform_(0, 1)
            self.c_embedding.weight[:, d:] = 0
            self.r_embedding.weight.uniform_(0, 1)

    def project_(self):
        """Clip all indicator vectors to :math:`[0, 1]` in place."""
        with th.no_grad():
            self.c_embedding.weight.clamp_(0, 1)
            self.r_embedding.weight.clamp_(0, 1)

    # -- accessors ---------------------------------------------------------------------
    def entity(self, idx):
        return self.e_embedding(idx)

    def concept(self, idx):
        """Indicator vectors of concepts ``idx`` (``owl:Thing``/``owl:Nothing`` fixed)."""
        y = self.c_embedding(idx)
        if self.top_index is None and self.bottom_index is None:
            return y
        d = self.dim
        if self.top_index is not None:
            top = th.cat([th.ones(d), th.zeros(d)]).to(y)
            y = th.where((idx == self.top_index).unsqueeze(-1), top, y)
        if self.bottom_index is not None:
            y = th.where((idx == self.bottom_index).unsqueeze(-1), th.zeros_like(y), y)
        return y

    def relation(self, idx):
        return self.r_embedding(idx)

    # -- structural and regularisation terms --------------------------------------------
    def structure_loss(self, class_idx=None):
        """Mean of eqs. (1), (2) and (3) over all entities, concepts (except the fixed
        top/bottom) and relations, summed."""
        d = self.dim
        x = self.e_embedding.weight
        r = self.r_embedding.weight
        if class_idx is None:
            class_idx = th.arange(self.c_embedding.weight.shape[0],
                                  device=x.device)
        c = self.c_embedding(class_idx)
        loss = L.entity_structure_loss(x, d).mean()
        if len(class_idx) > 0:
            loss = loss + L.concept_structure_loss(c, d).mean() + L.binary_loss(c).mean()
        loss = loss + L.relation_structure_loss(r, d).mean() + L.binary_loss(r).mean()
        return loss

    def orthogonality_loss(self, pairs, relations=False):
        """Eq. (11) summands for sibling pairs (``[N, 2]``)."""
        get = self.relation if relations else self.concept
        return L.orthogonality_loss(get(pairs[:, 0]), get(pairs[:, 1]))

    def separation_loss(self, pairs, gap, relations=False):
        """Eq. (12) summands for (sub, super) pairs (``[N, 2]``)."""
        get = self.relation if relations else self.concept
        return L.separation_loss(get(pairs[:, 0]), get(pairs[:, 1]), gap)

    # -- per-axiom losses ---------------------------------------------------------------
    def forward(self, axiom, x, e_emb=None, neg=False):
        """Loss of a batch ``x`` of axioms of kind ``axiom`` (a string of
        ``AXIOM_KINDS``). ``neg=True`` is accepted for ``class_assertion`` and
        ``role_assertion`` and returns the negated membership loss."""
        if axiom not in AXIOM_KINDS:
            raise ValueError(f"Unknown E2R axiom kind {axiom!r}.")
        fn = getattr(self, f"{axiom}_loss")
        if axiom in ("class_assertion", "role_assertion"):
            return fn(x, neg=neg)
        return fn(x)

    def forward_fs(self, cexpr, x, e_emb, cur_index=0):
        raise NotImplementedError(
            "E2R does not use fuzzy-set semantics; see E2RModule.forward.")

    def class_assertion_loss(self, x, neg=False):
        """Rows ``(individual, class)``; eq. (4) (or its negation for ``neg``)."""
        y, e = self.concept(x[:, 1]), self.entity(x[:, 0])
        if neg:
            return L.negative_membership_loss(y, e)
        return L.membership_loss(y, e)

    def role_assertion_loss(self, x, neg=False):
        """Rows ``(subject, relation, object)``; eq. (4) (or its negation for ``neg``)."""
        z = self.relation(x[:, 1])
        xp, xq = self.entity(x[:, 0]), self.entity(x[:, 2])
        if neg:
            return L.negative_role_membership_loss(z, xp, xq, self.dim)
        return L.role_membership_loss(z, xp, xq, self.dim, unit_length=self.role_unit_length)

    def concept_inclusion_loss(self, x):
        """Rows ``(sub, super)``; eq. (5)."""
        return L.inclusion_loss(self.concept(x[:, 0]), self.concept(x[:, 1]))

    def concept_conjunction_loss(self, x):
        """Rows ``(C, D1, ..., Dk)`` for :math:`C \\equiv D_1 \\sqcap \\dots \\sqcap D_k`;
        eq. (6)."""
        return L.conjunction_loss(self.concept(x[:, 0]),
                                  *[self.concept(x[:, i]) for i in range(1, x.shape[1])])

    def concept_disjunction_loss(self, x):
        """Rows ``(C, D1, ..., Dk)`` for :math:`C \\equiv D_1 \\sqcup \\dots \\sqcup D_k`;
        eq. (7)."""
        return L.disjunction_loss(self.concept(x[:, 0]),
                                  *[self.concept(x[:, i]) for i in range(1, x.shape[1])])

    def concept_negation_loss(self, x):
        """Rows ``(C, D)`` for :math:`C \\equiv \\neg D`; eq. (8) on the real half."""
        return L.negation_loss(self.concept(x[:, 0]), self.concept(x[:, 1]), dim=self.dim)

    def concept_disjointness_loss(self, x):
        """Rows ``(C, D)`` for :math:`C \\sqcap D \\sqsubseteq \\bot`; eq. (6) with
        :math:`\\bot = \\{0\\}`."""
        return L.disjointness_loss(self.concept(x[:, 0]), self.concept(x[:, 1]))

    def role_inclusion_loss(self, x):
        """Rows ``(R, S)`` for :math:`R \\sqsubseteq S`; eq. (5)."""
        return L.inclusion_loss(self.relation(x[:, 0]), self.relation(x[:, 1]))

    def role_conjunction_loss(self, x):
        """Rows ``(R, S1, ..., Sk)``; eq. (6) for relations."""
        return L.conjunction_loss(self.relation(x[:, 0]),
                                  *[self.relation(x[:, i]) for i in range(1, x.shape[1])])

    def role_disjunction_loss(self, x):
        """Rows ``(R, S1, ..., Sk)``; eq. (7) for relations."""
        return L.disjunction_loss(self.relation(x[:, 0]),
                                  *[self.relation(x[:, i]) for i in range(1, x.shape[1])])

    def role_negation_loss(self, x):
        """Rows ``(R, S)`` for :math:`R = \\neg S`; eq. (8) on all coordinates."""
        return L.negation_loss(self.relation(x[:, 0]), self.relation(x[:, 1]))

    def range_loss(self, x):
        """Rows ``(R, C_k)``: one summand of eq. (9) for a restriction
        :math:`\\forall R.C_j` and a non-descendant :math:`C_k` of :math:`C_j`."""
        return L.range_loss(self.concept(x[:, 1]), self.relation(x[:, 0]), self.dim)

    # -- scoring ------------------------------------------------------------------------
    def membership_residual(self, y, x):
        """Residual length :math:`\\|\\overline{y} \\odot x\\|` (non-fitment score, Sec. 5;
        lower is better)."""
        return L.residual_sq(y, x).clamp_min(0).sqrt()
