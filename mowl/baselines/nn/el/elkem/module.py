r"""Neural module of ELK-Em [singh2026]_."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import mowl.baselines.nn.el.elkem.losses as L
from mowl.nn.el.elmodule import ELModule


class ELKEmModule(ELModule):
    r"""ELK-Em: closure-aware box embeddings for :math:`\mathcal{EL}^+_\bot` [singh2026]_.

    * A concept name :math:`A` is an axis-aligned box :math:`B_A = [c_A - o_A, c_A + o_A]`
      with :math:`o_A = |\tilde o_A| + \epsilon > 0`, so every trained box is non-empty.
      :math:`\top` is the full box and :math:`\bot` (and every concept fixed to be
      unsatisfiable) the canonical empty box; these boxes are not trained.
    * An individual :math:`a` is a point :math:`e_a`.
    * A role :math:`r` is a triple :math:`\langle R_r, E_r, T_r\rangle`: a non-empty range box
      :math:`R_r`, a non-empty error box :math:`E_r` and a diagonal affine map
      :math:`T_r(x) = m_r x + c_r` with slope :math:`m_r = |\tilde m_r| + \epsilon > 0`.
      :math:`E_r = [-o^E_r, o^E_r]` is centred at the origin; this loses no generality
      because a centre of :math:`E_r` can be absorbed into the shift :math:`c_r` (only
      :math:`T_r(y) \oplus E_r` enters the semantics), and it matches the authors' code.

    Per-axiom losses (Table 2 of the paper), with :math:`\gamma` = ``margin``:

    * ``gci0`` :math:`C \sqsubseteq D`: :math:`\ell_\subseteq(B_C, B_D)`;
    * ``gci0_bot`` :math:`C \sqsubseteq \bot`: none, :math:`B_C` is fixed to the empty box;
    * ``gci1`` :math:`C_1 \sqcap C_2 \sqsubseteq D`: :math:`\ell_\subseteq(B_{C_1} \cap
      B_{C_2}, B_D) + \ell_{\cap+}(B_{C_1}, B_{C_2})`;
    * ``gci1_bot`` :math:`C_1 \sqcap C_2 \sqsubseteq \bot`:
      :math:`\ell_{\cap-}(B_{C_1}, B_{C_2})`;
    * ``gci2`` :math:`C \sqsubseteq \exists r.D`: :math:`\ell_\subseteq(B_C, B_{\exists r.D})`;
    * ``gci3`` :math:`\exists r.C \sqsubseteq D`: :math:`\ell_\subseteq(B_{\exists r.C}, B_D)
      + \ell_{\cap+}(R_r, B_C)`;
    * ``gci3_bot`` :math:`\exists r.C \sqsubseteq \bot`: :math:`\ell_{\cap-}(R_r, B_C)`;
    * ``role_inclusion`` :math:`r \sqsubseteq s`: :math:`\ell_\subseteq(R_r, R_s) +
      \sum_{y \in \{l^R_r, u^R_r\}} \ell_\subseteq(T_r(y) \oplus E_r, T_s(y) \oplus E_s)`;
    * ``role_chain`` :math:`r_1 \circ r_2 \sqsubseteq s`: the role inclusion loss of
      :math:`\max(r_1 \circ r_2)` against :math:`s`.

    Concept assertions :math:`C(a)` use :math:`\ell_\subseteq([e_a, e_a], B_C)` and role
    assertions :math:`r(a, b)` use :math:`\ell_\subseteq([e_a, e_a], T_r([e_b, e_b]) \oplus
    E_r) + \ell_\subseteq([e_b, e_b], R_r)`. With ``neg=True`` an assertion loss returns the
    loss of a corrupted assertion, :math:`\mathrm{ReLU}(-\log\tau - \ell)`.

    .. rubric:: Margin and scores

    In training mode the losses use ``margin``; in evaluation mode (``module.eval()``) they
    use margin 0, i.e. they return :math:`\ell^0(\alpha)`, which mOWL's rank-based
    evaluators rank in ascending order. :meth:`score` returns the paper's score
    :math:`s(\alpha) = \exp(-\ell^0(\alpha)) \in (0, 1]` regardless of the mode.

    .. rubric:: Deviations

    * The box of :math:`\exists r.C` is kept empty when :math:`C \cap R_r` is empty
      (:math:`\emptyset \oplus E_r = \emptyset`). Interval arithmetic on the inverted box, as
      in the authors' code, can return a non-empty box there, which would let the ``gci2``
      loss (which has no overlap term in Table 2) vanish for an unsatisfied axiom.
    * The overlap terms :math:`\ell_{\cap+}` of ``gci1`` and ``gci3`` are dropped for rows in
      which one of the two boxes is fixed to be empty. Such an axiom holds trivially, and the
      overlap term (a heuristic against degenerate solutions) could never be zero.
    * A ``gci0_bot`` row whose class was *not* fixed to be empty gets the positive loss
      :math:`\frac{1}{n}\lVert\mathrm{ReLU}(o_C + \gamma)\rVert_1` instead of being silently
      ignored. :class:`~mowl.models.elkem.model.ELKEm` fixes every such class, so this loss
      is zero in training.

    * A ``gci1_bot`` row with a conjunct fixed to the empty box, and a ``gci3_bot`` row whose
      filler is fixed to the empty box, hold trivially and cost 0. The disjointness primitive
      would charge the margin for :math:`\\bot \\sqcap \\top \\sqsubseteq \\bot`, a cost
      that no parameter can remove (``variant="reference"`` keeps that value).
    * :meth:`zero_margin_loss` returns a column ``(B, 1)`` like every other loss
      (``per_axiom_column = False`` gives ``(B,)``).

    .. rubric:: Variants

    ``variant="paper"`` (default) implements Table 2 of the paper with the deviations above.
    ``variant="reference"`` reproduces the geometry of the authors' ranking code
    (``paper/ranking/ELK_Em.py``, ``geometry.py``), with which the published Table 3
    numbers were obtained: the box of :math:`\exists r.C` is computed by plain interval
    arithmetic (no empty special case), and the ``gci2`` loss gets the overlap term
    :math:`\ell_{\cap+}(R_r, B_D)` that the code adds (``nf3_loss``) but Table 2 does not.

    :param nb_ont_classes: Number of classes.
    :param nb_rels: Number of object properties.
    :param nb_inds: Number of individuals (``None`` for no individual embeddings).
    :param embed_dim: Dimension :math:`n`.
    :param margin: Margin :math:`\gamma > 0`.
    :param bot_ids: Indices of the classes fixed to the empty box (:math:`\bot` and the
        classes asserted or derived unsatisfiable).
    :param top_ids: Indices of the classes fixed to the full box (:math:`\top`).
    :param abox_threshold: Target-score threshold :math:`\tau` of corrupted assertions.
    :param epsilon: Lower bound added to class half-widths and role slopes.
    :param variant: ``"paper"`` (default) or ``"reference"`` (see *Variants*).
    """

    VARIANTS = ("paper", "reference")

    #: Only the assertion losses have a distinct negative version (``ReLU(δ - ℓ)``). The
    #: contrastive term for ``gci0``/``gci2`` is built from the margin-free scores of
    #: positives and corruptions by :class:`~mowl.models.elkem.model.ELKEm`.
    neg_capable_gcis = frozenset({"class_assertion", "object_property_assertion"})

    role_axiom_capable = True

    #: ``relu(offset + margin)`` terms: a positive margin requires the inclusion with a gap
    #: (in training mode; ``module.eval()`` uses margin 0).
    margin_semantics = "strict"

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 bot_ids=(), top_ids=(), abox_threshold=0.4, epsilon=1e-5, variant="paper"):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}.")
        self.variant = variant
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.margin = margin
        self.abox_threshold = abox_threshold
        self.epsilon = epsilon

        # Initialisation ranges follow the authors' reference implementation.
        self.class_center = self._uniform(nb_ont_classes, -0.5, 0.5)
        self.class_offset = self._uniform(nb_ont_classes, -0.1, 0.1)
        self.range_center = self._uniform(nb_rels, -0.5, 0.5)
        self.range_offset = self._uniform(nb_rels, -0.1, 0.1)
        self.error_offset = self._uniform(nb_rels, -0.1, 0.1)
        self.slope = self._uniform(nb_rels, 0.9, 1.1)
        self.shift = self._uniform(nb_rels, -0.1, 0.1)
        if nb_inds:
            self.ind_embedding = self._uniform(nb_inds, -0.5, 0.5)
        else:
            self.ind_embedding = None

        self.register_buffer("bot_mask", th.zeros(nb_ont_classes, dtype=th.bool))
        self.register_buffer("top_mask", th.zeros(nb_ont_classes, dtype=th.bool))
        self.set_fixed_classes(bot_ids, top_ids)

    def _uniform(self, num, low, high):
        embed = nn.Embedding(num, self.embed_dim)
        nn.init.uniform_(embed.weight, low, high)
        return embed

    def set_fixed_classes(self, bot_ids=(), top_ids=()):
        """Fix classes to the empty box (``bot_ids``) or the full box (``top_ids``)."""
        self.bot_mask.zero_()
        self.top_mask.zero_()
        bot_ids = th.as_tensor(list(bot_ids), dtype=th.long)
        top_ids = th.as_tensor(list(top_ids), dtype=th.long)
        self.bot_mask[bot_ids.to(self.bot_mask.device)] = True
        self.top_mask[top_ids.to(self.top_mask.device)] = True
        if (self.bot_mask & self.top_mask).any():
            raise ValueError("A class cannot be fixed to both the empty and the full box.")

    # ------------------------------------------------------------------ geometry

    def class_box(self, idx):
        """Lower and upper corners of the boxes of the classes ``idx``."""
        c = self.class_center(idx)
        o = self.class_offset(idx).abs() + self.epsilon
        lower, upper = c - o, c + o
        bot = self.bot_mask[idx].unsqueeze(-1)
        top = self.top_mask[idx].unsqueeze(-1)
        big = th.full_like(c, L.BIG)
        lower = th.where(bot, big, th.where(top, -big, lower))
        upper = th.where(bot, -big, th.where(top, big, upper))
        return lower, upper

    def role(self, idx):
        """Role tuple ``(range_l, range_u, error_l, error_u, slope, shift)`` of ``idx``."""
        rc = self.range_center(idx)
        ro = self.range_offset(idx).abs()
        eo = self.error_offset(idx).abs()
        m = self.slope(idx).abs() + self.epsilon
        return rc - ro, rc + ro, -eo, eo, m, self.shift(idx)

    def individual(self, idx):
        if self.ind_embedding is None:
            raise ValueError("The number of individuals must be specified to use "
                             "assertion losses.")
        return self.ind_embedding(idx)

    def existential_box(self, role_idx, class_idx):
        r"""Box of :math:`\exists r.C` for role indices ``role_idx`` and class indices
        ``class_idx``."""
        return L.existential_box(self.role(role_idx), *self.class_box(class_idx),
                                 sound=self.variant == "paper")

    def _fixed_empty(self, idx):
        return self.bot_mask[idx]

    def _trivial_to_zero(self, loss, trivial):
        r"""Rows of a disjointness axiom with a conjunct fixed to the empty box hold trivially
        (:math:`\bot \sqcap \top \sqsubseteq \bot`). The disjointness primitive would give
        them the margin as cost (the half-widths :math:`-\text{BIG}` and :math:`+\text{BIG}` of
        :math:`\bot` and :math:`\top` cancel), a cost no parameter can remove. The
        ``"reference"`` variant keeps the primitive's value."""
        if self.variant == "reference":
            return loss
        return th.where(trivial, th.zeros_like(loss), loss)

    # ------------------------------------------------------------------ losses

    def _current_margin(self):
        return self.margin if self.training else 0.0

    def _loss(self, gci_name, data, margin, neg=False):
        if gci_name == "gci0":
            return L.inclusion_loss(*self.class_box(data[:, 0]), *self.class_box(data[:, 1]),
                                    margin)
        if gci_name == "gci0_bot":
            c = data[:, 0]
            o = self.class_offset(c).abs() + self.epsilon
            loss = F.relu(o + margin).mean(dim=-1)
            return th.where(self._fixed_empty(c), th.zeros_like(loss), loss)
        if gci_name == "gci1":
            l1, u1 = self.class_box(data[:, 0])
            l2, u2 = self.class_box(data[:, 1])
            il, iu = L.intersection(l1, u1, l2, u2)
            incl = L.inclusion_loss(il, iu, *self.class_box(data[:, 2]), margin)
            over = L.overlap_loss(l1, u1, l2, u2, margin)
            trivial = self._fixed_empty(data[:, 0]) | self._fixed_empty(data[:, 1])
            return incl + th.where(trivial, th.zeros_like(over), over)
        if gci_name == "gci1_bot":
            loss = L.disjoint_loss(*self.class_box(data[:, 0]), *self.class_box(data[:, 1]),
                                   margin)
            return self._trivial_to_zero(
                loss, self._fixed_empty(data[:, 0]) | self._fixed_empty(data[:, 1]))
        if gci_name == "gci2":
            el, eu = self.existential_box(data[:, 1], data[:, 2])
            incl = L.inclusion_loss(*self.class_box(data[:, 0]), el, eu, margin)
            if self.variant == "reference":
                rl, ru = self.role(data[:, 1])[:2]
                incl = incl + L.overlap_loss(rl, ru, *self.class_box(data[:, 2]), margin)
            return incl
        if gci_name == "gci3":
            role = self.role(data[:, 0])
            cl, cu = self.class_box(data[:, 1])
            el, eu = L.existential_box(role, cl, cu, sound=self.variant == "paper")
            incl = L.inclusion_loss(el, eu, *self.class_box(data[:, 2]), margin)
            over = L.overlap_loss(role[0], role[1], cl, cu, margin)
            trivial = self._fixed_empty(data[:, 1])
            return incl + th.where(trivial, th.zeros_like(over), over)
        if gci_name == "gci3_bot":
            role = self.role(data[:, 0])
            loss = L.disjoint_loss(role[0], role[1], *self.class_box(data[:, 1]), margin)
            return self._trivial_to_zero(loss, self._fixed_empty(data[:, 1]))
        if gci_name == "role_inclusion":
            return L.role_inclusion_loss(self.role(data[:, 0]), self.role(data[:, 1]), margin)
        if gci_name == "role_chain":
            return L.role_chain_loss(self.role(data[:, 0]), self.role(data[:, 1]),
                                     self.role(data[:, 2]), margin)
        if gci_name == "class_assertion":
            # Column order of mOWL's ClassAssertionDataset: [individual, class].
            loss = L.point_membership_loss(self.individual(data[:, 0]),
                                           *self.class_box(data[:, 1]), margin)
        elif gci_name == "object_property_assertion":
            # Column order of ObjectPropertyAssertionDataset: [subject, property, object].
            loss = L.role_assertion_loss(self.individual(data[:, 0]), self.role(data[:, 1]),
                                         self.individual(data[:, 2]), margin)
        else:
            raise ValueError(f"Unknown normal form {gci_name!r}.")
        if neg:
            return L.hinge_negative_loss(loss, self.abox_threshold)
        return loss

    def zero_margin_loss(self, data, gci_name):
        r"""Margin-free loss :math:`\ell^0(\alpha)` of each row, whatever the mode."""
        loss = self._loss(gci_name, data, 0.0)
        if self.per_axiom_column and loss.ndim == 1:
            return loss.unsqueeze(-1)
        return loss

    def score(self, data, gci_name):
        r"""Score :math:`s(\alpha) = \exp(-\ell^0(\alpha)) \in (0, 1]` (Eq. 4 of the paper)."""
        return th.exp(-self.zero_margin_loss(data, gci_name))

    def gci0_loss(self, data, neg=False):
        return self._loss("gci0", data, self._current_margin(), neg)

    def gci0_bot_loss(self, data, neg=False):
        return self._loss("gci0_bot", data, self._current_margin(), neg)

    def gci1_loss(self, data, neg=False):
        return self._loss("gci1", data, self._current_margin(), neg)

    def gci1_bot_loss(self, data, neg=False):
        return self._loss("gci1_bot", data, self._current_margin(), neg)

    def gci2_loss(self, data, neg=False):
        return self._loss("gci2", data, self._current_margin(), neg)

    def gci3_loss(self, data, neg=False):
        return self._loss("gci3", data, self._current_margin(), neg)

    def gci3_bot_loss(self, data, neg=False):
        return self._loss("gci3_bot", data, self._current_margin(), neg)

    def role_inclusion_loss(self, data, neg=False):
        return self._loss("role_inclusion", data, self._current_margin(), neg)

    def role_chain_loss(self, data, neg=False):
        return self._loss("role_chain", data, self._current_margin(), neg)

    def class_assertion_loss(self, data, neg=False):
        return self._loss("class_assertion", data, self._current_margin(), neg)

    def object_property_assertion_loss(self, data, neg=False):
        return self._loss("object_property_assertion", data, self._current_margin(), neg)

    # ------------------------------------------------------------------ regularisers

    def size_regularization_loss(self, class_cap=0.25, range_cap=0.25, error_cap=0.05):
        r"""Box-size penalty of the authors' reference implementation (not described in the
        paper): :math:`\mathrm{mean}\,\mathrm{ReLU}(o - \mathrm{cap})^2` over the half-widths
        of the trained class boxes, role ranges and role error boxes."""
        free = ~(self.bot_mask | self.top_mask)
        class_o = self.class_offset.weight[free].abs()
        terms = [F.relu(self.range_offset.weight.abs() - range_cap).pow(2).mean(),
                 F.relu(self.error_offset.weight.abs() - error_cap).pow(2).mean()]
        if class_o.numel():
            terms.append(F.relu(class_o - class_cap).pow(2).mean())
        return sum(terms)

    def lexical_loss(self, ind_idx, features, decay=1.0):
        r"""Lexical regularisation loss (Eq. 6 of the paper) of the individuals ``ind_idx``,
        whose external representations are the rows of ``features``."""
        return L.lexical_loss(self.individual(ind_idx), features, decay)
