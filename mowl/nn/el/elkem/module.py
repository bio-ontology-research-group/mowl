r"""Neural module of ELK-Em [singh2026]_."""

import torch as th

import mowl.nn.el.elkem.losses as L
from mowl.nn.el.part_built import _PartBuiltModule


class ELKEmModule(_PartBuiltModule):
    r"""ELK-Em: closure-aware box embeddings for :math:`\mathcal{EL}^+_\bot` [singh2026]_.

    The module is a :class:`~mowl.representation.ComposedGeometry` made of the parts that the
    builder ``elkem`` of :mod:`mowl.representation.library.boxrel_methods` names (boxes with
    classes fixed to the empty or the full box, a relation with a range box, an error box and
    an affine map, bounds averaged over the coordinates, points for individuals, the ELK-Em
    objective); ``module.parts`` says which. The earlier hand-written module, which the tests
    compare this one with, is ``mowl.baselines.nn.el.elkem.module.ELKEmModule``; checkpoints of
    that module load here.

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
      would charge the margin for :math:`\bot \sqcap \top \sqsubseteq \bot`, a cost
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

    neg_capable_gcis = frozenset({"class_assertion", "object_property_assertion"})
    role_axiom_capable = True

    #: With ``per_axiom_column = False`` every loss is ``(B,)``, as the hand-written module
    #: returned them.
    flat_forms = frozenset({"gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
                            "class_assertion", "object_property_assertion", "role_inclusion",
                            "role_chain"})

    state_dict_aliases = {"class_center.weight": "class_model.centres.weight",
                          "class_offset.weight": "class_model.offsets.weight",
                          "bot_mask": "class_model.bot_mask",
                          "top_mask": "class_model.top_mask",
                          "range_center.weight": "relation_model.range_centres.weight",
                          "range_offset.weight": "relation_model.range_offsets.weight",
                          "error_offset.weight": "relation_model.error_offsets.weight",
                          "slope.weight": "relation_model.slopes.weight",
                          "shift.weight": "relation_model.vectors.weight",
                          "ind_embedding.weight": "individual_model.centres.weight"}

    def __init__(self, nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, margin=0.1,
                 bot_ids=(), top_ids=(), abox_threshold=0.4, epsilon=1e-5, variant="paper"):
        from mowl.representation.library.boxrel_methods import elkem
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}.")
        built = elkem(nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                      margin=margin, bot_ids=bot_ids, top_ids=top_ids,
                      abox_threshold=abox_threshold, epsilon=epsilon, variant=variant)
        super().__init__(built)
        self.variant = variant
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim

    # -- the names of the earlier module -----------------------------------------------------
    @property
    def class_center(self):
        return self.class_model.centres

    @property
    def class_offset(self):
        return self.class_model.offsets

    @property
    def range_center(self):
        return self.relation_model.range_centres

    @property
    def range_offset(self):
        return self.relation_model.range_offsets

    @property
    def error_offset(self):
        return self.relation_model.error_offsets

    @property
    def slope(self):
        return self.relation_model.slopes

    @property
    def shift(self):
        return self.relation_model.vectors

    @property
    def ind_embedding(self):
        return None if self.individual_model is None else self.individual_model.centres

    @property
    def bot_mask(self):
        return self.class_model.bot_mask

    @property
    def top_mask(self):
        return self.class_model.top_mask

    @property
    def abox_threshold(self):
        return self.objective.abox_threshold

    @abox_threshold.setter
    def abox_threshold(self, value):
        self.objective.abox_threshold = float(value)

    @property
    def epsilon(self):
        """Lower bound added to the half-widths of classes and to the slopes of roles."""
        return self.class_model.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.class_model.epsilon = value
        self.relation_model.eps = value

    def set_fixed_classes(self, bot_ids=(), top_ids=()):
        """Fix classes to the empty box (``bot_ids``) or the full box (``top_ids``)."""
        self.class_model.fix(bot_ids, top_ids)

    # ------------------------------------------------------------------ geometry

    def class_box(self, idx):
        """Lower and upper corners of the boxes of the classes ``idx``."""
        box = self.class_model(idx)
        return box.lower, box.upper

    def role(self, idx):
        """Role tuple ``(range_l, range_u, error_l, error_u, slope, shift)`` of ``idx``."""
        rel = self.relation_model
        box, error = rel.range(idx), rel.error(idx)
        return box.lower, box.upper, -error, error, rel.slope(idx), rel.vectors(idx)

    def individual(self, idx):
        if self.ind_embedding is None:
            raise ValueError("The number of individuals must be specified to use "
                             "assertion losses.")
        return self.ind_embedding(idx)

    def existential_box(self, role_idx, class_idx):
        r"""Box of :math:`\exists r.C` for role indices ``role_idx`` and class indices
        ``class_idx``."""
        box = self.exists(role_idx, self.classes(class_idx))
        return box.lower, box.upper

    # ------------------------------------------------------------------ losses

    def _current_margin(self):
        return self.margin if self.training else 0.0

    def zero_margin_loss(self, data, gci_name):
        r"""Margin-free loss :math:`\ell^0(\alpha)` of each row, whatever the mode."""
        margin = self.objective.margin
        self.objective.margin = 0.0
        try:
            loss = self.cost(gci_name, data)
        finally:
            self.objective.margin = margin
        if self.per_axiom_column and loss.ndim == 1:
            return loss.unsqueeze(-1)
        return loss

    def score(self, data, gci_name):
        r"""Score :math:`s(\alpha) = \exp(-\ell^0(\alpha)) \in (0, 1]` (Eq. 4 of the paper)."""
        return th.exp(-self.zero_margin_loss(data, gci_name))

    # ------------------------------------------------------------------ regularisers

    def size_regularization_loss(self, class_cap=0.25, range_cap=0.25, error_cap=0.05):
        r"""Box-size penalty of the authors' reference implementation (not described in the
        paper): :math:`\mathrm{mean}\,\mathrm{ReLU}(o - \mathrm{cap})^2` over the half-widths
        of the trained class boxes, role ranges and role error boxes (the part
        :class:`~mowl.representation.library.boxrel_regularizers.SizeCapPenalty`)."""
        from mowl.representation.library.boxrel_regularizers import SizeCapPenalty
        return SizeCapPenalty(class_cap, range_cap, error_cap).penalty(self)

    def lexical_loss(self, ind_idx, features, decay=1.0):
        r"""Lexical regularisation loss (Eq. 6 of the paper) of the individuals ``ind_idx``,
        whose external representations are the rows of ``features``."""
        return L.lexical_loss(self.individual(ind_idx), features, decay)


__all__ = ["ELKEmModule"]
