r"""Embed2Reason (E2R): quantum-logic subspace embeddings (Garg et al., NeurIPS 2019)."""
from functools import partial

import torch as th

from mowl.models.e2r.common import E2RBase, _THING, _NOTHING  # noqa: F401
from mowl.nn.alc.e2r.module import E2RModule
from mowl.representation.library import logic_subspace_algebra
from mowl.representation.library.logic_subspace import Orthogonality, Separation, SubspaceStructure
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.graph import EdgeCorruption, KnownRowMask
from mowl.training.loop import RegimeTrained, Rows
from mowl.training.losses import SumLoss
from mowl.training.regularization import IntervalProjection, Regularization


class E2R(RegimeTrained, E2RBase):
    r"""Embed2Reason [garg2019]_: concepts and relations as axis-parallel linear subspaces,
    entities as vectors, the logic is quantum logic (Sec. 3).

    The space :math:`\mathbb{C}^d` over the field :math:`\mathbb{R}` is simulated by
    :math:`\mathbb{R}^{2d}` (Sec. 4). A concept is the span of the axes selected by a
    learnable indicator :math:`y \in [0, 1]^{2d}` (real half only), a relation the span
    selected by :math:`z` (both halves), an entity a vector :math:`x = (x_R, 0)` and an
    ordered pair :math:`(a, b)` the vector :math:`(x_{aR}, x_{bR})`. :math:`C(a)` holds iff
    :math:`x_a \in C`, :math:`R(a, b)` iff :math:`x_{ab} \in R`. :math:`\sqcap` is
    intersection, :math:`\sqcup` the span (sum) of subspaces, :math:`\neg` the orthogonal
    complement, :math:`\sqsubseteq` inclusion. The loss is eq. (10): the sum of the terms
    of eqs. (1)-(9), each averaged over its assertions, plus the regularisers of eqs. (11)
    and (12) and the negated-membership negatives of Sec. 4 (3 per positive, Sec. 5).

    Axioms used for training (everything else is listed in :attr:`skipped_axioms`):

    ======================================================  ===========================
    OWL axiom                                               loss
    ======================================================  ===========================
    ``ClassAssertion(A, a)``                                eq. (4)
    ``ObjectPropertyAssertion(R, a, b)``                    eq. (4)
    ``SubClassOf(A, B)``, ``SubClassOf(A, B1 ⊓ … ⊓ Bk)``    eq. (5) (per conjunct)
    ``EquivalentClasses(A, B)``                             eq. (5) both directions
    ``EquivalentClasses(A, B1 ⊓ ... ⊓ Bk)``                 eq. (6)
    ``EquivalentClasses(A, B1 ⊔ ... ⊔ Bk)``                 eq. (7)
    ``EquivalentClasses(A, ¬B)``                            eq. (8) (real half)
    ``DisjointClasses(A, B)``, ``SubClassOf(A, ¬B)``        eq. (6) with A ⊓ B = ⊥ = {0}
    ``SubObjectPropertyOf(R, S)``, equivalences             eq. (5) for relations
    ``ObjectPropertyRange(R, C)``, ``⊤ ⊑ ∀R.C``             eq. (9)
    ======================================================  ===========================

    All class and property operands must be named. The paper's universal type
    restriction (Sec. 4) is the statement "every :math:`R`-successor is in :math:`C_j`",
    i.e. a range axiom; eq. (9) is summed over the non-descendants :math:`C_k` of
    :math:`C_j` chosen by the paper's shrinking rule (children of :math:`\top` and children
    of ancestors of :math:`C_j` that are not ancestors of :math:`C_j`), computed from the
    told class hierarchy. Existential restrictions have no E2R loss (the paper defines
    :math:`\exists R.C` in Sec. 2 but gives none) and are skipped. Role conjunction,
    disjunction and negation (eqs. (6)-(8) for relations) are implemented in the module,
    but OWL 2 has no syntax for them, so the model never produces them.

    Deviations from the paper (see also :mod:`mowl.baselines.nn.alc.e2r.losses`):

    * the second term of eq. (4) is omitted (as printed it is not zero in the situation
      where the text says it is); pair vectors are built from entity vectors, so the
      pair-structure terms of eq. (4) vanish identically;
    * role assertions also carry the unit-length terms of both entities;
    * concept negation (eq. (8)) is evaluated on the real half (otherwise it can never
      be zero together with eq. (2));
    * negatives: only the residual term of the membership loss is negated, the
      unit-length term is kept, so the objective is bounded below; corrupted assertions
      that are asserted in the training ontology are dropped;
    * indicators are clipped to :math:`[0, 1]` after every step; the paper does not say how
      they are kept in range, nor how parameters are initialised, how the terms are
      weighted against each other, or whether eqs. (11)-(12) were used together with
      the negatives (Sec. 4 offers the negatives as an alternative, Sec. 5 uses
      "regularization terms as described in Section 4" and negatives).

    :param dataset: mOWL dataset (training ontology with TBox and ABox).
    :param embed_dim: :math:`d` (vectors have :math:`2d` real coordinates). Defaults to
        ``100`` (Sec. 5).
    :param learning_rate: Adam learning rate (not stated in the paper). Defaults to
        ``0.01``.
    :param batch_size: Rows per axiom kind and step. Defaults to ``4096``.
    :param num_negatives: Corrupted assertions per positive membership assertion.
        Defaults to ``3`` (Sec. 5).
    :param negative_weight: Weight of the negated-membership term. Defaults to ``1``.
    :param reg_orth: Weight of :math:`\Omega_{\text{orth}}` (eq. (11)). Defaults to ``1``.
    :param reg_sep: Weight of :math:`\Omega_{\text{sep}}` (eq. (12)). Defaults to ``1``.
    :param sep_gap: Constant of eq. (12); ``None`` means :math:`\sqrt{d}` as in the paper.
    :param range_negatives: ``"shrunk"`` (the paper's reduced set of non-descendants) or
        ``"all"`` (every class that is neither an ancestor nor a descendant of
        :math:`C_j`).
    :param max_sibling_pairs: Upper bound on the sibling pairs used by eq. (11); larger
        sets are subsampled once. Defaults to ``100000``.
    :param role_unit_length: See :class:`~mowl.baselines.nn.alc.e2r.module.E2RModule`.
    :param seed: Seed of the Python random generator used for subsampling.
    """

    _algebra = logic_subspace_algebra

    def init_module(self):
        self.module = E2RModule(
            len(self._iid), len(self._cid), len(self._rid), dim=self.embed_dim,
            top_index=self._cid.get(_THING), bottom_index=self._cid.get(_NOTHING),
            role_unit_length=self.role_unit_length).to(self.device)
        # the terms of the objective that are not axioms: structure of the indicators (eqs.
        # (1)-(3)), siblings orthogonal (eq. (11)), separation of a class from its parents
        # (eq. (12)); the pairs are read from the told hierarchy
        self.structure = SubspaceStructure()
        self.orthogonality = Orthogonality(self.reg_orth, self.sibling_pairs,
                                           self.role_sibling_pairs)
        self.separation = Separation(self.reg_sep, self.sep_gap, self.separation_pairs,
                                     self.role_separation_pairs)
        self.module.regularizers = (self.structure, self.orthogonality, self.separation)

    # -- training -----------------------------------------------------------------------
    @property
    def default_regimes(self):
        """The regimes that make the training of E2R (Sec. 4 and 5 of the paper): a random
        batch per axiom kind and step (:class:`~mowl.training.PerNormalFormMiniBatches`,
        an epoch has :math:`\\max_k \\lceil N_k / \\text{batch size} \\rceil` steps),
        ``num_negatives`` corruptions of every assertion that replace an individual
        (:class:`~mowl.training.EdgeCorruption`) minus the known assertions
        (:class:`~mowl.training.KnownRowMask` with ``drop``), the sum of the axiom losses and
        ``negative_weight`` times the negated-membership losses (:class:`~mowl.training.SumLoss`),
        and the projection of the indicators into :math:`[0, 1]`
        (:class:`~mowl.training.Regularization` of an
        :class:`~mowl.training.IntervalProjection`). The regularisers of eqs. (1)-(3), (11) and
        (12) are those of the module (``module.regularizers``)."""
        assertions = ("class_assertion", "role_assertion")
        parts = [partial(PerNormalFormMiniBatches, self.batch_size, "replacement")]
        n_ind = len(self._iid)
        if min(self.num_negatives, self.negative_weight, n_ind - 1) > 0:
            parts += [
                partial(EdgeCorruption, self.num_negatives, (0,), n_ind, ("class_assertion",),
                        "block"),
                partial(EdgeCorruption, self.num_negatives, (0, 2), n_ind, ("role_assertion",),
                        "block", False),
                partial(KnownRowMask, assertions, False, True)]
        parts.append(partial(SumLoss, self.negative_weight))
        parts.append(partial(Regularization, IntervalProjection(
            ("class_model.centres", "relation_model.vectors"), 0.0, 1.0)))
        return tuple(parts)

    @property
    def training_datasets(self):
        """One dataset per axiom kind and arity of the training rows (``training_rows``)."""
        return {key: Rows(rows) for key, rows in self.training_rows.items()}

    def score_rows(self, form, rows, neg=False):
        return self.module(self._kind(form), rows, neg=neg)

    def compute_loss(self, full=True):
        """The E2R objective (eq. (10) with regularisers and negatives) on all rows as
        training computes it; without negatives when ``num_negatives`` is 0."""
        return self.batch_loss(self.training_datasets, negatives=self.num_negatives > 0)
