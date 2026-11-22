r"""Embed2Reason (E2R): quantum-logic subspace embeddings (Garg et al., NeurIPS 2019)."""
from mowl.models.composed.subspace import ComposedSubspace, _THING, _NOTHING  # noqa: F401
from mowl.nn.alc.e2r.module import E2RModule


class E2R(ComposedSubspace):
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

    def __init__(self, dataset, embed_dim=100, learning_rate=0.01, batch_size=4096,
                 num_negatives=3, negative_weight=1.0, reg_orth=1.0, reg_sep=1.0,
                 sep_gap=None, range_negatives="shrunk", max_sibling_pairs=100000,
                 role_unit_length=True, model_filepath=None, device="cpu", seed=0):
        super().__init__(
            dataset, method="e2r", embed_dim=embed_dim, learning_rate=learning_rate,
            batch_size=batch_size, num_negatives=num_negatives,
            negative_weight=negative_weight, reg_orth=reg_orth, reg_sep=reg_sep,
            sep_gap=sep_gap, range_negatives=range_negatives,
            max_sibling_pairs=max_sibling_pairs, role_unit_length=role_unit_length,
            model_filepath=model_filepath, device=device, seed=seed)

    def init_module(self):
        """The method ``"e2r"`` as :class:`~mowl.nn.E2RModule` (its parts are those of
        :func:`~mowl.representation.library.logic_subspace.e2r`), with the regularisers of
        eqs. (1)-(3), (11) and (12)."""
        self._attach_regularizers(E2RModule(
            len(self._iid), len(self._cid), len(self._rid), dim=self.embed_dim,
            top_index=self._cid.get(_THING), bottom_index=self._cid.get(_NOTHING),
            role_unit_length=self.role_unit_length))
