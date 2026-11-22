"""OnT: Ontology Transformer encoders (Yang, Chen, He, Gao, Horrocks, ISWC 2025)."""

import logging
from functools import partial

from mowl.models.hit.model import HiT
from mowl.models.ont.common import DEFAULT_CENTRI_GRID, OnTBase  # noqa: F401
from mowl.training.text import SampledLogicalLosses

logger = logging.getLogger(__name__)


class OnT(OnTBase, HiT):
    """Ontology Transformer encoder (OnT) [yang2025ont]_.

    OnT extends :class:`~mowl.models.hit.HiT` from taxonomies to :math:`\\mathcal{EL}`
    (Section 4). Every concept, atomic or complex, is the point of the Poincaré ball
    obtained by encoding its verbalisation (Section 4.1, :class:`mowl.text.Verbaliser`);
    every role is a rotation (optionally followed by a Möbius scaling) whose parameters
    are a linear function of the encoding of the role's label (Section 4.2, Eq. 4). The
    training loss (Section 4.3) is the sum of

    * :math:`L_\\prec(x_C \\prec x_D)` (Eq. 5, HiT's clustering and centripetal losses)
      for every axiom :math:`C \\sqsubseteq D` of ``dataset.ontology``
      (``EquivalentClasses`` give both directions), with one random negative per axiom:
      a random class, or :math:`\\exists r.X` for a random class :math:`X` when
      :math:`D = \\exists r.D'` (as in the OnT data release);
    * :math:`L_r` (Eq. 6) for every occurrence of :math:`\\exists r.D`, aligning its
      verbalisation with :math:`f_r(x_D)`;
    * :math:`L_\\sqcap` (Eq. 7) for every occurrence of :math:`C_1 \\sqcap \\dots \\sqcap
      C_n` (mean over the conjuncts; Eq. 7 for :math:`n = 2`).

    As in the reference implementation, every training step takes a batch of
    subsumption axioms and an equally large random sample of existential restrictions
    and of conjunctions, whose losses use the negatives of the subsumption batch
    (randomly permuted); the losses are averaged over the batch (the paper writes sums).

    Defaults are those of Section 5.1 and the released configuration:
    all-MiniLM-L12-v2, :math:`\\alpha = 3.0`, :math:`\\beta = 0.5`, learning rate
    :math:`10^{-5}`, one epoch, batch size 64, one negative per axiom, 500 warm-up steps,
    :math:`\\lambda` chosen on validation subsumptions by MRR.

    The ontology is used as it is: OnT is normally trained on an ontology normalised by
    its own name-preserving normaliser (``normalization/`` of the reference code), which
    mOWL does not reimplement; the OnT benchmark datasets
    (:mod:`mowl.datasets.builtin.yang2025ont`) are already normalised. Axioms with
    ``owl:Nothing`` or with ``owl:Thing`` as superclass, and axioms outside
    :math:`\\mathcal{EL}`, are skipped (and counted in :attr:`skipped_axioms`).

    :param use_roles: ``False`` trains the ablation OnT(w/o r) (only Eq. 5).
    :param conjunction_weight: Weight of :math:`L_\\sqcap`.
    :param existential_weight: Weight of :math:`L_r`.
    :param role_pairing: ``"half"`` (reference code) or ``"adjacent"`` (paper's matrix),
        see :func:`mowl.nn.lm.hyperbolic.rotate`.
    :param role_scaling: Apply the scaling :math:`k_r` of Eq. 4 (the reference code does
        not).
    The model is the combination of parts that the method ``ont`` defines (see
    :class:`~mowl.models.hit.HiT` for the parts of HiT): a relation model
    (:class:`~mowl.representation.library.hierarchy_relations.TextRotation`) and the
    existential and conjunction losses of its geometry
    (:class:`~mowl.representation.library.hierarchy_geometry.TextGeometry`). The reference
    implementation that the combination is tested against is ``mowl.baselines.models.ont``.

    :param existential_mode: How :math:`\\exists r.D` is embedded when scoring:
        ``"text"`` (the encoding of its verbalisation; the reference evaluation's default
        "sentence" mode) or ``"role"`` (:math:`f_r(x_D)`, the "constructed" mode).

    The other parameters are those of :class:`~mowl.models.hit.HiT`.
    """

    def init_module(self):
        """The module is the combination of parts that the method ``ont`` defines
        (:func:`mowl.representation.library.hierarchy_methods.ont`): HiT's class model and
        measure and, unless ``use_roles`` is ``False``, a relation model that rotates the point
        of a filler by angles computed from the text of the role."""
        from mowl.nn.lm.ont import OnTModule
        return OnTModule(self.encoder, curvature=self.curvature,
                         clustering_margin=self.clustering_margin,
                         centripetal_margin=self.centripetal_margin,
                         clustering_weight=self.clustering_weight,
                         centripetal_weight=self.centripetal_weight,
                         use_roles=self.use_roles,
                         conjunction_weight=self.conjunction_weight,
                         existential_weight=self.existential_weight,
                         role_pairing=self.role_pairing, role_scaling=self.role_scaling)

    # -- training ----------------------------------------------------------------------------
    @property
    def default_regimes(self):
        """The regimes of HiT and, with ``use_roles``, the logical losses of OnT added to every
        step (:class:`~mowl.training.text.SampledLogicalLosses`)."""
        regimes = list(HiT.default_regimes.fget(self))
        if self.use_roles:
            regimes.insert(1, partial(SampledLogicalLosses, self.existential_weight,
                                      self.conjunction_weight))
        return tuple(regimes)
