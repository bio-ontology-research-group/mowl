"""CanonicalMixtureModule: the canonical-element mixture, built from parts."""
from mowl.representation.library.hierarchy_modules import FlatLossOption
from mowl.representation.library.mixture_individuals import MAX_SUPPORT_ENTRIES
from mowl.representation.library.mixture_methods import MixtureGeometry, canonical_mixture

__all__ = ["CanonicalMixtureModule", "MAX_SUPPORT_ENTRIES"]


class CanonicalMixtureModule(FlatLossOption, MixtureGeometry):
    """The canonical-element mixture: individuals as distributions over the elements of the
    canonical model, the combination that
    :func:`mowl.representation.library.mixture_methods.canonical_mixture` names (read
    ``module.parts``).

    Every individual :math:`a` is a distribution
    :math:`p_a = \\operatorname{softmax}_{S_a}(U_a V^\\top + b + \\delta_a)` over its
    admissible support :math:`S_a` (the elements of the canonical model that belong to every
    class entailed for :math:`a`), so that every entailed class assertion has probability 1
    (score exactly ``0.0``), every contradicted one probability 0 (score ``+inf``) and
    :math:`O \\models C \\sqsubseteq D` implies ``score(a, D) <= score(a, C)``, for every value
    of the parameters. Classes and roles are not learned: they are the element sets and the
    pairs of the constructed canonical model.

    The earlier hand-written module is kept for comparison as
    ``mowl.baselines.nn.el.canonical_mixture.CanonicalMixtureModule``.

    :param structure: The constructed model, a
        :class:`~mowl.geometric.membership.MembershipStructure`.
    :param rank: Rank :math:`r` of the logits.
    :param entropy_reg: Weight of the entropy regulariser :math:`\\lambda \\operatorname{mean}_a
        H(p_a)` (:meth:`regularization_loss` of a batch of individuals).
    :param init_self_mass: Initial mass :math:`\\alpha` of an individual's own element.
    :param training_class_assertions: ``(N, 2)`` rows, checked to be entailed by ``structure``.

    As every :class:`~mowl.nn.ELModule`, a form it does not read (the role axioms) raises
    :class:`NotImplementedError`, and ``per_axiom_column = False`` gives the flat ``(B,)``
    losses of the earlier module.
    """

    def __init__(self, structure, rank=50, entropy_reg=0.0, init_self_mass=0.9,
                 training_class_assertions=None):
        from mowl.geometric.membership import MembershipStructure
        if not isinstance(structure, MembershipStructure):
            raise TypeError(
                f"structure must be a MembershipStructure, got {type(structure).__name__}")
        n_ind = int(structure.individual_element.numel())
        if n_ind == 0:
            raise ValueError("CanonicalMixture needs individuals")
        built = canonical_mixture(
            structure.M.shape[1], len(structure.roles), n_ind, embed_dim=rank,
            structure=structure, entropy_reg=entropy_reg, init_self_mass=init_self_mass,
            training_class_assertions=training_class_assertions, geometry=_Parts)
        MixtureGeometry.__init__(self, *built.args, **built.kwargs)
        self.margin_semantics = None


class _Parts:
    """The arguments the builder would give the geometry (so that the module is the
    builder's combination, whatever class it is)."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
