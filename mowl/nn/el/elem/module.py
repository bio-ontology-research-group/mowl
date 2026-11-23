"""ELEmbeddings as a combination of parts (:class:`ELEmModule`).

The module is not a hand-written loss: it is the geometry that the method ``"elem"`` of
:mod:`mowl.representation.methods` builds (balls, translations, the centre-distance measure,
the ELEmbeddings objective, the unit-norm regulariser), wrapped in the interface that the
earlier module had, so that every attribute, option and training regime that worked on it
still works. The direct, hand-written module is kept for comparison in
:mod:`mowl.baselines.nn.el.elem`.
"""

from deprecated.sphinx import versionchanged

from mowl.nn.el.elmodule import ELModule
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.methods import METHODS
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["ELEmModule"]


class _NoIndividuals(ConstructorUnavailable, ValueError):
    """The module has no individuals (a :class:`ValueError`, as the earlier module raised)."""


#: The forms with a unit-norm penalty of the class centres, per axiom, in ELEmbeddings' code.
_PER_AXIOM_FORMS = ("gci0", "gci1", "gci2", "gci3", "gci1_bot")


class _PartBuiltModule(ComposedGeometry):
    """The interface of the earlier hand-written EL modules, on a geometry that is built
    from parts: what the modules of ELEmbeddings, EmEL++, EmEL(var) and RegD have in common.

    A subclass calls :meth:`_assemble` with the geometry its method builds. The module then
    *is* that geometry (same parts, same parameters), with the attributes and options of the
    earlier module (the DELE negative losses, the margin, the tables under their old names)
    and a state dict that can be loaded by either naming of the tables.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})

    #: Normal forms that get a negative loss with ``all_negative_losses=True``.
    DELE_NEG_GCIS = frozenset({"gci0", "gci1", "gci2", "gci3",
                               "gci0_bot", "gci1_bot", "gci3_bot"})

    #: These modules have no losses for the EL++ role axioms (EmEL++ has).
    role_axiom_capable = False

    # The per-axiom loss functions are looked up as methods (``gci0_loss``, ...), so that a
    # regime that replaces one of them on the module (the DELE code losses) takes effect.
    get_loss_function = ELModule.get_loss_function

    def _assemble(self, geometry, nb_ont_classes, nb_rels, nb_inds, embed_dim,
                  all_negative_losses=False):
        """Make this module the geometry ``geometry`` (the parts it was built from)."""
        ComposedGeometry.__init__(
            self, geometry.class_model, geometry.relation_model, geometry.measure,
            geometry.objective, individuals=geometry.individual_model,
            regularizers=geometry.regularizers)
        self.neg_capable_gcis = frozenset(type(self).neg_capable_gcis)
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds or None
        self.embed_dim = embed_dim
        self._all_negative_losses = False
        # a module refuses the negative loss of a form that has none
        self.strict_negatives = True
        if all_negative_losses:
            self.enable_all_negative_losses()

    def role_inclusion_loss(self, gci, neg=False):
        if not self.role_axiom_capable:  # the module has no loss for it (the stub says so)
            return ELModule.role_inclusion_loss(self, gci, neg)
        return super().role_inclusion_loss(gci, neg)

    def role_chain_loss(self, gci, neg=False):
        if not self.role_axiom_capable:
            return ELModule.role_chain_loss(self, gci, neg)
        return super().role_chain_loss(gci, neg)

    def individuals(self, ids):
        if self.individual_model is None:
            raise _NoIndividuals(
                "The number of individuals must be specified to use this loss function.")
        return super().individuals(ids)

    def _objectives(self):
        """The objective and, for an objective that wraps a host, the host."""
        host = getattr(self.objective, "host", None)
        return (self.objective,) if host is None else (self.objective, host)

    @property
    def epsilon(self):
        """The :math:`\\varepsilon` of the DELE negative losses of the bottom forms."""
        return self.objective.epsilon

    @epsilon.setter
    def epsilon(self, value):
        for objective in self._objectives():
            objective.epsilon = float(value)

    @property
    def all_negative_losses(self):
        """Whether the DELE negative losses of all normal forms are enabled."""
        return self._all_negative_losses

    @all_negative_losses.setter
    def all_negative_losses(self, value):
        self._all_negative_losses = bool(value)
        if value:
            self._objective_negatives(self.DELE_NEG_GCIS | self.neg_capable_gcis)

    def _objective_negatives(self, forms):
        for objective in self._objectives():
            objective.neg_forms = frozenset(forms)
            objective.negatives = "forms"
        outer = self.objective
        if getattr(outer, "host", None) is not None:
            outer.neg_forms = outer.host.neg_forms | (
                {"gci0"} & getattr(outer, "regd_forms", frozenset()))

    def enable_all_negative_losses(self, epsilon=None):
        """Switch on the DELE negative losses of all normal forms (see the class \
parameter ``all_negative_losses``) and extend :attr:`neg_capable_gcis` accordingly.

        :param epsilon: New value of :math:`\\varepsilon`, or ``None`` to keep the current one.
        :type epsilon: float, optional
        """
        if epsilon is not None:
            self.epsilon = epsilon
        self.neg_capable_gcis = frozenset(type(self).neg_capable_gcis | self.DELE_NEG_GCIS)
        self.all_negative_losses = True

    def _dele_neg(self, neg):
        return neg and self.all_negative_losses

    # -- loading ----------------------------------------------------------------------------

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """As :meth:`torch.nn.Module.load_state_dict`; a state dict that names a table by one
        of its names (``class_embed.weight`` of the earlier module,
        ``class_model.centres.weight`` of the parts) fills it under every name."""
        return super().load_state_dict(self._with_aliases(state_dict), strict=strict,
                                       assign=assign)

    def _with_aliases(self, state_dict):
        groups = {}
        for name, tensor in self.state_dict(keep_vars=True).items():
            groups.setdefault(id(tensor), []).append(name)
        state_dict = dict(state_dict)
        for names in groups.values():
            given = next((n for n in names if n in state_dict), None)
            if given is not None:
                for name in names:
                    state_dict.setdefault(name, state_dict[given])
        return state_dict


@versionchanged(version="0.4.0", reason="The class ELEmModule receives an optional parameter nb_inds")
class ELEmModule(_PartBuiltModule):
    """
    ELEmbeddings from [kulmanov2019]_, as the combination of parts that the method ``"elem"``
    names: ball class model, translation relation model, centre-distance measure, the
    ELEmbeddings objective (slack margin, strict margin for the negatives, signed radius for
    the bottom concept) and the unit-norm regulariser of the centres. The module *is* a
    :class:`~mowl.representation.ComposedGeometry`: ``module.parts`` names its parts, and
    the parts can be swapped (:class:`~mowl.models.ComposedEL`).

    The embedding tables are those of the parts and are also reachable by the names the
    earlier module had: ``class_embed``, ``class_rad``, ``rel_embed``, ``ind_embed``,
    ``ind_rad``.

    :param all_negative_losses: If ``True``, the module also implements the negative losses \
that DELE [mashkova2024dele]_ defines for the normal forms other than GCI2 (``gci0``, ``gci1``, \
``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``), and :attr:`neg_capable_gcis` lists them. \
Defaults to ``False``, which leaves the module exactly as before. See also \
:meth:`enable_all_negative_losses`.
    :type all_negative_losses: bool, optional
    :param epsilon: The :math:`\\varepsilon` of the DELE negative losses for bottom normal \
forms (minimal radius / offset norm of a satisfiable class). Defaults to ``0.01``.
    :type epsilon: float, optional
    :param gci1_radius_term: If ``True`` (default), the loss of \
:math:`C \\sqcap D \\sqsubseteq E` includes the term \
:math:`\\max(0, \\min(r_C, r_D) - r_E - \\gamma)` of the published loss. ``False`` drops it, \
as the authors' released code does (and as mOWL did before); use it to reproduce numbers \
obtained with that code.
    :type gci1_radius_term: bool, optional
    """

    #: Which of the losses of the DELE reference code combines the scores of this module
    #: (:class:`~mowl.training.DELECodeLoss`).
    dele_code_form = "ball"

    #: What the joint initialisation of the reference code normalises together
    #: (:class:`~mowl.training.JointNormalisedInitialisation`).
    joint_init = "ball"
    joint_init_roles = "rel_embed"

    #: ``"slack"``: a positive margin relaxes the inclusion losses (see the objective).
    margin_semantics = "slack"

    def __init__(self, nb_ont_classes, nb_rels, nb_inds, embed_dim=50, margin=0.1, reg_norm=1,
                 all_negative_losses=False, epsilon=0.01, gci1_radius_term=True):
        geometry = METHODS["elem"](nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
                                   margin=margin, reg_norm=reg_norm, epsilon=epsilon,
                                   gci1_radius_term=gci1_radius_term)
        self._assemble(geometry, nb_ont_classes, nb_rels, nb_inds, embed_dim, reg_norm,
                       all_negative_losses)

    # -- construction from the parts -------------------------------------------------------

    def _assemble(self, geometry, nb_ont_classes, nb_rels, nb_inds, embed_dim, reg_norm=1,
                  all_negative_losses=False):
        """Make this module the geometry ``geometry`` (the parts it was built from), with
        the attributes of an ELEmbeddings module."""
        super()._assemble(geometry, nb_ont_classes, nb_rels, nb_inds, embed_dim)
        self._reg_norm = reg_norm
        self._global_regularizers = self.regularizers
        self._per_axiom = False
        self.objective.unit_norm_target = float(reg_norm)
        self.class_rad = getattr(self.class_model, "radii", None)
        self.ind_rad = getattr(self.individual_model, "radii", None)
        if getattr(self.objective, "gci0_shift", None) is None:
            self.objective.gci0_shift = self._subclass_shift
        if all_negative_losses:
            self.enable_all_negative_losses()

    def _subclass_shift(self):
        """The learned "SubClassOf" translation of a training regime (``None``: none)."""
        return getattr(self, "subclass_relation", None)

    def _ball_measure(self):
        measure = self.measure
        return measure if getattr(measure, "host", None) is None else measure.host

    @property
    def gci1_radius_term(self):
        """Whether the loss of :math:`C \\sqcap D \\sqsubseteq E` has the fourth term of the
        published loss (the measure's ``radius_term``)."""
        return self._ball_measure().radius_term

    @gci1_radius_term.setter
    def gci1_radius_term(self, value):
        self._ball_measure().radius_term = bool(value)

    @property
    def negative_loss_form(self):
        """Form of the negative loss of :math:`C \\sqsubseteq \\exists R.D`: ``"hinge"``
        (default) :math:`\\max(0, r_C + r_D - \\lVert c + R - d \\rVert + \\gamma)`, or
        ``"linear"``, the same expression without the hinge, as in the ELEm and EmEL++
        baselines of the Box\\ :sup:`2`\\ EL code. Set by the training regime
        :class:`~mowl.training.UnboundedNegativeLoss`; it is the ``gci2_negative`` of the
        objective (``"apart"`` or ``"linear"``)."""
        return "linear" if self.objective.gci2_negative == "linear" else "hinge"

    @negative_loss_form.setter
    def negative_loss_form(self, value):
        if value not in ("hinge", "linear"):
            raise ValueError("negative_loss_form must be 'hinge' or 'linear'.")
        for objective in self._objectives():
            objective.gci2_negative = "linear" if value == "linear" else "apart"

    @property
    def reg_norm(self):
        """The norm the class centres are pulled towards."""
        return self._reg_norm

    @reg_norm.setter
    def reg_norm(self, value):
        self._reg_norm = value
        self.objective.unit_norm_target = float(value)
        for regularizer in self._global_regularizers:
            if hasattr(regularizer, "target"):
                regularizer.target = value

    @property
    def norm_penalty(self):
        """Where the unit-norm penalty :math:`|\\lVert c \\rVert - 1|` of the class centres
        is applied: ``"global"`` (default) once per step, averaged over all classes (the
        module's regulariser), or ``"per_axiom"``, added to the loss of every axiom for every
        class centre it contains (including negatives), with
        :meth:`regularization_loss` zero, as in the original ELEmbeddings code and the
        baselines of the Box\\ :sup:`2`\\ EL code. Set by the training regime
        :class:`~mowl.training.PerAxiomNormPenalty`."""
        return "per_axiom" if self._per_axiom else "global"

    @norm_penalty.setter
    def norm_penalty(self, value):
        if value not in ("global", "per_axiom"):
            raise ValueError("norm_penalty must be 'global' or 'per_axiom'.")
        self._per_axiom = value == "per_axiom"
        self.regularizers = () if self._per_axiom else self._global_regularizers
        # (the forms RegD reads with its energy have no penalty of ELEmbeddings)
        read_by_regd = getattr(self.objective, "regd_forms", ())
        self.objective.unit_norm = tuple(f for f in _PER_AXIOM_FORMS
                                         if f not in read_by_regd) if self._per_axiom else ()

    def gci2_score(self, data):
        """The positive loss of :math:`C \\sqsubseteq \\exists R.D` (the score of the pair)."""
        return self.gci2_loss(data)
