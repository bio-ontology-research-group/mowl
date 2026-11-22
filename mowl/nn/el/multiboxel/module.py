"""MultiBoxEL neural module."""
from __future__ import annotations

import torch as th

from mowl.representation.library.structural_points import POINT_READINGS as _VALID_OBJECTIVES
from mowl.nn.el.structural import Regions
from mowl.representation.library.multibox_methods import multiboxel
from mowl.representation.library.multibox_modules import (
    UnionModule, check_nonnegative, check_union_options, released_names)
from mowl.representation.objectives import FORMS
from mowl.representation.library.structural_methods import StructuralGeometry
from mowl.representation.library.structural_regions import to_slots
from mowl.training.regularization import NormPenalty

_MULTIBOXEL_OBJECTIVES = _VALID_OBJECTIVES + ("structural",)


@released_names(
    class_center="class_model.centres", class_offset="class_model.offsets",
    bump_classes="class_model.bumps",
    head_center="relation_model.head_centres", head_offset="relation_model.head_offsets",
    tail_center="relation_model.tail_centres", tail_offset="relation_model.tail_offsets",
    ind_center="individual_model.centres", ind_offset="individual_model.offsets",
    bump_individuals="individual_model.bumps", elem_embed="point_terms.elem_embed")
class MultiBoxELModule(UnionModule, StructuralGeometry):
    """Multi-component Box\\ :math:`^2`\\ EL neural module.

    The module is the combination of parts that MultiBoxEL is
    (:func:`~mowl.representation.library.multibox_methods.multiboxel`): every class is a union
    of :math:`k = \\text{n\\_components}` axis-parallel boxes (one bump vector per class,
    shared by its boxes), roles are a head and a tail box (as in
    :class:`~mowl.nn.BoxSquaredELModule`), the measure is the box measure lifted to unions of
    boxes, and the objective is the axiom reading of Box\\ :math:`^2`\\ EL, or, as the
    argument ``objective`` says, the structural reading (slots of boxes and
    definitions), or one of the point readings. It is a
    :class:`~mowl.representation.ComposedGeometry`
    (a :class:`~mowl.representation.library.structural_methods.StructuralGeometry`): the
    tables of the parts keep the names they had (``class_center`` of shape
    ``(nb_ont_classes, k * d)``, ``class_offset``, ``bump_classes``, ``head_center``, ...), and
    models saved under those names load.

    ``module.objective`` is the :class:`~mowl.representation.Objective` of the parts;
    the name of the reading is ``module.reading``, and it is fixed when the module is made.
    The weights of the regularisers are views of the regularisers of the parts
    (``reg_factor``, ``component_reg``, ``diversity_reg``).

    Guarantees:

    - **Exact k = 1 reduction:** When ``n_components == 1`` and ``component_reg == 0.0``
      with weights copied from a :class:`~mowl.nn.BoxSquaredELModule`,
      every GCI loss (positive and negative) and :meth:`regularization_loss`
      agree with :class:`~mowl.nn.BoxSquaredELModule` to
      within floating-point tolerance ``1e-6``.
    - **Effective component count:** :meth:`effective_components` counts, for
      each class on CPU via corner arithmetic, the components whose half-widths
      exceed ``eps`` in every coordinate and whose box is not contained in
      another component of the same class (with exact ties broken by keeping
      the lowest-index component).
    - **Checkpointed temperature buffer:** ``tau`` is stored as a registered
      ``float32`` buffer so it is saved and restored in ``state_dict()`` while
      still accepting ``module.tau = float_value`` assignments.

    Non-guarantees:

    - Under ``objective="structural"`` the regions of a DEFINED class
      :math:`A \\equiv X \\sqcap Y` are computed from its definers, but its translation
      bump (``bump_classes[A]``) is not: no definition determines it, so it stays a free
      parameter trained only through the role axioms that mention :math:`A`. This is
      intended: the bump is a role-interaction parameter of Box\\ :math:`^2`\\ EL, not part
      of the region, and an intersection of two bumps has no canonical value. It is not
      constrained by the existence or size terms.
    - Role axiom losses (``role_inclusion``, ``role_chain``) are not implemented
      (``role_axiom_capable = False``).
    - Negative sampling is only supported for ``gci2`` and
      ``object_property_assertion`` (``neg_capable_gcis``).
    - Under ``component_aggregation="soft"`` at finite ``tau > 0``, the soft
      loss is not zero at a hard model because :func:`~mowl.representation.library.multibox_parts.soft_minimum`
      (:math:`\\le \\text{mean}`) pulls all target components toward each source
      component in proportion to their softmax weights; ``diversity_reg``
      (:class:`~mowl.representation.library.multibox_objective.DiversityPenalty`) exists to counteract this
      collapse pressure.

    Parameters
    ----------
    nb_ont_classes : int
        Number of ontology classes.
    nb_rels : int
        Number of object properties (roles).
    nb_inds : Optional[int], default None
        Number of individuals (ABox).
    embed_dim : int, default 50
        Ambient Euclidean dimension :math:`d`.
    n_components : int, default 2
        Number of box components :math:`k \\ge 1` per class.
    gamma : float, default 0.0
        Inclusion margin :math:`\\gamma`.
    delta : float, default 2.0
        Negative-sampling distance target :math:`\\delta` for ``gci2`` and
        ``object_property_assertion``.
    reg_factor : float, default 0.05
        Weight for the Box²EL bump :math:`L_2` regulariser.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser
        :math:`\\text{mean}_C \\text{mean}_{i<j} \\text{relu}(\\tau - \\|c_{C,i} - c_{C,j}\\|_2)`
        with :math:`\\tau = 0.1`.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-assignment temperature stored as a registered buffer on the module.
    diversity_reg : float, default 0.0
        Weight for the pairwise component volume overlap regulariser
        :class:`~mowl.representation.library.multibox_objective.DiversityPenalty`.
    objective : str, default "axiom"
        Training objective mode: ``"axiom"``, ``"point"``, ``"mixed"``,
        ``"hybrid"``, or ``"structural"``. Under ``objective="point"`` and
        ``"mixed"``, ABox ``class_assertion`` is still trained in the main loop
        via :math:`-\\log m_C(a)` while ``object_property_assertion`` is zeroed
        (handled via :meth:`point_role_loss`).
    beta : float, default 0.1
        Sigmoid temperature :math:`\\beta > 0` for soft box membership under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``.
    neg_weight : float, default 0.1
        Weight multiplying random-negative BCE terms under ``objective="point"``,
        ``"mixed"``, or ``"hybrid"``.
    volume_reg : float, default 0.0
        Volume regularisation weight stored on the module for ``objective="point"``,
        ``"mixed"``, or ``"hybrid"``.
    ind_halfwidth : float, optional
        Virtual box half-width around individual targets in role pairs under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``; defaults to
        ``5.0 * beta``.
    margin : float, default 0.02
        Tightening margin :math:`m` used by ``objective="structural"`` for GCI
        inclusions, ``gci1_bot`` separation, and :func:`~mowl.baselines.nn.el.multiboxel.structural_losses.existence_loss`.
        For Box\\ :math:`^2`\\ EL role and assertion scores (where ``gamma`` loosens
        inclusions via :math:`\\text{relu}(\\text{dist} + 2 o - \\gamma)`), positive
        scores pass ``gamma = -margin`` so :math:`m > 0` tightens inclusion, while
        negative scores (:func:`~mowl.baselines.nn.el.multiboxel.losses.gci2_score_neg` and
        negative ``object_property_assertion``) keep Box\\ :math:`^2`\\ EL's
        convention (``gamma = +margin``, which relaxes them by the margin), so
        the negative-sampling term is unchanged between the axiom and structural
        objectives.
    max_components : int, optional, default None
        Maximum component slots :math:`K \\ge k` per class under
        ``objective="structural"``; defaults to ``n_components ** 2`` when ``None``.
    structural_definitions : bool, default True
        Whether ``objective="structural"`` compiles acyclic conjunction
        definitions :math:`A \\equiv X \\sqcap Y` via
        :class:`~mowl.nn.el.structural.DefinitionTable`.
    existence_weight : float, default 1.0
        Non-negative weight on :func:`~mowl.baselines.nn.el.multiboxel.structural_losses.existence_loss` in
        :meth:`structural_extra_loss`.
    size_reg : float, default 1e-3
        Non-negative weight on the primitive side-length regulariser
        ``mean(2 * |offset|)`` in :meth:`structural_extra_loss`.
    size_reg_without_tables : bool, default False
        The size term needs to know which class is ``owl:Thing`` (it must not shrink it) and
        which classes are defined. It is therefore applied only once
        :meth:`set_structural_tables` has installed the tables. ``True`` applies it to every
        class also before that (the earlier behaviour, which shrinks ``owl:Thing``).
    reseed_every : int, default 0
        Epoch interval for heuristic discrete re-seeding (:meth:`reseed`) under
        ``objective="structural"`` (``0`` disables re-seeding).
    bot_agg : {"mean", "sum"}, default "mean"
        Aggregation over active component pairs in :meth:`gci1_bot_loss` under
        ``objective="structural"`` (passed as ``agg`` to
        :func:`~mowl.baselines.nn.el.multiboxel.structural_losses.gci1_bot_structural`).
    witness_weight : float, default 0.0
        Non-negative weight on :func:`~mowl.baselines.nn.el.multiboxel.structural_losses.witness_loss` in
        :meth:`structural_extra_loss` under ``objective="structural"``. When
        ``witness_weight > 0.0`` and ``targets.witness`` is attached via
        :meth:`set_structural_tables`, allocates ``self.witness_embed`` with
        one learnable point per satisfiable target class.
    reseed_mode : {"dead", "idle"}, default "dead"
        Discrete re-seeding mode for :meth:`reseed` under
        ``objective="structural"``: ``"dead"`` relocates nested or thin
        primitive components to violated conjunction pair partners; ``"idle"``
        requires ``witness_weight > 0`` and relocates the least-used primitive
        leaf components of dead defined classes to their witness points.
    reseed_size : float, default 0.15
        Positive box half-width :math:`> 0` assigned in every coordinate to
        re-seeded components under ``reseed_mode="idle"``.
    reseed_batch : int, default 1
        Maximum number of dead defined classes :math:`\\ge 1` processed per
        :meth:`reseed` call under ``reseed_mode="idle"``.

    References
    ----------
    G. Federico, M. Leemhuis and O. Kutz, "Introducing Multiboxes: Increasing
    the Trustworthiness of Ontology Embeddings", JOWO 2026.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = False

    #: The order in which the tables were initialised (a seeded module has its numbers).
    released_order = (
        "class_model.centres", "class_model.offsets", "relation_model.head_centres",
        "relation_model.head_offsets", "relation_model.tail_centres",
        "relation_model.tail_offsets", "class_model.bumps", "individual_model.bumps",
        "individual_model.centres", "individual_model.offsets")
    #: ``per_axiom_column = False``: the flat ``(B,)`` of the formulas.
    flat_forms = FORMS

    def __init__(
        self,
        nb_ont_classes,
        nb_rels,
        nb_inds=None,
        embed_dim=50,
        n_components=2,
        gamma=0.0,
        delta=2.0,
        reg_factor=0.05,
        component_reg=0.0,
        component_aggregation="hard",
        tau=1.0,
        diversity_reg=0.0,
        objective="axiom",
        beta=0.1,
        neg_weight=0.1,
        volume_reg=0.0,
        ind_halfwidth=None,
        margin=0.02,
        max_components=None,
        structural_definitions=True,
        existence_weight=1.0,
        size_reg=1e-3,
        size_reg_without_tables=False,
        reseed_every=0,
        bot_agg="mean",
        witness_weight=0.0,
        reseed_mode="dead",
        reseed_size=0.15,
        reseed_batch=1,
    ):
        tau, beta_f, halfwidth = check_union_options(
            n_components, embed_dim, component_aggregation, tau, diversity_reg, beta,
            ind_halfwidth, neg_weight, volume_reg)
        if objective not in _MULTIBOXEL_OBJECTIVES:
            raise ValueError(f"objective must be one of {_MULTIBOXEL_OBJECTIVES}, got "
                             f"{objective!r}")
        if objective == "structural" and component_aggregation != "hard":
            raise ValueError("component_aggregation must be 'hard' when "
                             f"objective='structural', got {component_aggregation!r}")
        if max_components is None:
            max_components = n_components * n_components
        elif (isinstance(max_components, bool) or not isinstance(max_components, int)
              or max_components < n_components):
            raise ValueError(f"max_components must be an int >= n_components "
                             f"({n_components}), got {max_components!r}")
        if not isinstance(structural_definitions, bool):
            raise ValueError(
                f"structural_definitions must be a bool, got {structural_definitions!r}")
        existence_weight = check_nonnegative("existence_weight", existence_weight)
        size_reg = check_nonnegative("size_reg", size_reg)
        if isinstance(reseed_every, bool) or not isinstance(reseed_every, int) \
                or reseed_every < 0:
            raise ValueError(f"reseed_every must be an int >= 0, got {reseed_every!r}")
        _check_reseed(bot_agg, witness_weight, reseed_mode, reseed_size, reseed_batch)
        structural = objective == "structural"
        rng_state = th.get_rng_state()
        multiboxel(
            nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim,
            margin=margin if structural else gamma, n_components=n_components, delta=delta,
            reg_factor=reg_factor, component_reg=component_reg,
            component_aggregation=component_aggregation, tau=tau,
            diversity_reg=float(diversity_reg), reading=objective, beta=beta_f,
            neg_weight=float(neg_weight), ind_halfwidth=halfwidth,
            volume_reg=float(volume_reg), max_components=max_components,
            existence_weight=existence_weight, size_reg=size_reg, bot_agg=bot_agg,
            witness_weight=float(witness_weight),
            size_reg_without_tables=size_reg_without_tables,
            structural_definitions=structural_definitions, into=self)
        self.initialise_in_released_order(rng_state)
        self._install_tau(tau)
        self.strict_negatives = True
        self.reading = objective
        self.__dict__["_custom_ind_halfwidth"] = ind_halfwidth is not None
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.__dict__["_ind_embed_readonly"] = True
        self.max_components = max_components
        self.structural_definitions = structural_definitions
        self.reseed_every = reseed_every
        if structural:
            self.reseed_options.update(mode=reseed_mode, size=float(reseed_size),
                                       batch=int(reseed_batch))
        else:
            # the other readings do not read the structure of the ontology: the module has
            # the attributes of the structural reading (settings, tables) without the parts
            self.__dict__["_bot_agg"] = bot_agg
            self.existence_weight, self.size_reg = existence_weight, size_reg
            self.witness_weight = float(witness_weight)
            self.size_reg_without_tables = bool(size_reg_without_tables)
            self.tables_installed = False
            self.definition_table = self.existence_targets = self.top_idx = None
            self.witness_embed = None
            self.last_reseed_count = 0
            self.reseed_options = {"mode": reseed_mode, "size": float(reseed_size),
                                   "batch": int(reseed_batch)}

    @property
    def margin_attribute(self):
        """The inclusion margin of the ``"structural"`` reading is ``margin`` (it tightens);
        in every other reading it is ``gamma`` of Box\\ :sup:`2`\\ EL (it loosens)."""
        return "margin" if self.reading == "structural" else "gamma"

    def __setattr__(self, name, value):
        if name == "ind_embed" and self.__dict__.get("_ind_embed_readonly", False):
            raise AttributeError("ind_embed is a read-only property aliasing ind_center.")
        if name == "witness_weight":
            value = check_nonnegative("witness_weight", value)
            if value <= 0.0 and self.__dict__.get("reseed_options", {}).get("mode") == "idle":
                raise ValueError("reseed_mode='idle' requires witness_weight > 0, "
                                 f"got witness_weight={value!r}")
        super().__setattr__(name, value)

    # -- the numbers of the structural reading, which are those of the parts ------------
    @property
    def bot_agg(self):
        """Aggregation over pairs of components in ``gci1_bot`` (structural reading)."""
        return getattr(self.objective, "bot_agg", self.__dict__.get("_bot_agg", "mean"))

    @bot_agg.setter
    def bot_agg(self, value):
        if not isinstance(value, str) or value not in ("mean", "sum"):
            raise ValueError(f"bot_agg must be 'mean' or 'sum', got {value!r}")
        if hasattr(self.objective, "bot_agg"):
            self.objective.bot_agg = value
        else:
            self.__dict__["_bot_agg"] = value

    @property
    def reseed_mode(self):
        """``"dead"`` or ``"idle"`` (see :meth:`reseed`)."""
        return self.reseed_options["mode"]

    @reseed_mode.setter
    def reseed_mode(self, value):
        if not isinstance(value, str) or value not in ("dead", "idle"):
            raise ValueError(f"reseed_mode must be 'dead' or 'idle', got {value!r}")
        if value == "idle" and float(self.witness_weight) <= 0.0:
            raise ValueError("reseed_mode='idle' requires witness_weight > 0, "
                             f"got witness_weight={self.witness_weight!r}")
        self.reseed_options["mode"] = value

    @property
    def reseed_size(self):
        """Half-width of a component placed at a witness point (``"idle"``)."""
        return self.reseed_options["size"]

    @reseed_size.setter
    def reseed_size(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
            raise ValueError(f"reseed_size must be > 0, got {value!r}")
        self.reseed_options["size"] = float(value)

    @property
    def reseed_batch(self):
        """Dead defined classes handled per call of :meth:`reseed` (``"idle"``)."""
        return self.reseed_options["batch"]

    @reseed_batch.setter
    def reseed_batch(self, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"reseed_batch must be an int >= 1, got {value!r}")
        self.reseed_options["batch"] = int(value)

    @property
    def reg_factor(self):
        """The weight of the norm penalty on the bump vectors."""
        return self._weight_of(NormPenalty)

    @reg_factor.setter
    def reg_factor(self, value):
        self._set_weight_of(NormPenalty, value)

    @property
    def delta(self):
        """The :math:`\\delta` of the negative loss."""
        return self.objective.delta

    @delta.setter
    def delta(self, value):
        self.objective.delta = float(value)

    @property
    def _tables_installed(self):
        return self.tables_installed

    @property
    def wants_structural_tables(self):
        """Only the structural reading reads the definitions of the ontology."""
        return self.reading == "structural" and super().wants_structural_tables

    def set_structural_tables(self, table, targets, top_idx=None):
        """Attach the :class:`~mowl.nn.el.structural.DefinitionTable` and
        :class:`~mowl.nn.el.structural.ExistenceTargets`: with the structural reading, defined
        classes become the intersections of their definers and the penalties read the targets
        (:meth:`StructuralGeometry.set_structural_tables`); the other readings keep them
        only."""
        if self.reading == "structural":
            return super().set_structural_tables(table, targets, top_idx)
        self.definition_table, self.existence_targets, self.top_idx = table, targets, top_idx
        self.tables_installed = True

    # -- regions ------------------------------------------------------------------------
    def class_regions(self, idx):
        """:class:`~mowl.nn.el.structural.Regions` with ``K = max_components`` slots of the
        classes ``idx``: the components of a primitive class (``w = |offset| >= 0``, active) and
        inactive padding (``cen = 0``, ``w = -1``); a defined class is the intersection of its
        definers (:func:`~mowl.baselines.nn.el.multiboxel.structural_losses.pair_slots`)."""
        regions = to_slots(self.classes(idx.to(device=self.device, dtype=th.long))).slots
        pad = self.max_components - regions.cen.shape[1]
        if pad > 0:
            b, _, d = regions.cen.shape
            regions = Regions(
                cen=th.cat([regions.cen, regions.cen.new_zeros((b, pad, d))], dim=1),
                w=th.cat([regions.w, -th.ones_like(regions.w[:, :1]).expand(b, pad, d)],
                         dim=1),
                active=th.cat([regions.active, regions.active.new_zeros((b, pad))], dim=1))
        return regions

    def all_class_regions(self):
        """:class:`~mowl.nn.el.structural.Regions` of all classes on the CPU, without
        gradients."""
        with th.no_grad():
            r = self.class_regions(th.arange(self.nb_ont_classes, device=self.device))
            return Regions(cen=r.cen.detach().cpu(), w=r.w.detach().cpu(),
                           active=r.active.detach().cpu())

    # -- the penalties of the structural reading ------------------------------------------
    def structural_extra_loss(self):
        """The existence, size and witness penalties of the structural reading (zero for
        the other readings, and before the tables are installed for the size term unless
        ``size_reg_without_tables``): the penalties of the parts, which the geometry itself
        counts in :meth:`~mowl.representation.ComposedGeometry.regularization_loss`; this
        module's :meth:`regularization_loss` is the sum of the other regularisers (the bump
        norm and the components), as it was, and the model adds the two."""
        total = th.zeros((), device=self.device)
        for penalty in self._structural_penalties():
            term = penalty.penalty(self)
            if term is not None:
                total = total + term
        return total

    def extra_penalties(self):
        """The penalties of the structural reading (existence, size, witness): the
        regularisers of the geometry that :meth:`regularization_loss` does not count; the
        regime :class:`~mowl.training.structural.ExtraPenalties` adds them to training."""
        return self._structural_penalties()

    def _structural_penalties(self):
        base = self.__dict__.get("base_regularizers", self.regularizers)
        return tuple(r for r in self.regularizers if r not in base)

    def regularization_loss(self):
        total = th.zeros((), device=self.device)
        for regularizer in self._regularizer_list():
            term = regularizer.penalty(self)
            if term is not None:
                total = total + term
        return total

    def reseed(self, rng):
        """Relocate dead or idle primitive components under ``no_grad``
        (:func:`~mowl.representation.library.structural_reseed.reseed_components`); returns
        the number of moves."""
        return StructuralGeometry.reseed(self, rng)


def _check_reseed(bot_agg, witness_weight, reseed_mode, reseed_size, reseed_batch):
    if not isinstance(bot_agg, str) or bot_agg not in ("mean", "sum"):
        raise ValueError(f"bot_agg must be 'mean' or 'sum', got {bot_agg!r}")
    check_nonnegative("witness_weight", witness_weight)
    if not isinstance(reseed_mode, str) or reseed_mode not in ("dead", "idle"):
        raise ValueError(f"reseed_mode must be 'dead' or 'idle', got {reseed_mode!r}")
    if reseed_mode == "idle" and float(witness_weight) <= 0.0:
        raise ValueError("reseed_mode='idle' requires witness_weight > 0, "
                         f"got witness_weight={witness_weight!r}")
    if isinstance(reseed_size, bool) or not isinstance(reseed_size, (int, float)) \
            or float(reseed_size) <= 0.0:
        raise ValueError(f"reseed_size must be > 0, got {reseed_size!r}")
    if isinstance(reseed_batch, bool) or not isinstance(reseed_batch, int) or reseed_batch < 1:
        raise ValueError(f"reseed_batch must be an int >= 1, got {reseed_batch!r}")
