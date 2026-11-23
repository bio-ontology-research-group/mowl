"""GroundedELModule: the geometry of GroundedEL, put together from its parts.

The module is the combination of the parts of
:mod:`mowl.representation.library.grounded_parts`, each an object with its own tables:

- ``elements`` (:class:`~mowl.representation.library.grounded_parts.ElementDomain`), the
  **shared part**: the finite domain of :math:`|\\Delta|` elements (indices
  ``0 .. nb_classes - 1`` are the witnesses :math:`x_A` of the classes, the rest the
  individuals), their positions :math:`p_x \\in \\mathbb{R}^d` (optionally from an encoder of
  features), their deadness :math:`\\delta_x \\in [0, 1]` and the membership temperature
  :math:`\\tau`;
- ``boxes`` (:class:`~mowl.representation.library.grounded_parts.BoxUnionClasses`), the
  class model: :math:`K` boxes per class, some inactive;
- ``roles`` (:class:`~mowl.representation.library.grounded_parts.ProductBoxRelations` for
  ``role_mode="product"``, :class:`~mowl.representation.library.grounded_parts.TranslationBoxRelations`
  for ``"translation"``), the relation model;
- ``depth_measure`` (:class:`~mowl.representation.library.grounded_parts.UnionDepth`), the
  measure: depth in a union of boxes (``agg``, ``agg_tau``), its soft and crisp readings;
- the individual model (:class:`~mowl.representation.library.grounded_parts.ElementPoints`),
  a view of the shared part.

The methods of the module (``positions``, ``dead``, ``class_regions``, ``depth``,
``membership``, ``role_regions``, ``role_depth``, ``role_membership``, ``exists_box``,
``exists_depth``) are those of the parts; the names of the tables of the earlier single
module (``class_centers``, ``role_head_centers``, ``elem_pos``, ``dead_logits``, ...) still
read and write the tables of the parts, and a ``state_dict`` with those names loads.

Guarantees:

- **Reasoner-free (R1):** Contains no imports or calls to reasoners or semantic
  completion routines.
- **Exact limit realisation (R2):** After :meth:`exact_limit_init` with labels
  and edges of a consistent least model, every crisp grounded element hinge and
  every crisp region hinge inside active boxes evaluates to exact ``0.0``.
- **Bitwise monotone capacity expansion (R3):** Calling :meth:`expand_components`
  with ``new_K >= n_components`` leaves :meth:`depth`, :meth:`membership`,
  :meth:`role_depth`, and :meth:`role_membership` bitwise identical.

Non-guarantees:

- Under ``agg="max"``, gradients with respect to box parameters flow only through
  the winning active component and bottleneck coordinate.
- Under ``agg="lse"`` at finite ``agg_tau > 0``, smooth depth exceeds crisp
  maximum depth by up to :math:`\\tau_{\\text{agg}} \\log K` when multiple active
  components have equal depth.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn

from mowl.nn.el.structural import Regions, inside_measure
from mowl.representation.geometry import Geometry
from mowl.representation.library.grounded_parts import (  # noqa: F401 (re-exported)
    EMPTY_DEPTH_FLOOR as _EMPTY_DEPTH_FLOOR,
    INACTIVE_RAW_WIDTH as _INACTIVE_RAW_WIDTH,
    TOP_DEPTH_CEIL as _TOP_DEPTH_CEIL,
    BoxUnionClasses,
    ElementDomain,
    ElementPoints,
    ExistsRegions as _ExistsRegions,
    ProductBoxRelations,
    TranslationBoxRelations,
    UnionDepth,
    align_points_and_ids,
    index_tensor,
    inv_softplus as _inv_softplus,
    validate_finite_float as _validate_finite_float,
    validate_int_ge as _validate_int_ge,
    validate_inverse_of as _validate_inverse_of,
    validate_positive_float as _validate_positive_float,
)

__all__ = ["GroundedELModule"]

_LIVE_LOGIT: float = 0.0
_DEAD_LOGIT: float = 1.0

#: The relation model of each ``role_mode``.
RELATION_MODELS = {"product": ProductBoxRelations, "translation": TranslationBoxRelations}

#: Names of the tables of the earlier single module -> (part, name of the table in the part).
_TABLES = {
    "elem_pos": ("elements", "elem_pos"),
    "dead_logits": ("elements", "dead_logits"),
    "element_features": ("elements", "element_features"),
    "encoder": ("elements", "encoder"),
    "raw_tau_s": ("elements", "raw_tau_s"),
    "tau": ("elements", "tau"),
    "class_centers": ("boxes", "centers"),
    "class_raw_widths": ("boxes", "raw_widths"),
    "class_active": ("boxes", "active"),
    "n_components": ("boxes", "n_components"),
    "top_id": ("boxes", "top_id"),
    "bot_id": ("boxes", "bot_id"),
    "role_head_centers": ("roles", "head_centers"),
    "role_head_raw_widths": ("roles", "head_raw_widths"),
    "role_tail_centers": ("roles", "tail_centers"),
    "role_tail_raw_widths": ("roles", "tail_raw_widths"),
    "role_centers": ("roles", "centers"),
    "role_raw_widths": ("roles", "raw_widths"),
    "role_active": ("roles", "active"),
    "n_role_components": ("roles", "n_components"),
    "inverse_of": ("roles", "inverse_of"),
    "agg": ("depth_measure", "agg"),
    "agg_tau": ("depth_measure", "agg_tau"),
}
#: ``state_dict`` keys of the earlier single module -> keys of the parts.
_LEGACY_KEYS = {old: f"{part}.{name}" for old, (part, name) in _TABLES.items()
                if part != "depth_measure" and old not in ("n_components", "n_role_components",
                                                            "top_id", "bot_id")}


def _resolve(kind, value):
    """A part given by name (:data:`GROUNDED_PARTS`) or as it is."""
    if isinstance(value, str):
        from mowl.representation.library.grounded_parts import GROUNDED_PARTS
        names = GROUNDED_PARTS[kind]
        if value not in names:
            raise ValueError(f"Unknown {kind} part {value!r}; available: {sorted(names)}.")
        return names[value]
    return value


def _part(kind, given, default, *args, **kwargs):
    """The part of ``kind``: the default, or ``given`` (a name, a factory called like the
    default, or a part)."""
    given = _resolve(kind, given)
    if given is None:
        return default(*args, **kwargs)
    if isinstance(given, (nn.Module, UnionDepth)) or not callable(given):
        return given
    return given(*args, **kwargs)


class GroundedELModule(nn.Module, Geometry):
    """Element-grounded multi-box and product-role geometry for :math:`\\mathcal{EL}^{++}`,
    the combination of an element domain, a class model, a relation model and a measure
    (module docstring).

    Parameters
    ----------
    nb_elements : int
        Total number of domain elements :math:`|\\Delta| \\ge \\text{nb\\_classes}`.
    nb_classes : int
        Number of ontology classes :math:`\\ge 1`.
    nb_rels : int, default 0
        Number of object properties (roles) :math:`\\ge 0`.
    embed_dim : int, default 50
        Ambient Euclidean dimension :math:`d \\ge 1`.
    n_components : int, default 2
        Number of box components :math:`K \\ge 1` per class.
    n_role_components : int, default 2
        Number of components :math:`K_r \\ge 1` per role.
    tau : float, default 1.0
        Sigmoid temperature :math:`\\tau > 0` converting signed depth to soft membership.
    agg : {"max", "lse"}, default "max"
        Aggregation over the active components (the measure).
    agg_tau : float, default 0.1
        Temperature :math:`\\tau_{\\text{agg}} > 0` for ``agg="lse"``.
    dead_init_logit : float, default 0.0
        Initial deadness :math:`\\delta_x` (clamped to ``[0, 1]``).
    element_features : Optional[torch.Tensor], default None
        Feature matrix ``(nb_elements, F)`` or ``(nb_elements - nb_classes, F)``.
    encoder : Optional[torch.nn.Module], default None
        Encoder mapping features to ``embed_dim``.
    top_id, bot_id : Optional[int], default None
        Class indices of :math:`\\top` and :math:`\\bot`.
    role_mode : {"product", "translation"}, default "product"
        The relation model (:data:`RELATION_MODELS`).
    inverse_of : Optional[Sequence[int]], default None
        Symmetric partner mapping of length ``nb_rels`` (``-1``: no inverse); a role with
        ``0 <= inverse_of[r] < r`` owns no parameters and is derived from its partner.
    class_model, relation_model : optional
        Replace the class model or the relation model: a name of
        :data:`~mowl.representation.library.grounded_parts.GROUNDED_PARTS`, a factory
        ``(size, n_components, embed_dim, measure, domain, ...)`` (such as the class of a
        part), or a part (it then reads the measure and the element domain of this module).
        The relation model replaces ``role_mode`` (its ``mode``).
    measure : optional
        Replace the measure (a name, a factory ``(agg, agg_tau)`` or a measure).

    Assigning ``module.class_model``, ``module.relation_model`` or ``module.measure``
    replaces the part in the same way.
    """

    #: The model that trains this geometry (``ComposedEL`` refuses it and says so); its
    #: parts are swapped there: ``GroundedEL(..., relation_model="translation")``.
    trained_by = ("mowl.models.GroundedEL (class_model=, relation_model=, measure=, "
                  "objective=, element_sampler=, region_sampler=, crisp_regime=, exact_regime=)"
                  " or mowl.models.ComposedEL(method='grounded', classes=, relations=, "
                  "measure=, objective=)")

    #: No form has a negative loss of rows (the corrupted role assertions are a term of the
    #: joint objective).
    neg_capable_gcis = frozenset()
    #: The joint objective reads the role inclusions and chains.
    role_axiom_capable = True
    #: The joint objective (:mod:`mowl.representation.joint_objective`) a general model
    #: trains the geometry with (set by :func:`~mowl.representation.library.solved_grounded.
    #: grounded`; ``None``: GroundedEL's own step).
    joint_objective = None

    def __init__(
        self,
        nb_elements: int,
        nb_classes: int,
        nb_rels: int = 0,
        embed_dim: int = 50,
        n_components: int = 2,
        n_role_components: int = 2,
        tau: float = 1.0,
        agg: str = "max",
        agg_tau: float = 0.1,
        dead_init_logit: float = 0.0,
        element_features: Optional[th.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        top_id: Optional[int] = None,
        bot_id: Optional[int] = None,
        role_mode: str = "product",
        inverse_of: Optional[Union[Sequence[int], th.Tensor]] = None,
        class_model=None,
        relation_model=None,
        measure=None,
    ) -> None:
        super().__init__()
        self.nb_classes = _validate_int_ge("nb_classes", nb_classes, 1)
        self.nb_elements = _validate_int_ge("nb_elements", nb_elements, self.nb_classes)
        self.nb_rels = _validate_int_ge("nb_rels", nb_rels, 0)
        self.embed_dim = _validate_int_ge("embed_dim", embed_dim, 1)
        n_components = _validate_int_ge("n_components", n_components, 1)
        n_role_components = _validate_int_ge("n_role_components", n_role_components, 1)
        _validate_positive_float("tau", tau)
        self.depth_measure = _part("measure", measure, UnionDepth, agg=agg, agg_tau=agg_tau)
        self.dead_init_logit = _validate_finite_float("dead_init_logit", dead_init_logit)
        if not isinstance(role_mode, str) or role_mode not in RELATION_MODELS:
            raise ValueError(
                f"role_mode must be 'product' or 'translation', got {role_mode!r}")
        self.role_mode: str = role_mode
        if relation_model is not None and not isinstance(relation_model, nn.Module):
            # a name or a factory: the mode of the part it makes
            relation_model = _resolve("relations", relation_model)
        inv_buf = _validate_inverse_of(self.nb_rels, inverse_of)
        for name, value in (("top_id", top_id), ("bot_id", bot_id)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or not (0 <= value < self.nb_classes)):
                raise ValueError(
                    f"{name} must be an int in [0, {self.nb_classes}), got {value!r}")
        if top_id is not None and bot_id is not None and top_id == bot_id:
            raise ValueError(f"top_id and bot_id must be distinct, got {top_id}")

        # The parts, in the order the earlier module drew its random numbers.
        self.elements = ElementDomain(self.nb_elements, self.nb_classes, self.embed_dim,
                                      tau=tau, dead_init=self.dead_init_logit,
                                      element_features=element_features, encoder=encoder)
        self.boxes = self._adopt(_part(
            "classes", class_model, BoxUnionClasses, self.nb_classes, n_components,
            self.embed_dim, self.depth_measure, self.elements, top_id=top_id, bot_id=bot_id))
        self.roles = self._adopt(_part(
            "relations", relation_model, RELATION_MODELS[role_mode], self.nb_rels,
            n_role_components, self.embed_dim, self.depth_measure, self.elements,
            inverse_of=inv_buf))
        self.role_mode = getattr(self.roles, "mode", role_mode)
        self.last_reseed_count: int = 0
        self._register_load_state_dict_pre_hook(self._legacy_keys)

    # -- the tables of the parts under the names of the earlier single module ---------------
    def __getattr__(self, name):
        alias = _TABLES.get(name)
        if alias is not None:
            owner = self.__dict__.get("_modules", {}).get(alias[0])
            if owner is None:
                owner = self.__dict__.get(alias[0])
            if owner is not None:
                return getattr(owner, alias[1], None)
        return super().__getattr__(name)

    def _adopt(self, part):
        """Make ``part`` read the measure and the element domain of this module."""
        for name, shared in (("measure", self.__dict__.get("depth_measure")),
                             ("domain", self._modules.get("elements"))):
            if shared is not None and name in getattr(part, "__dict__", {}):
                part.__dict__[name] = shared
        return part

    def __setattr__(self, name, value):
        # replacing a part (nn.Module would otherwise register a new sub-module under the
        # name and leave the part that is read in place)
        if name == "class_model":
            super().__setattr__("boxes", self._adopt(value))
            return
        if name == "relation_model":
            super().__setattr__("roles", self._adopt(value))
            self.role_mode = getattr(value, "mode", self.role_mode)
            return
        if name == "measure":
            super().__setattr__("depth_measure", value)
            for part in (self._modules.get("boxes"), self._modules.get("roles")):
                if part is not None:
                    self._adopt(part)
            return
        alias = _TABLES.get(name)
        if alias is not None:
            owner = self.__dict__.get("_modules", {}).get(alias[0])
            if owner is None:
                owner = self.__dict__.get(alias[0])
            if owner is not None:
                setattr(owner, alias[1], value)
                return
        super().__setattr__(name, value)

    @staticmethod
    def _legacy_keys(state_dict, prefix, *args, **kwargs):
        """Load a ``state_dict`` of the earlier single module (flat table names)."""
        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            head, dot, rest = key[len(prefix):].partition(".")
            new = _LEGACY_KEYS.get(head)
            if new is not None and prefix + new + dot + rest not in state_dict:
                state_dict[prefix + new + dot + rest] = state_dict.pop(key)

    # -- the shared part ----------------------------------------------------------------------
    def project(self) -> None:
        """After an optimiser step: the deadness back in :math:`[0, 1]`, bottom dead (what
        GroundedEL's ``project_parameters`` does; a general model calls this)."""
        self.elements.clamp_dead(self.bot_id if self.bot_id is not None and self.bot_id >= 0
                                 else None, _DEAD_LOGIT)

    def regularization_loss(self):
        """No penalty besides the joint objective's minimality term."""
        return 0

    def ensure_tau_s(self) -> nn.Parameter:
        """The score temperature ``raw_tau_s`` of the shared part (created on first use)."""
        return self.elements.ensure_tau_s()

    @property
    def tau_s(self) -> th.Tensor:
        """Positive inductive score temperature :math:`\\max(\\operatorname{softplus}(\\text{raw}), 10^{-3})`."""
        return self.elements.tau_s

    def set_tau(self, t: float) -> None:
        """Set the membership temperature of the shared part."""
        self.elements.set_tau(t)

    def _to_index_tensor(self, ids, max_bound, name):
        return index_tensor(ids, max_bound, name, self.elements.device)

    def positions(self, element_ids=None) -> th.Tensor:
        """Element positions :math:`p_x` (the shared part)."""
        return self.elements.positions(element_ids)

    def dead(self, element_ids=None) -> th.Tensor:
        """Element deadness :math:`\\delta_x \\in [0, 1]` (the shared part)."""
        return self.elements.dead(element_ids)

    # -- the class model ----------------------------------------------------------------------
    def class_regions(self, class_ids=None) -> Regions:
        """The boxes of the classes (the class model)."""
        return self.boxes.regions(class_ids)

    def _aggregate_slot_depth(self, slot_depth, active, crisp=False):
        return self.depth_measure.aggregate(slot_depth, active, crisp=crisp)

    def _align_points_and_ids(self, points, ids, max_bound, id_name):
        return align_points_and_ids(points, ids, max_bound, id_name, self.embed_dim,
                                    self.elements.device)

    def depth(self, points, class_ids, top_id=None, bot_id=None, crisp=False) -> th.Tensor:
        """Signed depth ``(B,)`` of ``points`` in the classes (the class model)."""
        return self.boxes.depth(points, class_ids, top_id=top_id, bot_id=bot_id, crisp=crisp)

    def membership(self, points, class_ids, top_id=None, bot_id=None) -> th.Tensor:
        """Soft membership :math:`\\sigma(\\operatorname{dep}(z, A) / \\tau)` (the class model)."""
        return self.boxes.membership(points, class_ids, top_id=top_id, bot_id=bot_id)

    # -- the relation model -------------------------------------------------------------------
    def translation_regions(self, rel_ids=None) -> Regions:
        """The boxes of translations of the roles (``role_mode="translation"``)."""
        if self.role_mode != "translation":
            raise ValueError(
                f"translation_regions is only available when role_mode='translation', got "
                f"{self.role_mode!r}")
        return self.roles.regions(rel_ids)

    def role_regions(self, rel_ids=None):
        """``(head, tail)`` regions (product mode) or the boxes of translations."""
        return self.roles.regions(rel_ids)

    def _align_role_inputs(self, x_points, y_points, rel_ids):
        return self.roles.align(x_points, y_points, rel_ids)

    def role_depth(self, x_points, y_points, rel_ids, crisp=False) -> th.Tensor:
        """Signed depth ``(B,)`` of the pairs in the roles (the relation model)."""
        return self.roles.depth(x_points, y_points, rel_ids, crisp=crisp)

    def role_membership(self, x_points, y_points, rel_ids) -> th.Tensor:
        """Soft role membership :math:`\\sigma(\\text{role depth} / \\tau)`."""
        return self.roles.membership(x_points, y_points, rel_ids)

    def _align_rel_and_class_ids(self, rel_ids, class_ids):
        return self.roles.align_with_classes(rel_ids, class_ids, self.nb_classes)

    def _translation_roles(self, fn_name):
        if self.role_mode != "translation":
            raise ValueError(
                f"{fn_name} is only available when role_mode='translation', got "
                f"{self.role_mode!r}")
        return self.roles

    def exists_box(self, rel_ids, class_ids) -> Regions:
        """The boxes of :math:`\\exists r.B` (``role_mode="translation"``)."""
        return self._translation_roles("exists_box").preimage_boxes(rel_ids, class_ids,
                                                                    self.boxes)

    def exists_inverse_box(self, rel_ids, class_ids) -> Regions:
        """The boxes of :math:`\\exists r^-.B` (``role_mode="translation"``)."""
        return self._translation_roles("exists_inverse_box").preimage_boxes(
            rel_ids, class_ids, self.boxes, inverse=True)

    def exists_depth(self, points, rel_ids, class_ids, crisp=False) -> th.Tensor:
        """Depth ``(B,)`` of ``points`` in :math:`\\exists r.B` (``role_mode="translation"``)."""
        return self._translation_roles("exists_depth").exists_depth(points, rel_ids,
                                                                     class_ids, self.boxes,
                                                                     crisp=crisp)

    # -- capacity -----------------------------------------------------------------------------
    def expand_components(self, new_K: int, new_role_K: Optional[int] = None) -> None:
        """Append inactive zero-width components to the class model (``new_K``) and the
        relation model (``new_role_K``); every depth and membership stays bitwise the same.
        Replaces the parameters, so an optimiser must be re-created."""
        self.boxes.expand(_validate_int_ge("new_K", new_K, self.n_components))
        if new_role_K is not None:
            self.roles.expand(_validate_int_ge("new_role_K", new_role_K,
                                               self.n_role_components))

    def exact_limit_init(
        self,
        least_model_labels: Union[th.Tensor, np.ndarray],
        edges: Optional[
            Union[
                th.Tensor,
                np.ndarray,
                Iterable[Tuple[int, int, int]],
                Mapping[int, Iterable[Tuple[int, int]]],
            ]
        ] = None,
        delta: float = 0.05,
        seed: int = 0,
        dead_mask: Optional[Union[th.Tensor, np.ndarray]] = None,
    ) -> None:
        """Initialise parameters to the exact-limit §5 construction from GIVEN least-model labels and edges.

        Requires ``n_components >= nb_elements``. Places elements at random
        generic positions rescaled so that :math:`\\|p_i - p_j\\|_\\infty \\ge 4\\delta > 2\\delta`
        for all :math:`i \\neq j` (ensuring boxes of distinct elements are
        strictly disjoint). Sets component :math:`k` of class :math:`A` to
        :math:`\\operatorname{Box}(p_k, \\delta)` if and only if
        ``least_model_labels[k, A]`` is truthy (and marks unused components
        inactive with width ``0``). Sets each role edge :math:`(x, y, r)` to a
        product component :math:`\\operatorname{Box}(p_x, \\delta) \\times \\operatorname{Box}(p_y, \\delta)`.

        Guarantees:

        - Does not call any reasoner or completion algorithm; uses only the
          caller-supplied ``least_model_labels`` and ``edges``.
        - Distinct element boxes :math:`\\operatorname{Box}(p_i, \\delta)` and
          :math:`\\operatorname{Box}(p_j, \\delta)` are separated by at least
          :math:`2\\delta` in :math:`\\ell_\\infty` norm.
        - Unused class and role components have width ``0.0`` and ``active == False``.
        - Raises :class:`ValueError` if ``n_components < nb_elements`` or if
          ``least_model_labels`` does not have shape ``(nb_elements, nb_classes)``.

        Non-guarantees:

        - Soundness of the resulting configuration holds if and only if the
          supplied ``least_model_labels`` and ``edges`` form a valid model of the
          ontology.

        Parameters
        ----------
        least_model_labels : Union[torch.Tensor, numpy.ndarray]
            Boolean or binary matrix of shape ``(nb_elements, nb_classes)`` where
            entry ``(k, A)`` is truthy iff element ``k`` belongs to class ``A``
            in the least model.
        edges : Optional[Union[torch.Tensor, numpy.ndarray, Iterable[Tuple[int, int, int]], Mapping[int, Iterable[Tuple[int, int]]]]], default None
            Role edges as an iterable/tensor of ``(x, y, r)`` triples or a
            mapping ``{r: [(x, y), ...]}``.
        delta : float, default 0.05
            Positive box half-width :math:`\\delta > 0` around each element.
        seed : int, default 0
            Random seed for generic element position generation.
        dead_mask : Optional[Union[torch.Tensor, numpy.ndarray]], default None
            Optional boolean mask of shape ``(nb_elements,)`` marking dead
            elements; when ``None``, derived from ``least_model_labels`` (a class
            element ``k < nb_classes`` is live iff ``labels[k, k]`` is truthy and
            ``k != bot_id``; an individual element ``k >= nb_classes`` is live iff
            ``labels[k].any()`` is truthy).
        """
        from mowl.representation.library.finite_model_placement import FiniteModelPlacement
        placement = FiniteModelPlacement(delta=delta, seed=seed, dead_value=_DEAD_LOGIT,
                                         live_value=_LIVE_LOGIT)
        placement.place(self.elements, self.boxes, self.roles, least_model_labels, edges=edges,
                        dead_mask=dead_mask, bot_id=self.bot_id,
                        expand=lambda k: self.expand_components(self.n_components, new_role_K=k))

    # -- the common interface (mowl.representation.Geometry) ----------------------------------
    def classes(self, ids):
        """The regions of the named classes ``ids``: a union of boxes each, some of them
        inactive (:class:`~mowl.representation.library.other_regions.MaskedUnionOfBoxes`)."""
        return self.boxes(ids)

    def individuals(self, ids):
        """The points of the individuals ``ids`` (elements ``nb_classes + ids``)."""
        return self.individual_model(ids)

    def exists(self, relation_ids, region):
        """The region of :math:`\\exists r.C` (``role_mode="translation"``, named filler)."""
        from mowl.representation.library.other_regions import MaskedUnionOfBoxes
        from mowl.representation.regions import ConstructorUnavailable
        if self.role_mode != "translation":
            return self.roles.preimage(relation_ids, region)
        if region.ids is None:
            raise ConstructorUnavailable("The filler of exists must be a named class.")
        return MaskedUnionOfBoxes.of(self.exists_box(relation_ids, region.ids))

    def conjunction(self, a, b):
        """``(region, gap)`` of :math:`C \\sqcap D`: exact (all pairs of boxes)."""
        return a.intersect(b)

    def disjunction(self, a, b):
        """:math:`C \\sqcup D`: the boxes of both."""
        return a.union(b)

    def inside(self, a, b):
        """The measure's comparison of two unions of boxes (sufficient for inclusion); for a
        point ``a``, the hinge of its negative depth."""
        return self.depth_measure.inside(a, b, depth=self._region_depth)

    def apart(self, a, b):
        """Exact: zero iff no active box of ``a`` meets one of ``b``."""
        return self.depth_measure.apart(a, b)

    def _region_depth(self, points, region):
        if region.ids is not None:
            return self.depth(points, region.ids, crisp=True)
        return inside_measure(points, region.structural)

    def member(self, individual_ids, region):
        """The score of a class assertion: minus the signed depth of the individual in the
        region (negative inside, zero on the boundary)."""
        points = self.positions(self.elements.individual(individual_ids))
        return -self._region_depth(points, region).reshape(-1, 1)

    def soft_membership(self, individual_ids, region):
        """``sigmoid(depth / tau)`` (:meth:`membership`), for named classes."""
        from mowl.representation.regions import ConstructorUnavailable
        if region.ids is None:
            raise ConstructorUnavailable("Soft membership is defined for named classes.")
        points = self.positions(self.elements.individual(individual_ids))
        return self.membership(points, region.ids).reshape(-1, 1)

    @property
    def num_classes(self):
        return self.nb_classes

    @property
    def num_individuals(self):
        return self.nb_elements - self.nb_classes

    # -- the parts ----------------------------------------------------------------------------
    @property
    def shared(self):
        """The shared part: the element domain (positions, deadness, temperature)."""
        return self.elements

    @property
    def class_model(self):
        """The class model: a union of boxes per class (owns its boxes)."""
        return self.boxes

    @property
    def relation_model(self):
        """The relation model: a region over pairs (``role_mode="product"``) or boxes of
        translations (owns its boxes)."""
        return self.roles

    @property
    def individual_model(self):
        """The individual model: the point of the element (a view of the shared part)."""
        return ElementPoints(self.elements)

    @property
    def measure(self):
        """The measure: depth in a union of boxes, bounds per coordinate."""
        return self.depth_measure

    @property
    def parts(self):
        return {"classes": self.boxes.part, "relations": self.roles.part,
                "individuals": ElementPoints.part, "measure": self.depth_measure.part,
                "conjunction": "exact", "shared": self.elements.part}

    def forward(
        self,
        data: th.Tensor,
        gci_name: str,
        neg: bool = False,
    ) -> th.Tensor:
        """Score evaluation queries in the mOWL ``ELModule(data, gci_name)`` convention.

        Currently implements ``gci_name == "class_assertion"`` (with ``neg=False``),
        where ``data[:, 0]`` holds individual indices in ``[0, nb_elements - nb_classes)``
        (mapped to element rows ``nb_classes + data[:, 0]``) and ``data[:, 1]`` holds
        class indices in ``[0, nb_classes)``. Returns crisp negated signed depth
        :math:`-\\operatorname{dep}(p_a, C)` so that lower scores correspond to more
        plausible class assertions without sigmoid saturation.

        Guarantees:

        - For ``gci_name == "class_assertion"`` and ``neg=False``, returns a 1-D
          tensor of shape ``(B,)`` equal to ``-self.depth(self.positions(nb_classes + data[:, 0]), data[:, 1], crisp=True)``.
        - Entry ``b`` is strictly negative if and only if individual ``data[b, 0]``
          lies strictly inside at least one active component box of class ``data[b, 1]``
          (or ``data[b, 1] == top_id``).
        - Raises :class:`NotImplementedError` for ``gci_name != "class_assertion"``
          or ``neg=True``.
        - Raises :class:`ValueError` if ``data`` is not a 2-D integer tensor of shape
          ``(B, 2)`` or if any individual/class index is out of bounds.

        Non-guarantees:

        - Intended as an evaluation / harness scorer; the GroundedEL training loop
          calls :meth:`~mowl.models.GroundedEL.compute_batch_losses` directly rather
          than routing through ``forward``.

        Parameters
        ----------
        data : torch.Tensor
            2-D integer index tensor of shape ``(B, 2)`` with ``data[:, 0]`` = individual
            index and ``data[:, 1]`` = class index.
        gci_name : str
            Normal-form / assertion name; only ``"class_assertion"`` is supported.
        neg : bool, default False
            Negative-sampling flag; ``True`` is not supported and raises
            :class:`NotImplementedError`.

        Returns
        -------
        torch.Tensor
            1-D score tensor of shape ``(B,)`` where lower values indicate higher
            plausibility (:math:`-\\operatorname{dep}(p_a, C)`).
        """
        if neg:
            raise NotImplementedError(
                "GroundedELModule.forward does not support negative-sampled scoring (neg=True)"
            )
        if gci_name != "class_assertion":
            raise NotImplementedError(
                f"GroundedELModule.forward only supports gci_name='class_assertion', got {gci_name!r}"
            )
        if not isinstance(data, th.Tensor):
            raise TypeError(f"data must be a torch.Tensor, got {type(data).__name__}")
        if data.dtype == th.bool or data.is_floating_point() or data.is_complex():
            raise ValueError(f"data must have integer dtype, got {data.dtype}")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                f"class_assertion data must have shape (B, 2), got {tuple(data.shape)}"
            )

        device = self.elem_pos.weight.device
        dtype = self.elem_pos.weight.dtype
        bsz = int(data.shape[0])
        if bsz == 0:
            return th.empty((0,), dtype=dtype, device=device)

        nb_inds = self.elements.nb_individuals
        if nb_inds <= 0:
            raise ValueError(
                "Cannot score class_assertion when the module has no individual elements "
                f"(nb_elements={self.nb_elements}, nb_classes={self.nb_classes})"
            )

        ind_ids = self._to_index_tensor(data[:, 0], nb_inds, "individual_ids")
        cls_ids = self._to_index_tensor(data[:, 1], self.nb_classes, "class_ids")
        el_ids = self.elements.individual(ind_ids)
        pts = self.positions(el_ids)
        return -self.depth(pts, cls_ids, crisp=True)
