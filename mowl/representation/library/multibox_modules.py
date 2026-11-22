"""What the public modules of the box-with-bump family add to a
:class:`~mowl.representation.ComposedGeometry`.

:class:`~mowl.nn.BoxSquaredELModule`, :class:`~mowl.nn.MultiBoxELModule`,
:class:`~mowl.nn.MultiELBEModule` and :class:`~mowl.nn.VolumeMultiBoxELModule` are released
classes: code reads their parameter tables by name (``module.class_center``,
``module.head_offset``, ...) and saved models hold those names. Each of them is now the
combination of parts that its method is (:mod:`~mowl.representation.library.multibox_methods`);
this file gives them back the names of the tables, as read-only views of the tables of the
parts, and lets a state dictionary written under those names be loaded.
"""

import torch as th
import torch.nn as nn

from mowl.representation.objectives import ROLE_FORMS

__all__ = ["released_names", "ReleasedTables", "table_of"]


def _refuse_role(module, form, rows):
    """The error of a module without the role axioms of EL++ (what
    :class:`~mowl.nn.ELModule` says)."""
    from mowl.nn.el.elmodule import ELModule
    return getattr(ELModule, f"{form}_loss")(module, rows)


def table_of(path, doc=None):
    """A read-only property: the object at the dotted ``path`` below the instance (``None``
    if a step of the path is missing, such as the individual model of a module without
    individuals)."""
    def get(self):
        obj = self
        for name in path.split("."):
            if obj is None:
                return None
            obj = getattr(obj, name, None)
        return obj
    return property(get, doc=doc or f"The table ``{path}`` of the parts.")


#: Names that :class:`~mowl.representation.ComposedGeometry` gives to the tables of the class
#: model, the relation model and the individual model (the same tables, a second time).
_ALIASES = ("class_embed", "rel_embed", "ind_embed")


def released_names(state=None, **paths):
    """Class decorator: give the class the attribute ``name`` for every ``name=path`` (see
    :func:`table_of`), and remember them for :class:`ReleasedTables`.

    :param state: The names (of ``paths``) under which the *state dictionary* of the earlier
        module holds its tables; default: all of them. Other tables of the parts are listed
        under these names too, and the names the geometry gives its tables
        (``class_embed``, ``rel_embed``, ``ind_embed``) are left out unless they are among
        them: the state dictionary of the module is that of the module it replaces.
    """
    def decorate(cls):
        for name, path in paths.items():
            if name not in _ALIASES:
                setattr(cls, name, table_of(path))
        cls.released_tables = {**getattr(cls, "released_tables", {}), **paths}
        cls.released_state = tuple(paths if state is None else state)
        return cls
    return decorate


def in_chunks(function, rows, step):
    """``function(rows)`` evaluated ``step`` rows at a time and concatenated: the same
    values with a bounded memory use. ``step=None`` (or at least ``len(rows)``) evaluates
    all rows at once."""
    if step is None or len(rows) <= step:
        return function(rows)
    return th.cat([function(rows[i:i + step]) for i in range(0, len(rows), step)])


class ReleasedTables:
    """Mixin for a module whose parameters are held by parts: it keeps the interface of the
    module it replaces. The state dictionary has the names of that module
    (``class_center.weight``, ...) and loads under them; state dictionaries saved by the parts
    under their own names load too. A loss of a form that earlier modules accepted in a form
    of its own (a single column for :math:`\\exists r.C \\sqsubseteq \\bot`, an error
    about individuals) is accepted."""

    released_tables = {}
    released_state = ()
    #: The tables in the order in which the earlier module drew their initial values.
    released_order = ()
    #: The forms whose loss has the flat shape ``(B,)`` when ``per_axiom_column`` is ``False``.
    flat_forms = ()

    @staticmethod
    def rng_state():
        """The state of the random generator: take it before the parts are made and hand it
        to :meth:`initialise_in_released_order`."""
        return th.get_rng_state()

    def initialise_in_released_order(self, rng_state):
        """Draw the initial values of the tables in the order of the earlier module, from the
        state ``rng_state`` of the generator (that of before the parts were made): a seeded
        module has the numbers it had. Every table is uniform in :math:`[-1, 1]` with rows (or
        the blocks of a row, for a union of boxes) of length one."""
        th.set_rng_state(rng_state)
        components = getattr(self.class_model, "components", 1)
        for path in self.released_order:
            table = table_of(path).fget(self)
            if table is None:
                continue
            rows, width = table.weight.shape
            parts = components if path.startswith("class_model.") \
                and path.split(".")[1] in ("centres", "offsets") else 1
            fresh = nn.Embedding(rows, width)
            with th.no_grad():
                nn.init.uniform_(fresh.weight, a=-1, b=1)
                if rows > 0:
                    view = fresh.weight.view(rows, parts, -1)
                    view.div_(th.linalg.norm(view, dim=-1, keepdim=True))
                table.weight.copy_(fresh.weight)

    def cost(self, form, rows, neg=False):
        if form in ROLE_FORMS and not self.role_axiom_capable:
            return _refuse_role(self, form, rows)
        if neg and self.strict_negatives and form not in self.neg_capable_gcis:
            raise NotImplementedError(
                f"{type(self).__name__} has no negative loss for '{form}' (it has one for "
                f"{sorted(self.neg_capable_gcis) or 'no form'}), so neg=True is refused. Do "
                f"not ask for a negative of this form (negative sampling is restricted to "
                f"'neg_capable_gcis' by default), or set 'strict_negatives = False' on the "
                f"module to get the positive loss back instead.")
        out = self._cost(form, rows, neg)
        if not self.per_axiom_column and form in self.flat_forms and out.ndim == 2 \
                and out.shape[1] == 1:
            return out.squeeze(1)
        return out

    def _cost(self, form, rows, neg=False):
        if form in ("class_assertion", "object_property_assertion") \
                and self.individual_model is None:
            raise ValueError("The number of individuals must be specified to use this loss "
                             "function.")
        if form == "gci3_bot" and rows.shape[1] == 1:
            # the earlier modules read the role of an axiom :math:`\\exists r.C \\sqsubseteq
            # \\bot` alone (the head box of the role is empty, whatever the filler is): a
            # single column is the role, and any filler gives the same loss
            rows = th.cat([rows, th.zeros_like(rows), th.zeros_like(rows)], dim=1)
        return super().cost(form, rows, neg)

    def _legacy_keys(self, names):
        """``{legacy key: [the names of the table in the parts]}`` for the tables of
        ``names`` (attributes of this class): every name the table has in the module (the
        class model's, the alias that the geometry gives it, the one below a wrapper of the
        class model)."""
        parameters = list(self.named_parameters(remove_duplicate=False))
        out = {}
        for legacy in names:
            table = table_of(self.released_tables[legacy]).fget(self)
            if table is None:
                continue
            weight = table.weight if hasattr(table, "weight") else table
            suffix = ".weight" if hasattr(table, "weight") else ""
            out[legacy + suffix] = [name for name, p in parameters if p is weight]
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        out = super().state_dict(*args, destination=destination, prefix=prefix,
                                 keep_vars=keep_vars)
        rename = {name: legacy for legacy, names in self._legacy_keys(self.released_state)
                  .items() for name in names}
        drop = set(_ALIASES) - set(self.released_state)
        for key in list(out):
            if not key.startswith(prefix):
                continue
            local = key[len(prefix):]
            if local in rename:
                value = out.pop(key)
                out.setdefault(prefix + rename[local], value)
            elif local.split(".")[0] in drop or local.startswith("class_model.def_"):
                # an alias of a table that has its own name, or the tables of the definitions
                # that a structural class model is wrapped in (not part of the weights)
                del out[key]
        return out

    def load_state_dict(self, state_dict, strict=True, assign=False):
        own = th.nn.Module.state_dict(self)
        legacy = self._legacy_keys(self.released_tables)
        renamed = {}
        for key, value in state_dict.items():
            if key in own:
                renamed[key] = value
            elif key in legacy:
                for name in legacy[key]:
                    renamed.setdefault(name, value)
            elif key.replace("class_model.base.", "class_model.") in own:
                renamed[key.replace("class_model.base.", "class_model.")] = value
            elif key.startswith("class_model.def_"):
                continue
            else:
                renamed[key] = value
        # tables the file does not have, because they are not weights: the definitions of a
        # structural class model; the aliases of a table that the file has under one name
        for key, value in own.items():
            if key not in renamed and key.startswith("class_model.def_"):
                renamed[key] = value
        by_table = {}
        for key, value in own.items():
            by_table.setdefault((value.data_ptr(), tuple(value.shape)), []).append(key)
        for keys in by_table.values():
            given = [k for k in keys if k in renamed]
            for key in keys:
                if given and key not in renamed:
                    renamed[key] = renamed[given[0]]
        return super().load_state_dict(renamed, strict=strict, assign=assign)


# ====================================================================================
# the modules of unions of boxes (MultiBoxEL, MultiELBE)
# ====================================================================================

from mowl.representation.library.multibox_objective import ComponentSpread, DiversityPenalty
from mowl.representation.library.multibox_parts import check_temperature as _validate_tau


def _positive(name, value):
    """``value`` as a float; an error unless it is a positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive float, got {value!r}")
    return float(value)


def _validate_beta(beta):
    return _positive("beta", beta)


def _validate_ind_halfwidth(ind_halfwidth):
    return _positive("ind_halfwidth", ind_halfwidth)

__all__ += ["UnionModule", "check_nonnegative", "check_union_options"]


def check_nonnegative(name, value):
    """``value`` as a float; an error unless it is a number that is not negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return float(value)


def check_union_options(n_components, embed_dim, component_aggregation, tau, diversity_reg,
                        beta, ind_halfwidth, neg_weight, volume_reg):
    """The checks of the options that MultiBoxEL and MultiELBE modules share; returns
    ``(tau, beta, ind_halfwidth)`` as floats (``ind_halfwidth`` is :math:`5\\beta` if
    ``None``)."""
    if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
        raise ValueError(f"n_components must be an int >= 1, got {n_components!r}")
    if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim < 1:
        raise ValueError(f"embed_dim must be an int >= 1, got {embed_dim!r}")
    if component_aggregation not in ("hard", "soft"):
        raise ValueError(
            f"component_aggregation must be 'hard' or 'soft', got {component_aggregation!r}")
    tau = _validate_tau(tau)
    check_nonnegative("diversity_reg", diversity_reg)
    beta = _validate_beta(beta)
    ind_halfwidth = 5.0 * beta if ind_halfwidth is None \
        else _validate_ind_halfwidth(ind_halfwidth)
    check_nonnegative("neg_weight", neg_weight)
    check_nonnegative("volume_reg", volume_reg)
    return tau, beta, ind_halfwidth


class UnionModule(ReleasedTables):
    """What :class:`~mowl.nn.MultiBoxELModule` and :class:`~mowl.nn.MultiELBEModule` share:
    the temperature of the soft aggregation as a buffer that is updated in place, the numbers
    of the readings as views of the parts that hold them, the point objectives
    (:class:`~mowl.representation.library.structural_points.PointTerms`) and the components of
    a class.

    The *reading* of the module (``"axiom"``, ``"point"``, ...; the argument ``objective`` of
    the constructor, which is an :class:`~mowl.representation.Objective` once the module is
    built) is fixed when the module is made.
    """

    #: ``"axiom"``, ``"structural"``, ``"point"``, ``"mixed"`` or ``"hybrid"``.
    reading = "axiom"

    # -- the temperature of the soft aggregation ------------------------------------------
    def _install_tau(self, tau):
        """Keep ``tau`` as the buffer ``tau`` of the module (written in saved models under
        that name) and make the measure read it."""
        self.register_buffer("tau", th.tensor(_validate_tau(tau), dtype=th.float32))
        self.measure.tau = self._buffers["tau"]

    def __setattr__(self, name, value):
        if name == "tau" and "tau" in self.__dict__.get("_buffers", {}):
            self._buffers["tau"].fill_(_validate_tau(value))
            return
        if name == "objective" and isinstance(value, str):
            raise ValueError(
                f"The reading of the module (objective={self.reading!r}) is fixed when it is "
                f"made: it chooses the parts the module is built from. Make a new module "
                f"with objective={value!r}; module.objective is the Objective of the parts.")
        super().__setattr__(name, value)

    # -- the regularisers -----------------------------------------------------------------
    def _regularizer_list(self):
        return self.__dict__.get("base_regularizers", self.regularizers)

    def _weight_of(self, kind):
        for regularizer in self._regularizer_list():
            if isinstance(regularizer, kind):
                return regularizer.weight
        return 0.0

    def _set_weight_of(self, kind, value):
        for regularizer in self._regularizer_list():
            if isinstance(regularizer, kind):
                regularizer.weight = float(value)
                return
        if float(value) > 0:
            added = tuple(self._regularizer_list()) + (kind(float(value)),)
            if "base_regularizers" in self.__dict__:
                self.base_regularizers = added
                self._set_penalties()
            else:
                self.regularizers = added

    @property
    def component_reg(self):
        """The weight of :class:`~mowl.representation.library.multibox_objective.\
ComponentSpread`."""
        return self._weight_of(ComponentSpread)

    @component_reg.setter
    def component_reg(self, value):
        self._set_weight_of(ComponentSpread, value)

    @property
    def diversity_reg(self):
        """The weight of :class:`~mowl.representation.library.multibox_objective.\
DiversityPenalty`."""
        return self._weight_of(DiversityPenalty)

    @diversity_reg.setter
    def diversity_reg(self, value):
        self._set_weight_of(DiversityPenalty, check_nonnegative("diversity_reg", value))

    @property
    def component_aggregation(self):
        """``"hard"`` or ``"soft"`` (the aggregation of the measure of unions of boxes)."""
        return getattr(self.measure, "aggregation", "hard")

    @component_aggregation.setter
    def component_aggregation(self, value):
        if value not in ("hard", "soft"):
            raise ValueError(f"component_aggregation must be 'hard' or 'soft', got {value!r}")
        if value != "hard" and self.reading == "structural":
            raise ValueError("component_aggregation must be 'hard' when "
                             "objective='structural'.")
        self.measure.aggregation = value

    # -- the margin -----------------------------------------------------------------------
    @property
    def gamma(self):
        """The margin of the objective (the :math:`\\gamma` of Box\\ :sup:`2`\\ EL)."""
        return self.objective.margin

    @gamma.setter
    def gamma(self, value):
        self.objective.margin = float(value)

    # -- the point objectives -------------------------------------------------------------
    @property
    def beta(self):
        """Temperature :math:`\\beta` of the soft membership of the point objectives."""
        return self.point_terms.measure.beta

    @beta.setter
    def beta(self, value):
        value = _validate_beta(value)
        self.point_terms.measure.beta = value
        if not self.__dict__.get("_custom_ind_halfwidth", False):
            self.point_terms.ind_halfwidth = 5.0 * value

    @property
    def ind_halfwidth(self):
        """Half-width of the virtual box around an individual target (default
        :math:`5\\beta`)."""
        return self.point_terms.ind_halfwidth

    @ind_halfwidth.setter
    def ind_halfwidth(self, value):
        self.point_terms.ind_halfwidth = _validate_ind_halfwidth(value)
        self.__dict__["_custom_ind_halfwidth"] = True

    @property
    def neg_weight(self):
        """Weight of the random negatives of the point objectives."""
        return self.point_terms.neg_weight

    @neg_weight.setter
    def neg_weight(self, value):
        self.point_terms.neg_weight = check_nonnegative("neg_weight", value)

    @property
    def volume_reg(self):
        """Weight of the volume penalty of the point objectives."""
        return self.point_terms.volume_reg

    @volume_reg.setter
    def volume_reg(self, value):
        self.point_terms.volume_reg = check_nonnegative("volume_reg", value)

    @property
    def point_supervision(self):
        """The :class:`~mowl.datasets.el.points.PointSupervision` of the point objectives, or
        ``None``."""
        return self.point_terms.supervision

    def set_point_supervision(self, ps):
        """Attach a :class:`~mowl.datasets.el.points.PointSupervision` and create the table
        ``elem_embed`` of the coordinates of its canonical elements (on the device and dtype of
        the classes; replaced if the number of elements changes)."""
        terms = self.point_terms
        weight = self.class_model.centres.weight
        if terms.elem_embed is None or terms.elem_embed.num_embeddings != ps.n_elements:
            terms.set_supervision(ps)
            terms.elem_embed.to(device=weight.device, dtype=weight.dtype)
        else:
            terms.supervision = ps

    def _need_points(self):
        if self.point_terms.supervision is None or self.point_terms.elem_embed is None:
            raise RuntimeError("PointSupervision has not been set on this module; call "
                               "set_point_supervision(ps) first.")

    def element_embeddings(self, elem_idx):
        """Coordinates ``(B, d)`` of canonical elements: the classes' from ``elem_embed``, the
        individuals' from the table of the individuals."""
        self._need_points()
        return self.point_terms.points(self, elem_idx)

    def point_membership_loss(self, batch, neg=False, weight=None):
        """Binary cross-entropy of the soft membership of element points in classes, for a
        ``(B, 2)`` tensor ``[element, class]`` (target 0 if ``neg``) or a sample dictionary
        (``pos``, ``hard``, ``rand``)."""
        self._need_points()
        return self.point_terms.membership_loss(self, batch, neg=neg, weight=weight)

    def target_membership(self, points, elem_y):
        """Soft membership of ``points`` ``(B, d)`` in the target elements ``elem_y``."""
        self._need_points()
        return self.point_terms.target_membership(self, points, elem_y)

    def point_role_loss(self, batch, neg=False, weight=None):
        """Binary cross-entropy of the membership of canonical role triples
        ``[x, r, y]`` (see :meth:`PointTerms.role_loss
        <mowl.representation.library.structural_points.PointTerms.role_loss>`)."""
        self._need_points()
        return self.point_terms.role_loss(self, batch, neg=neg, weight=weight)

    def class_assertion_score_point(self, data):
        """:math:`-\\log(\\max(m_C(x_a), \\varepsilon))` for ``data`` ``[individual, class]``,
        shape ``(B,)``."""
        if self.individual_model is None:
            raise ValueError("The number of individuals must be specified to use this loss "
                             "function.")
        return self.point_terms.class_assertion_score(self, data).squeeze(1)

    def point_modelness(self, ps=None, chunk_size=8192):
        """``pos_frac`` (:math:`m \\ge 0.5`) and ``hardneg_frac`` (:math:`m < 0.5`) of the
        pairs of a supervision."""
        if ps is None:
            ps = self.point_supervision
        if ps is None:
            raise ValueError("PointSupervision must be provided or set via "
                             "set_point_supervision(ps).")
        if self.point_supervision is None:
            self.set_point_supervision(ps)
        return self.point_terms.modelness(self, ps, chunk_size)

    # -- the components of a class --------------------------------------------------------
    def class_boxes(self, idx):
        """``(centres, offsets)`` ``(B, k, d)`` of the ``k`` boxes of the classes ``idx``;
        the offsets are not negative."""
        classes = self.class_model
        classes = getattr(classes, "base", classes)
        shape = (idx.shape[0], classes.components, classes.embed_dim)
        return classes.centres(idx).view(shape), th.abs(classes.offsets(idx)).view(shape)

    @property
    def n_components(self):
        """Boxes per class."""
        classes = self.class_model
        return getattr(getattr(classes, "base", classes), "components", 1)

    def effective_components(self, eps=1e-3):
        """Per class the number of components that are not degenerate and not nested in
        another (:func:`mowl.nn.el.multibox_utils.effective_components`)."""
        from mowl.nn.el.multibox_utils import effective_components
        classes = self.class_model
        classes = getattr(classes, "base", classes)
        shape = (len(classes), classes.components, classes.embed_dim)
        return effective_components(classes.centres.weight.view(shape),
                                    classes.offsets.weight.view(shape), eps=eps)
