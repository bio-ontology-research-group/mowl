"""The common interface: what a trained model contains.

Whatever a method looks like inside, it answers the same questions: what is the region of
a class, of an individual, of :math:`\\exists r.C`, of :math:`C \\sqcap D`, and is one
region inside another. :class:`Geometry` is that interface. Training and evaluation code,
and people who use a trained model, go through it instead of reading a model's embedding
tables by name.

:class:`ComposedGeometry` is the geometry that is put together from independent parts: a
class model, a relation model, a measure and an objective. It is also an
:class:`~mowl.nn.ELModule`, so a composed geometry is trained like any other module.
"""

import torch as th

from mowl.nn.el.elmodule import ELModule
from mowl.representation.objectives import Objective, FORMS, ROLE_FORMS
from mowl.representation.regions import ConstructorUnavailable, Region

__all__ = ["Geometry", "ComposedGeometry"]


def _names(classes, others):
    """Names of region ``classes`` for a message; the module is added to a name that another
    region class in ``others`` also has (two different classes can be called ``Point``)."""
    others = others if isinstance(others, (tuple, list)) else (others,)
    clash = {c.__name__ for c in others}
    return [f"{c.__module__.rsplit('.', 1)[-1]}.{c.__name__}" if c.__name__ in clash
            and any(o is not c for o in others if o.__name__ == c.__name__) else c.__name__
            for c in classes]


class Geometry:
    """Interface of a geometry. Regions are whatever the geometry uses to represent a set
    (a :class:`~mowl.representation.regions.Region`, a fuzzy membership vector, the state
    of a query model); callers only pass them back to the geometry. All costs have shape
    ``(B, 1)``, are lower for "more true" and are zero when the relation holds exactly
    (for geometries whose measure is exact).
    """

    # -- regions ----------------------------------------------------------------------------
    def classes(self, ids):
        """Regions of the named classes ``ids`` (``LongTensor (B,)``)."""
        raise NotImplementedError

    def individuals(self, ids):
        """Regions (or points) of the individuals ``ids``."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no individuals.")

    def exists(self, relation_ids, region):
        """Region of :math:`\\exists r.C` for the relations ``relation_ids`` and the
        regions of the fillers."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no existential restriction.")

    def conjunction(self, a, b):
        """``(region, gap)`` of :math:`C \\sqcap D`; ``gap`` is zero iff the result is not
        empty."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no conjunction.")

    def negation(self, a):
        """Region of :math:`\\neg C`."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no negation.")

    def disjunction(self, a, b):
        """Region of :math:`C \\sqcup D`."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no disjunction.")

    # -- relations between regions ----------------------------------------------------------
    def inside(self, a, b):
        """Cost of :math:`A \\subseteq B`."""
        raise NotImplementedError

    def apart(self, a, b):
        """Cost of :math:`A \\cap B = \\emptyset`."""
        raise NotImplementedError

    def member(self, individual_ids, region):
        """Cost of the individuals being members of the regions."""
        return self.inside(self.individuals(individual_ids), region)

    # -- sizes ------------------------------------------------------------------------------
    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def num_individuals(self):
        return 0

    @property
    def parts(self):
        """A dictionary naming the parts of this geometry (vocabulary values of
        :mod:`mowl.representation`), for geometries that are put together from parts."""
        return {}


class ComposedGeometry(ELModule, Geometry):
    """A geometry put together from parts.

    :param classes: The class model (:class:`mowl.representation.class_models.Classes`).
    :param relations: The relation model
        (:class:`mowl.representation.relation_models.Relations`), or ``None`` for a
        geometry of classes only.
    :param measure: The measure (:class:`mowl.representation.measures.RegionMeasure`).
    :param objective: The reading of the axiom forms
        (:class:`mowl.representation.Objective`; default: the semantic one, strict, with
        margin 0).
    :param individuals: The individual model: a class model whose rows are individuals, or
        ``None``.
    :param regularizers: Penalties (:class:`mowl.training.Regularizer`) summed by
        :meth:`regularization_loss`.
    :param supported_forms: The axiom forms this geometry can read (and so be trained on):
        a collection of names of :data:`~mowl.representation.objectives.FORMS` and
        :data:`~mowl.representation.objectives.ROLE_FORMS`. Default: the concept and
        assertion forms, and the role forms if the relation model defines them.
    :param neg_capable_gcis: The forms that have a negative loss of their own (default:
        those of the objective, see its ``negatives``).

    The measure is told about the relation model (:meth:`RegionMeasure.bind
    <mowl.representation.measures.RegionMeasure.bind>`), for measures whose cost reads it
    (the triple score). The class model's ``part`` describes the construction of compound
    classes (:attr:`parts`).
    """

    def __init__(self, classes, relations, measure, objective=None, individuals=None,
                 regularizers=(), supported_forms=None, neg_capable_gcis=None):
        ELModule.__init__(self)
        self.class_model = classes
        self.relation_model = relations
        self.individual_model = individuals
        self.measure = measure
        self.objective = objective if objective is not None else Objective()
        self.regularizers = tuple(regularizers)
        self._check_parts()
        if relations is not None:
            measure.bind(relations)
        # Names other code expects of an EL module (class models without a table, such as
        # text encoders, have none).
        self.class_embed = getattr(classes, "centres", None)
        self.rel_embed = getattr(relations, "vectors", None)
        self.ind_embed = getattr(individuals, "centres", None)
        if supported_forms is None:
            supported_forms = FORMS + (ROLE_FORMS if self.role_axiom_capable else ())
        unknown = set(supported_forms) - set(FORMS + ROLE_FORMS)
        if unknown:
            raise ValueError(f"Unknown axiom forms {sorted(unknown)}.")
        self.supported_forms = frozenset(supported_forms)
        self.neg_capable_gcis = frozenset(
            self.objective.neg_forms if neg_capable_gcis is None else neg_capable_gcis)
        self.margin_semantics = "slack" if self.objective.slack else "strict"

    def replace_parts(self, classes=None, relations=None, individuals=None, measure=None,
                      objective=None, regularizers=None):
        """Replace parts of this geometry (built parts, not names; ``None`` keeps the part;
        :func:`mowl.representation.swap.build_method` resolves the names). The pairs that
        cannot be combined raise :class:`~mowl.representation.ConstructorUnavailable`,
        naming both sides, and leave the geometry as it was.

        :param regularizers: A tuple that *replaces* the regularisers.
        """
        old = (self.class_model, self.relation_model, self.individual_model, self.measure,
               self.objective, self.regularizers, self.supported_forms,
               self.neg_capable_gcis)
        default_forms = self.supported_forms == frozenset(
            FORMS + (ROLE_FORMS if self.role_axiom_capable else ()))
        default_neg = self.neg_capable_gcis == frozenset(self.objective.neg_forms)
        try:
            if classes is not None:
                self.class_model = classes
            if relations is not None:
                self.relation_model = None if relations is False else relations
            if individuals is not None:
                self.individual_model = individuals
            if measure is not None:
                self._assign("measure", measure)
            if objective is not None:
                self.objective = objective
            if regularizers is not None:
                self.regularizers = tuple(regularizers)
            self._check_parts()
            if self.relation_model is not None:
                self.measure.bind(self.relation_model)
            self._assign("class_embed", getattr(self.class_model, "centres", None))
            self._assign("rel_embed", getattr(self.relation_model, "vectors", None))
            self._assign("ind_embed", getattr(self.individual_model, "centres", None))
            if relations is not None and default_forms:
                self.supported_forms = frozenset(
                    FORMS + (ROLE_FORMS if self.role_axiom_capable else ()))
            if objective is not None:
                if default_neg:
                    self.neg_capable_gcis = frozenset(self.objective.neg_forms)
                self.margin_semantics = "slack" if self.objective.slack else "strict"
        except Exception:
            (self.class_model, self.relation_model, self.individual_model, self.measure,
             self.objective, self.regularizers, self.supported_forms,
             self.neg_capable_gcis) = old
            self._assign("class_embed", getattr(self.class_model, "centres", None))
            self._assign("rel_embed", getattr(self.relation_model, "vectors", None))
            self._assign("ind_embed", getattr(self.individual_model, "centres", None))
            self._assign("measure", self.measure)
            raise
        return self

    def _assign(self, name, value):
        """Set an attribute that may be registered as a parameter or a module of another
        type."""
        for registry in (self._parameters, self._modules, self._buffers):
            registry.pop(name, None)
        self.__dict__.pop(name, None)
        setattr(self, name, value)

    def _check_parts(self):
        classes, measure, individuals = self.class_model, self.measure, self.individual_model

        def compared(region):
            return issubclass(region, measure.accepts) or (
                isinstance(region, type) and issubclass(region, Region)
                and region.viewable_as(measure.accepts))
        if not compared(classes.region):
            raise ConstructorUnavailable(
                f"{type(measure).__name__} compares "
                f"{_names(measure.accepts, classes.region)} regions; "
                f"{type(classes).__name__} gives {_names((classes.region,), measure.accepts)} "
                f"regions.")
        if individuals is not None and not compared(individuals.region):
            raise ConstructorUnavailable(
                f"{type(measure).__name__} compares "
                f"{_names(measure.accepts, individuals.region)} regions; the individual model "
                f"{type(individuals).__name__} gives "
                f"{_names((individuals.region,), measure.accepts)} regions.")

    @property
    def strict_negatives(self):
        """The ``strict_negatives`` of the objective (the one rule of every EL module: a
        negative loss of a form without one is an error): the geometry's per-axiom losses
        and the objective cannot disagree."""
        return self.objective.strict_negatives

    @strict_negatives.setter
    def strict_negatives(self, value):
        self.objective.strict_negatives = bool(value)

    # -- Geometry -----------------------------------------------------------------------
    def classes(self, ids, frozen=False):
        """The regions of ``ids``; ``frozen`` detaches their sizes (the class model must
        accept ``detach_size``; see ``freeze_negatives`` of the objective)."""
        if frozen:
            return self.class_model.regions(ids, detach_size=True)
        return self.class_model(ids)

    def individuals(self, ids):
        if self.individual_model is None:
            raise ConstructorUnavailable("This geometry has no individual model.")
        return self.individual_model(ids)

    def exists(self, relation_ids, region):
        if self.relation_model is None:
            raise ConstructorUnavailable("This geometry has no relation model.")
        return self.relation_model.preimage(relation_ids, region)

    def conjunction(self, a, b):
        return self.measure.intersect(a, b)

    def negation(self, a):
        return a.complement()

    def disjunction(self, a, b):
        return a.union(b)

    def inside(self, a, b):
        return self.measure.inside(a, b, self.objective.m)

    def apart(self, a, b):
        return self.measure.apart(a, b, self.objective.margin)

    def member(self, individual_ids, region):
        return self.measure.member(self.individuals(individual_ids), region,
                                   self.objective.m)

    @property
    def num_classes(self):
        return len(self.class_model)

    @property
    def num_individuals(self):
        return 0 if self.individual_model is None else len(self.individual_model)

    @property
    def role_axiom_capable(self):
        """Whether the relation model defines the role axioms (and so the geometry reads
        ``role_inclusion`` and ``role_chain``)."""
        return self.relation_model is not None and self.relation_model.reads_role_axioms

    @property
    def scored_forms(self):
        """The forms of this geometry that are scored (those it can read, in order)."""
        return tuple(f for f in FORMS + ROLE_FORMS if f in self.supported_forms)

    @property
    def parts(self):
        conjunction = self.class_model.conjunction_kind
        parts = {"classes": self.class_model.part,
                 "relations": "none" if self.relation_model is None
                 else self.relation_model.part,
                 "individuals": "none" if self.individual_model is None else "region",
                 "measure": self.measure.part,
                 "conjunction": conjunction}
        parts.update(self.class_model.extra_parts)
        return parts

    # -- ELModule -----------------------------------------------------------------------
    @property
    def margin(self):
        """The margin of the objective. For the geometry of an existing module
        (:func:`~mowl.representation.geometry_of`) this *is* the module's margin: changing
        either changes the losses of both. Its sense is :attr:`margin_semantics`."""
        return self.objective.margin

    @margin.setter
    def margin(self, value):
        self.objective.margin = float(value)

    def link_margin(self, owner):
        """Make this geometry's margin the margin of ``owner`` (its ``margin_attribute``,
        default ``"margin"``), so that the two are one number (see
        :meth:`Objective.link_margin <mowl.representation.Objective.link_margin>`)."""
        self.objective.link_margin(owner, getattr(owner, "margin_attribute", "margin"))
        return self

    def tables(self):
        """The raw parameter tables of the parts, ``{"classes": {"centres": ..., ...},
        "relations": {...}, "individuals": {...}}`` (parts that have none are left out):
        the trained tensors, not copies, for regimes that initialise or inspect the raw
        parameters."""
        parts = {"classes": self.class_model, "relations": self.relation_model,
                 "individuals": self.individual_model}
        return {name: part.tables() for name, part in parts.items()
                if part is not None and hasattr(part, "tables")}

    def cost(self, form, rows, neg=False):
        """Loss of the axioms ``rows`` of form ``form``."""
        self.last_form = form
        return self.objective.cost(self, form, rows, neg)

    def forward(self, gci, gci_name, neg=False):
        # remembered for losses that depend on the form (ComposedEL.compute_loss)
        self.last_form = gci_name
        return super().forward(gci, gci_name, neg)

    def get_loss_function(self, gci_name):
        if gci_name in FORMS or gci_name in ROLE_FORMS:
            return lambda rows, neg=False: self.cost(gci_name, rows, neg)
        return super().get_loss_function(gci_name)

    @property
    def device(self):
        """The device of the parameters (or buffers); ``cpu`` for a geometry without any."""
        for tensor in list(self.parameters()) + list(self.buffers()):
            return tensor.device
        return th.device("cpu")

    def regularization_loss(self):
        return self.penalty_sum(self.regularizers)

    def penalty_sum(self, regularizers):
        """The sum of the penalties of ``regularizers`` (those that have one) on this geometry,
        zero if none does."""
        total = th.zeros((), device=self.device)
        for regularizer in regularizers:
            term = regularizer.penalty(self)
            if term is not None:
                total = total + term
        return total

    # -- training -----------------------------------------------------------------------
    def project(self):
        """Project the parameters of the class model and the relation model back onto their
        constraint set (the unit cube) where they have one; a training loop calls it after
        each step (a no-op otherwise)."""
        for part in (self.class_model, self.relation_model, self.individual_model):
            project = getattr(part, "project", None)
            # (a part may own a *module* called project, which is not this hook)
            if callable(project) and not isinstance(project, th.nn.Module):
                project()

    def param_groups(self):
        """Parameter groups for :class:`~mowl.representation.library.manifolds.RiemannianSGD`: those of
        the class model, with the manifold of every table that lives on one, and the rest."""
        groups = list(self.class_model.param_groups())
        seen = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.parameters() if id(p) not in seen]
        if rest:
            groups.append({"params": rest, "manifold": None})
        return groups


def _loss_method(form):
    # (the argument is called ``data`` as in the released modules, which callers may name)
    def loss(self, data, neg=False):
        return self.cost(form, data, neg)
    loss.__name__ = f"{form}_loss"
    loss.__doc__ = f"Loss of the ``{form}`` axioms ``data`` (see the objective)."
    return loss


def _make_losses():
    for form in FORMS + ROLE_FORMS:
        setattr(ComposedGeometry, f"{form}_loss", _loss_method(form))


_make_losses()
