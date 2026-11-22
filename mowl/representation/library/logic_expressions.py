"""Class expressions on any geometry.

:func:`expression_region` evaluates an OWL class expression (an OWL API object, as the
datasets give them) to a region of a :class:`~mowl.representation.Geometry`, using nothing
but the interface of the geometry: ``classes``, ``individuals`` (nominals),
``conjunction``, ``disjunction``, ``negation`` and ``exists``. A geometry that lacks a
constructor raises :class:`~mowl.representation.ConstructorUnavailable`, naming the
constructor and the geometry. On top of it:

* :func:`subsumption_cost`: the cost of :math:`C \\sqsubseteq D` for two class expressions,
* :func:`rank_classes`: the named classes :math:`A` ordered by the cost of
  :math:`A \\sqsubseteq C` (as in ``answer`` of the query models, where named classes are
  answers),
* :func:`rank_individuals`: the individuals ordered by the cost of being members of
  :math:`C` (as in ``rank_members`` of E2R).

All costs are lower for "more true"; the ranking functions return ``(iri, cost)`` pairs,
best first. The functions need the id maps of a dataset (``class_to_id``,
``object_property_to_id``, ``individual_to_id``, as in
``Model.class_index_dict`` and friends, see :func:`id_maps`); an expression may also be a
class IRI string.

The handling follows the DL equivalences where the geometry has no constructor of its own:
:math:`\\forall r.C = \\neg\\exists r.\\neg C`, :math:`\\exists r.\\{a\\}` for
``ObjectHasValue``, a nominal with several individuals is their disjunction, and
``owl:Nothing`` is the complement of ``owl:Thing`` if the geometry has no class for it.
``owl:Thing`` and ``owl:Nothing`` are taken from the classes of the geometry when the id
map has them, or from ``geometry.thing()`` / ``geometry.nothing()`` when a geometry offers
those.
"""

import dataclasses

import torch as th

from mowl.owlapi import ClassExpressionType
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["expression_region", "pattern_region", "subsumption_cost", "rank_classes",
           "rank_individuals", "id_maps", "THING", "NOTHING"]

THING = "http://www.w3.org/2002/07/owl#Thing"
NOTHING = "http://www.w3.org/2002/07/owl#Nothing"

_T = ClassExpressionType


def id_maps(model):
    """``(class_to_id, object_property_to_id, individual_to_id)`` of a model or dataset
    model (the ``*_index_dict`` properties of mOWL models)."""
    return (model.class_index_dict, model.object_property_index_dict,
            model.individual_index_dict)


def _device(geometry):
    for owner in (geometry, getattr(geometry, "module", None)):
        try:
            return next(owner.parameters()).device
        except (AttributeError, StopIteration):
            continue
    return th.device("cpu")


def _iri(entity):
    return str(entity.toStringID())


def _missing(geometry, what):
    return ConstructorUnavailable(
        f"{type(geometry).__name__} has no {what}, so a class expression that needs it "
        f"cannot be evaluated.")


class _Evaluator:
    def __init__(self, geometry, class_to_id, object_property_to_id, individual_to_id):
        self.g = geometry
        self.classes = class_to_id
        self.properties = object_property_to_id
        self.individuals = individual_to_id or {}
        self.device = _device(geometry)
        self.gap = None            # accumulated gap of the conjunctions of the expression

    def ids(self, value):
        return th.tensor([value], dtype=th.long, device=self.device)

    def add_gap(self, gap):
        self.gap = gap if self.gap is None else self.gap + gap

    # -- constructors, each with a clear failure ----------------------------------------------
    def call(self, name, what, *args):
        try:
            return getattr(self.g, name)(*args)
        except ConstructorUnavailable as error:
            raise ConstructorUnavailable(
                f"{error} A class expression with {what} cannot be evaluated by "
                f"{type(self.g).__name__}.") from None

    def named(self, iri):
        if iri in self.classes:
            return self.g.classes(self.ids(self.classes[iri]))
        if iri == THING and hasattr(self.g, "thing"):
            return self.g.thing()
        if iri == NOTHING and hasattr(self.g, "nothing"):
            return self.g.nothing()
        if iri == NOTHING:
            return self.call("negation", "owl:Nothing", self.named(THING))
        if iri == THING:
            raise _missing(self.g, "region for owl:Thing (it is not among the classes)")
        raise KeyError(f"Class {iri} is not in the class map.")

    def individual(self, individual):
        iri = _iri(individual)
        if iri not in self.individuals:
            raise KeyError(f"Individual {iri} is not in the individual map.")
        return self.call("individuals", "a nominal", self.ids(self.individuals[iri]))

    def fold(self, name, what, operands):
        regions = [self.evaluate(o) for o in operands]
        out = regions[0]
        for other in regions[1:]:
            if name == "conjunction":
                out, gap = self.call("conjunction", what, out, other)
                self.add_gap(gap)
            else:
                out = self.call(name, what, out, other)
        return out

    def some(self, prop, filler):
        iri = _iri(prop.asOWLObjectProperty())
        if iri not in self.properties:
            raise KeyError(f"Object property {iri} is not in the property map.")
        return self.call("exists", f"an existential restriction on {iri}",
                         self.ids(self.properties[iri]), filler)

    def evaluate(self, e):
        t = e.getClassExpressionType()
        if t == _T.OWL_CLASS:
            return self.named(_iri(e.asOWLClass()))
        if t == _T.OBJECT_INTERSECTION_OF:
            return self.fold("conjunction", "an intersection", e.getOperandsAsList())
        if t == _T.OBJECT_UNION_OF:
            return self.fold("disjunction", "a union", e.getOperandsAsList())
        if t == _T.OBJECT_COMPLEMENT_OF:
            return self.call("negation", "a complement", self.evaluate(e.getOperand()))
        if t == _T.OBJECT_SOME_VALUES_FROM:
            return self.some(e.getProperty(), self.evaluate(e.getFiller()))
        if t == _T.OBJECT_ALL_VALUES_FROM:
            inner = self.call("negation", "a universal restriction (as not exists not)",
                              self.evaluate(e.getFiller()))
            return self.call("negation", "a universal restriction (as not exists not)",
                             self.some(e.getProperty(), inner))
        if t == _T.OBJECT_ONE_OF:
            inds = list(e.getOperandsAsList()) if hasattr(e, "getOperandsAsList") \
                else list(e.getIndividuals())
            out = self.individual(inds[0])
            for other in inds[1:]:
                out = self.call("disjunction", "a nominal with several individuals", out,
                                self.individual(other))
            return out
        if t == _T.OBJECT_HAS_VALUE:
            return self.some(e.getProperty(), self.individual(e.getFiller()))
        raise ConstructorUnavailable(f"The class expression type {t} is not supported (only "
                                     f"named classes, nominals, ⊓, ⊔, ¬, ∃, ∀ and hasValue).")


def _evaluator(geometry, class_to_id, object_property_to_id, individual_to_id):
    return _Evaluator(geometry, class_to_id, object_property_to_id, individual_to_id)


class _Named:
    """A class given by IRI behaves as the expression of that class."""

    def __init__(self, iri):
        self.iri = iri

    def getClassExpressionType(self):
        return _T.OWL_CLASS

    def asOWLClass(self):
        class _C:
            def __init__(self, iri):
                self.iri = iri

            def toStringID(self):
                return self.iri
        return _C(self.iri)


def _as_expression(expression):
    return _Named(expression) if isinstance(expression, str) else expression


def expression_region(geometry, expression, class_to_id, object_property_to_id,
                      individual_to_id=None, with_gap=False):
    """The region of a class expression in a geometry.

    :param geometry: Any :class:`~mowl.representation.Geometry` (or a model/module that
        :func:`~mowl.representation.geometry_of` understands).
    :param expression: An OWL API class expression, or a class IRI.
    :param class_to_id, object_property_to_id, individual_to_id: The id maps (IRI string to
        row index); the last is needed only for nominals and ``ObjectHasValue``.
    :param with_gap: Also return the sum of the gaps of the conjunctions inside the
        expression (zero iff they are all non-empty), ``(1, 1)``, or ``None`` if there is no
        conjunction.
    :returns: The region, with a leading batch dimension of 1 (and the gap).
    :raises ConstructorUnavailable: If the geometry lacks a constructor the expression uses.
    """
    from mowl.representation.adapters import geometry_of
    geometry = geometry_of(geometry)
    ev = _evaluator(geometry, class_to_id, object_property_to_id, individual_to_id)
    region = ev.evaluate(_as_expression(expression))
    return (region, ev.gap) if with_gap else region


def pattern_region(geometry, expression, x, start=0, with_gap=False):
    """The regions of a class expression *pattern* for a batch of rows: the form in which the
    ALC datasets (:class:`~mowl.datasets.alc.ALCDataset`) give their axioms. The expression
    is a pattern (its named classes and properties are placeholders), and the row ``x[b]``
    holds the ids that fill it, in the order in which the names occur: a named class takes
    one column, :math:`\\exists r.C` and :math:`\\forall r.C` the column of ``r`` followed by
    those of ``C``, :math:`\\sqcap`, :math:`\\sqcup` the columns of their operands, :math:`\\neg`
    those of its operand.

    It uses the same constructors as :func:`expression_region` (:math:`\\forall r.C` as
    :math:`\\neg\\exists r.\\neg C`), only for ``B`` rows at once.

    :param geometry: Any :class:`~mowl.representation.Geometry`.
    :param expression: The pattern (an OWL API class expression).
    :param x: ``LongTensor (B, k)``.
    :param start: The column where the pattern starts.
    :param with_gap: Also return the sum of the gaps of the conjunctions of the pattern, or
        ``None`` if it has none.
    :returns: ``(region, next column)``, and the gap if ``with_gap``.
    :raises ConstructorUnavailable: If the geometry lacks a constructor the pattern uses.
    """
    from mowl.representation.adapters import geometry_of
    geometry = geometry_of(geometry)
    gaps = []

    def call(name, what, *args):
        try:
            return getattr(geometry, name)(*args)
        except ConstructorUnavailable as error:
            raise ConstructorUnavailable(
                f"{error} A class expression with {what} cannot be evaluated by "
                f"{type(geometry).__name__}.") from None

    def walk(e, column):
        t = e.getClassExpressionType()
        if t == _T.OWL_CLASS:
            return geometry.classes(x[:, column]), column + 1
        if t in (_T.OBJECT_SOME_VALUES_FROM, _T.OBJECT_ALL_VALUES_FROM):
            relation = x[:, column]
            filler, column = walk(e.getFiller(), column + 1)
            if t == _T.OBJECT_SOME_VALUES_FROM:
                return call("exists", "an existential restriction", relation, filler), column
            inner = call("negation", "a universal restriction (as not exists not)", filler)
            some = call("exists", "a universal restriction (as not exists not)", relation, inner)
            return call("negation", "a universal restriction (as not exists not)", some), column
        if t in (_T.OBJECT_INTERSECTION_OF, _T.OBJECT_UNION_OF):
            operands = e.getOperandsAsList()
            out, column = walk(operands[0], column)
            for operand in operands[1:]:
                other, column = walk(operand, column)
                if t == _T.OBJECT_INTERSECTION_OF:
                    out, gap = call("conjunction", "an intersection", out, other)
                    gaps.append(gap)
                else:
                    out = call("disjunction", "a union", out, other)
            return out, column
        if t == _T.OBJECT_COMPLEMENT_OF:
            region, column = walk(e.getOperand(), column)
            return call("negation", "a complement", region), column
        raise ConstructorUnavailable(f"The class expression type {t} is not supported "
                                     f"(only named classes, ⊓, ⊔, ¬, ∃ and ∀).")

    region, column = walk(expression, start)
    if not with_gap:
        return region, column
    return region, column, (sum(gaps[1:], gaps[0]) if gaps else None)


def _expand(region, n):
    """The region repeated ``n`` times along the batch dimension."""
    if not dataclasses.is_dataclass(region):
        raise TypeError(f"Cannot repeat a region of type {type(region).__name__}.")
    values = {}
    for f in dataclasses.fields(region):
        v = getattr(region, f.name)
        values[f.name] = v.expand(n, *v.shape[1:]) if isinstance(v, th.Tensor) \
            and v.dim() > 0 else v
    return type(region)(**values)


def _scalar(cost):
    return float(cost.reshape(-1)[0])


@th.no_grad()
def subsumption_cost(geometry, sub, sup, class_to_id, object_property_to_id,
                     individual_to_id=None, require_nonempty=False):
    """The cost of :math:`C \\sqsubseteq D` for two class expressions (lower: more true;
    zero when the geometry's measure holds exactly).

    :param require_nonempty: Add the gap of the conjunctions inside ``sub``, as the losses
        of the normal form :math:`C \\sqcap D \\sqsubseteq E` do: an empty intersection is then
        not a subclass of everything. Off by default (an empty class is a subclass of every
        class; ``score_subsumption`` of E2R does not add it either).
    :rtype: float
    """
    from mowl.representation.adapters import geometry_of
    geometry = geometry_of(geometry)
    maps = (class_to_id, object_property_to_id, individual_to_id)
    a, gap = expression_region(geometry, sub, *maps, with_gap=True)
    b = expression_region(geometry, sup, *maps)
    cost = _scalar(geometry.inside(a, b))
    if require_nonempty and gap is not None:
        cost += _scalar(gap)
    return cost


def _ranked(names, costs, top_k):
    order = sorted(range(len(names)), key=lambda i: (costs[i], i))
    if top_k is not None:
        order = order[:top_k]
    return [(names[i], costs[i]) for i in order]


@th.no_grad()
def rank_classes(geometry, expression, class_to_id, object_property_to_id,
                 individual_to_id=None, top_k=None):
    """All named classes :math:`A`, best first, by the cost of :math:`A \\sqsubseteq C`.

    :returns: List of ``(class IRI, cost)``.
    """
    from mowl.representation.adapters import geometry_of
    geometry = geometry_of(geometry)
    target = expression_region(geometry, expression, class_to_id, object_property_to_id,
                               individual_to_id)
    names = sorted(class_to_id, key=class_to_id.get)
    ids = th.tensor([class_to_id[n] for n in names], dtype=th.long, device=_device(geometry))
    n = len(names)
    cost = geometry.inside(geometry.classes(ids), _expand(target, n)).reshape(-1)
    return _ranked(names, cost.tolist(), top_k)


@th.no_grad()
def rank_individuals(geometry, expression, class_to_id, object_property_to_id,
                     individual_to_id, top_k=None):
    """All individuals, best first, by the cost of being a member of the class expression
    (:meth:`~mowl.representation.Geometry.member`).

    :returns: List of ``(individual IRI, cost)``.
    """
    from mowl.representation.adapters import geometry_of
    geometry = geometry_of(geometry)
    if not individual_to_id:
        raise ValueError("The dataset has no individuals to rank.")
    target = expression_region(geometry, expression, class_to_id, object_property_to_id,
                               individual_to_id)
    names = sorted(individual_to_id, key=individual_to_id.get)
    ids = th.tensor([individual_to_id[n] for n in names], dtype=th.long,
                    device=_device(geometry))
    cost = geometry.member(ids, _expand(target, len(names))).reshape(-1)
    return _ranked(names, cost.tolist(), top_k)
