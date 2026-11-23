"""Swap one part of a named method.

``Model(dataset, method="elbe", relations="affine map")`` builds ELBE with another relation
model: this module is the one place that does it for every general model
(:class:`~mowl.models.ComposedEL`, :class:`~mowl.models.ComposedALC`).

The convention, the same for every method:

* A part is given by *name* (a key of the registry of its kind:
  :data:`~mowl.representation.CLASS_MODELS`, :data:`~mowl.representation.RELATION_MODELS`,
  :data:`~mowl.representation.MEASURES`), by a *factory* (a function ``(size, embed_dim)``
  for class and relation models, a class for a measure) or by an *instance*, which is taken
  as it is. ``relations="none"`` removes the relation model.
* ``objective`` is an :class:`~mowl.representation.Objective`; ``objective_options`` are
  options of the method's own objective (``{"gci3": "overlap"}``, see
  :meth:`Objective.replace <mowl.representation.Objective.replace>`).
* ``regularizers`` are added to those of the method. A regulariser of the method on a table
  of a part that is replaced (TransBox's penalty on ``relation_model.offsets``) goes with the
  part when the new part has no such table; the geometry lists it in
  ``replaced_regularizers``, with the reason.
* The method is built first, then the part is replaced in its geometry
  (:meth:`ComposedGeometry.replace_parts <mowl.representation.ComposedGeometry.replace_parts>`),
  whatever the builder does with a part argument of its own (``boxlite(measure=...)`` takes an
  instance, ``boxe(classes=...)`` a name; the general models use this one convention).
* A combination that cannot exist (balls under a measure of boxes, a relation model that
  does not map the regions of the class model) raises
  :class:`~mowl.representation.ConstructorUnavailable` naming both sides. An argument the
  method cannot take raises :class:`TypeError` naming the accepted ones. Nothing is ignored.
"""

import inspect

import torch.nn as nn

from mowl.representation import methods as M
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.objectives import Objective
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["build_method", "accepted_parts", "current_parts", "PART_KINDS", "NO_PART"]

#: The part kinds of a method that is a :class:`~mowl.representation.ComposedGeometry`.
PART_KINDS = ("classes", "relations", "individuals", "measure", "objective",
              "objective_options", "regularizers")

#: ``relations=NO_PART`` removes the relation model.
NO_PART = "none"


def accepted_parts(key):
    """The part arguments of :func:`build_method` for the method ``key`` (those of a
    composed geometry; a method whose geometry is not one takes none)."""
    build = M.METHODS[key]
    kind = M.METHOD_KINDS[key]
    return tuple(sorted(PART_KINDS)) if kind in ("axioms", "expressions") else ()


def current_parts(geometry):
    """``{kind: part}`` of a composed geometry: what the swap tests compare with."""
    return {"classes": geometry.class_model, "relations": geometry.relation_model,
            "individuals": geometry.individual_model, "measure": geometry.measure,
            "objective": geometry.objective, "regularizers": geometry.regularizers}


def _resolve_model(value, registry, kind, size, embed_dim):
    if isinstance(value, str):
        if value not in registry:
            raise ValueError(f"Unknown {kind} {value!r}; available: {sorted(registry)}.")
        value = registry[value]
    if isinstance(value, nn.Module):
        return value
    if callable(value):
        try:
            return value(size, embed_dim)
        except (ValueError, TypeError) as error:
            raise ConstructorUnavailable(
                f"The {kind} {getattr(value, '__name__', value)} cannot be built with "
                f"{size} rows of dimension {embed_dim}: {error}") from error
    raise TypeError(f"A {kind} is a name, a function (size, embed_dim) or a built part, "
                    f"not {value!r}.")


def _guarded(factory, kind):
    def build(size, embed_dim):
        return _resolve_model(factory, {}, kind, size, embed_dim)
    return build


def _resolve_measure(value):
    if isinstance(value, str):
        if value not in M.MEASURES:
            raise ValueError(f"Unknown measure {value!r}; available: {sorted(M.MEASURES)}.")
        value = M.MEASURES[value]
    if inspect.isclass(value) or (callable(value) and not hasattr(value, "accepts")):
        value = value()
    return value


def forms_reading_relations(geometry, rows_by_form, limit=8):
    """The forms among ``rows_by_form`` (``{form: rows}``, the training rows) whose loss is
    computed from the relation model of ``geometry``: the autograd graph of the loss of the
    form reaches a parameter of the relation model. Empty if no form reads one."""
    import torch as th
    parameters = [p for p in geometry.relation_model.parameters() if p.requires_grad]
    read = []
    for form, rows in rows_by_form.items():
        if len(rows) == 0 or not parameters:
            continue
        for neg in (False, True) if form in getattr(geometry, "neg_capable_gcis", ()) \
                else (False,):
            cost = geometry(rows[:limit], form, neg=True) if neg else geometry(rows[:limit], form)
            if not isinstance(cost, th.Tensor) or not cost.requires_grad:
                continue
            grads = th.autograd.grad(cost.sum(), parameters, allow_unused=True)
            if any(g is not None for g in grads):
                read.append(form)
                break
    return read


def check_relations_are_read(geometry, key, rows_by_form):
    """Refuse a swapped relation model that no training form reads: its parameters would
    never move and the losses would be those of the method without it."""
    if geometry.relation_model is None or not rows_by_form:
        return
    if not list(geometry.relation_model.parameters()):
        return
    if not forms_reading_relations(geometry, rows_by_form):
        raise ConstructorUnavailable(
            f"none of the forms {key!r} is trained on "
            f"({sorted(f for f, r in rows_by_form.items() if len(r))}) reads one in this "
            f"combination of parts {geometry.parts}, so the loss would not change and the "
            f"relation parameters would never move.")


def build_method(key, nb_classes, nb_relations, nb_individuals=0, embed_dim=50,
                 classes=None, relations=None, individuals=None, measure=None,
                 objective=None, objective_options=None, regularizers=(), rows=None,
                 **options):
    """The method ``key`` with the given parts replaced (see the module).

    :param rows: ``{form: rows}`` of the training data. With it, a replaced relation model
        that no form reads is refused (:func:`check_relations_are_read`).

    :param options: Options of the builder of the method.
    :returns: The geometry.
    :raises TypeError: A part the method cannot take.
    :raises ConstructorUnavailable: A replacement that cannot be combined with the rest.
    """
    if key not in M.METHODS:
        raise ValueError(f"Unknown method {key!r}; available: {sorted(M.METHODS)}.")
    build = M.METHODS[key]
    parameters = inspect.signature(build).parameters
    given = {"classes": classes, "relations": relations, "individuals": individuals,
             "measure": measure, "objective": objective,
             "objective_options": objective_options or None,
             "regularizers": tuple(regularizers) or None}
    given = {k: v for k, v in given.items() if v is not None}
    rest = given
    sizes = (nb_classes, nb_relations, nb_individuals)
    geometry = build(*sizes, embed_dim=embed_dim, **options)
    if not rest:
        return geometry
    if not isinstance(geometry, ComposedGeometry):
        raise ConstructorUnavailable(
            f"The method {key!r} builds a {type(geometry).__name__}, which is not put "
            f"together from replaceable class, relation and measure parts; it does not take "
            f"{sorted(rest)}. Its own options: "
            f"{[p for p in parameters if p not in ('tables', 'nb_classes', 'nb_relations', 'nb_individuals', 'embed_dim')]}."
            + (f" Its parts are swapped through {geometry.trained_by}."
               if getattr(geometry, "trained_by", None) else ""))
    geometry = swap(geometry, key, sizes, embed_dim, **rest)
    if relations is not None and rows is not None:
        try:
            check_relations_are_read(geometry, key, rows)
        except ConstructorUnavailable as error:
            raise ConstructorUnavailable(f"{key!r} cannot take ['relations']: {error}") \
                from error
    return geometry


def _kept_regularizers(geometry, parts, replaced):
    """The regularisers of ``geometry`` that still have their tables once the parts
    ``replaced`` (``"class_model"``, ...) are those of ``parts``. A regulariser of the method
    that names a table of a replaced part (``"relation_model.offsets"``) belongs to that part
    and goes with it when the new part has no such table; the geometry records it in
    ``replaced_regularizers`` with the reason."""
    new = {"class_model": parts.get("classes"), "relation_model": parts.get("relations"),
           "individual_model": parts.get("individuals")}
    dropped = []
    for regularizer in geometry.regularizers:
        names = getattr(regularizer, "parameters", None)
        names = (names,) if isinstance(names, str) else (
            tuple(names) if isinstance(names, (tuple, list)) else ())
        missing = []
        for name in names:
            owner, _, rest = name.partition(".")
            if owner in replaced and rest:
                target = new[owner]
                try:
                    for attribute in rest.split("."):
                        target = getattr(target, attribute)
                except AttributeError:
                    missing.append(name)
        if missing:
            dropped.append((regularizer, f"it regularises {missing}, tables of the replaced "
                                         f"part, which the new part does not have"))
            continue
        yield regularizer
    if dropped:
        geometry.replaced_regularizers = tuple(dropped)


def swap(geometry, key, sizes, embed_dim, classes=None, relations=None, individuals=None,
         measure=None, objective=None, objective_options=None, regularizers=None):
    """Replace parts of ``geometry`` (see :func:`build_method`)."""
    nb_classes, nb_relations, nb_individuals = sizes
    parts = {}
    if classes is not None:
        parts["classes"] = _resolve_model(classes, M.CLASS_MODELS, "class model",
                                          nb_classes, embed_dim)
    if relations is not None:
        parts["relations"] = False if (isinstance(relations, str) and relations == NO_PART) \
            else _resolve_model(relations, M.RELATION_MODELS, "relation model",
                                nb_relations, embed_dim)
    if individuals is not None:
        parts["individuals"] = _resolve_model(individuals, M.CLASS_MODELS, "individual model",
                                              nb_individuals, embed_dim)
    elif classes is not None and geometry.individual_model is not None and nb_individuals:
        # individuals are regions of the kind of the classes (as in ``compose``)
        factory = classes if not isinstance(classes, nn.Module) else None
        if isinstance(factory, str):
            factory = M.CLASS_MODELS[factory]
        if factory is not None:
            # a class model may say which individual model goes with it (subspaces: vectors)
            factory = _guarded(getattr(factory, "individual_factory", None) or factory,
                               'individual model')
        if factory is None:
            raise ConstructorUnavailable(
                f"The class model {type(classes).__name__} was given as an instance, so the "
                f"individual model of the method ({type(geometry.individual_model).__name__}) "
                f"cannot be rebuilt in its kind: give individuals= too.")
        parts["individuals"] = factory(nb_individuals, embed_dim)
    if measure is not None:
        parts["measure"] = _resolve_measure(measure)
    if objective is not None or objective_options:
        base = objective if objective is not None else geometry.objective
        if objective_options:
            unread = base.unread_options(geometry.supported_forms)
            measure_unread = getattr(parts.get("measure", geometry.measure),
                                     "unread_objective_options", None)
            if measure_unread is not None:
                unread = {**unread, **measure_unread()}
            current = base.options
            for name in sorted(objective_options):
                if name in unread and objective_options[name] != current.get(name):
                    raise ConstructorUnavailable(
                        f"{key!r} cannot take the objective option {name}="
                        f"{objective_options[name]!r}: {unread[name]}, so the losses would "
                        f"not change.")
        parts["objective"] = base.replace(**objective_options) if objective_options else base
    replaced = [name for kind, name in (("classes", "class_model"),
                                        ("relations", "relation_model"),
                                        ("individuals", "individual_model")) if kind in parts]
    if replaced:
        kept = tuple(_kept_regularizers(geometry, parts, replaced))
        if len(kept) != len(geometry.regularizers):
            parts["regularizers"] = kept + tuple(parts.get("regularizers", ()))
    if regularizers is not None:
        parts["regularizers"] = tuple(parts.get("regularizers", geometry.regularizers)) \
            + tuple(regularizers)
    try:
        geometry.replace_parts(**parts)
    except ConstructorUnavailable as error:
        raise ConstructorUnavailable(
            f"{key!r} cannot take {sorted(k for k in parts if k != 'regularizers')}: "
            f"{error} (the method's other parts: {geometry.parts}).") from error
    new = parts.get("objective")
    if objective is not None and getattr(new, "strict_negatives", False):
        # an objective that refuses the negative loss of a form it has none for: the forms
        # with a negative loss are the ones it defines (a ranking form has one)
        geometry.neg_capable_gcis = frozenset(new.neg_forms) | frozenset(new.ranking or ())
    refuses = getattr(geometry, "refuses", None)
    reason = refuses() if callable(refuses) else None
    if reason:
        # the geometry knows why its parts cannot work together (an open domain of elements
        # that only its own class and relation models evaluate on)
        raise ConstructorUnavailable(
            f"{key!r} cannot take {sorted(k for k in parts if k != 'regularizers')}: "
            f"{reason} (the method's other parts: {geometry.parts}).")
    return geometry
