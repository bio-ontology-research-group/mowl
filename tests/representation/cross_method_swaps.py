"""Cross-method swaps among the region-based EL methods.

For every ordered pair of methods ``(A, B)`` of :data:`REGION_METHODS` and every part kind of
:data:`KINDS`, method ``A`` is built through :class:`~mowl.models.ComposedEL` with the part of
``B`` in place of its own (by registry name where the registry has the part's class, else as
the built part of ``B``; the objective of ``B`` as it is). A donor part of the same class as
``A``'s own is not a cross-method swap and is skipped; donors that give the same replacement
are tried once. Each case ends in one of

``works``
    accepted, the part is in the model, the losses (or the parameters that receive gradient)
    differ from those of ``A``, and two epochs train to finite parameters and costs;
``refused``
    :class:`~mowl.representation.ConstructorUnavailable` at build time (the reason is kept);
``same losses``
    an objective or measure of another method that is accepted and read, but gives every
    form the data has the costs of the method's own (BoxSEL's objective in BoxEL, the
    hierarchy objectives of two methods with options that only forms absent from the data
    read, the signed distance of BoxLitE and the bounds of ELBE on boxes): not a
    cross-method swap on these data, counted apart;
``no effect``
    a class model, relation model or measure that is accepted but changes nothing (an
    ignored part: a failure);
``bug``
    any other exception, at build time or in training (a failure).

``python -m tests.representation.cross_method_swaps [out.tsv]`` runs the sweep and prints
the counts; :func:`summary_rst` is the table of the documentation.
"""
import copy
import sys
import time
import traceback

import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.models import ComposedEL  # noqa: E402
from mowl.representation import (CLASS_MODELS, MEASURES, RELATION_MODELS,  # noqa: E402
                                 ConstructorUnavailable)

#: The region-based EL methods: their classes are regions of a space (boxes, balls, cones,
#: octagons, prisms, unions of boxes, ellipsoids) and ComposedEL trains them on normal-form
#: axioms. Left out: text encoders (``hit``, ``ont``), element sets (``canonical_mixture``),
#: triple models (``boxe``, ``expressive``, ``graph_kge``, ``gda_projection``), and the methods
#: no general model trains (``grounded``, ``faithel``).
REGION_METHODS = (
    "elem", "emelpp", "emelvar", "regd_elem",                       # balls
    "elbe", "boxel", "boxsel", "box2el", "transbox", "elkem", "regd",  # boxes
    "box_lattice", "gumbel_box", "bcbox", "concept2box", "boxlite",  # boxes (volumes, solved)
    "octagon", "prism", "eike",                                     # other convex shapes
    "multiboxel", "multielbe", "mcmultibox",                        # unions of boxes
    "order_embeddings", "cate", "hyperbolic_cones", "shadow_cones",  # cones and orders
    "disk_embeddings")
KINDS = ("classes", "relations", "measure", "objective")
D = 6
WORKS, REFUSED, SAME, NO_EFFECT, BUG = "works", "refused", "same losses", "no effect", "bug"


def options(key):
    o = {"embed_dim": D, "margin": 0.05, "batch_size": 64}
    if key == "mcmultibox":
        o["margin"] = 0.0
    return o


_DATA = []


def dataset():
    if not _DATA:
        from mowl.datasets.builtin import FamilyDataset
        _DATA.append(FamilyDataset())
    return _DATA[0]


def build(key, **parts):
    th.manual_seed(0)
    return ComposedEL(dataset(), method=key, **options(key), **parts)


def _registry_name(part, registry, measure=False):
    """The registry name whose entry builds a part of the class of ``part`` (``None``)."""
    for name, entry in registry.items():
        try:
            made = entry if measure and isinstance(entry, type) else (
                entry() if measure else entry(3, D))
        except Exception:
            continue
        if (made if isinstance(made, type) else type(made)) is type(part):
            return name
    return None


def donors(kind, key, module):
    """``(label, {argument: value})`` of the part ``kind`` of the method ``key``."""
    if kind == "objective":
        objective = module.objective
        return (f"{key}'s objective {type(objective).__name__}",
                {"objective": copy.deepcopy(objective)})
    part = {"classes": module.class_model, "relations": module.relation_model,
            "measure": module.measure}[kind]
    if part is None:
        return None
    registry = {"classes": CLASS_MODELS, "relations": RELATION_MODELS,
                "measure": MEASURES}[kind]
    name = _registry_name(part, registry, measure=kind == "measure")
    if name is not None:
        return f"{name!r}", {kind: name}
    value = copy.deepcopy(part)
    if kind == "classes":
        individuals = module.individual_model
        argument = {"classes": value}
        if individuals is not None:
            argument["individuals"] = copy.deepcopy(individuals)
        return f"{key}'s {type(part).__name__}", argument
    return f"{key}'s {type(part).__name__}", {kind: value}


def _same(kind, base, argument):
    module = base.module
    if kind == "objective":
        theirs = argument["objective"]
        mine = module.objective
        return type(theirs) is type(mine) and theirs.options == mine.options
    value = argument[kind]
    own = {"classes": module.class_model, "relations": module.relation_model,
           "measure": module.measure}[kind]
    if isinstance(value, str):
        registry = {"classes": CLASS_MODELS, "relations": RELATION_MODELS,
                    "measure": MEASURES}[kind]
        return _registry_name(own, registry, measure=kind == "measure") == value \
            if own is not None else False
    return own is not None and type(value) is type(own)


def cases():
    """``[(receiver, kind, donor methods, label, argument)]``, one per distinct replacement."""
    built = {}
    for key in REGION_METHODS:
        built[key] = build(key)
    out = []
    for receiver in REGION_METHODS:
        base = built[receiver]
        for kind in KINDS:
            seen = {}
            for donor in REGION_METHODS:
                if donor == receiver:
                    continue
                given = donors(kind, donor, built[donor].module)
                if given is None:
                    continue
                label, argument = given
                if _same(kind, base, argument):
                    continue
                signature = label if kind != "objective" else (
                    type(argument["objective"]).__name__,
                    repr(sorted(argument["objective"].options.items(), key=str)))
                if signature in seen:
                    seen[signature][2].append(donor)
                    continue
                seen[signature] = [receiver, kind, [donor], label, argument]
            out.extend(tuple(v) for v in seen.values())
    return built, out


def run_case(base, receiver, kind, argument):
    """``(outcome, detail)`` of one case."""
    from tests.representation.test_user_level_swaps import has_effect, check_axioms
    try:
        model = build(receiver, **copy.deepcopy(argument))
    except ConstructorUnavailable as error:
        cause = error.__cause__
        while isinstance(cause, ConstructorUnavailable) and cause.__cause__ is not None:
            cause = cause.__cause__
        wrapped = "" if cause is None or isinstance(cause, ConstructorUnavailable) \
            else f"[wraps {type(cause).__name__}] "
        return REFUSED, wrapped + " ".join(str(error).split())
    except Exception as error:
        return BUG, f"build: {type(error).__name__}: {error} | " + _where()
    try:
        if not has_effect(base, model, kind):
            if kind in ("objective", "measure"):
                # read (the geometry calls it) and computes what the method's own part
                # computes on these forms: two formulas that agree here
                return SAME, f"the {kind} gives the costs of the method's own on the forms " \
                             f"of the data"
            return NO_EFFECT, "the losses and the parameters that receive gradient are unchanged"
        check_axioms(model)
    except Exception as error:
        return BUG, f"train: {type(error).__name__}: {error} | " + _where()
    return WORKS, ""


def _where():
    lines = [line.strip() for line in traceback.format_exc().splitlines()]
    return " / ".join(lines[-5:-1])[:400]


def sweep(out=None, receivers=None):
    built, todo = cases()
    results = []
    for receiver, kind, donor_keys, label, argument in todo:
        if receivers and receiver not in receivers:
            continue
        start = time.time()
        outcome, detail = run_case(built[receiver], receiver, kind, argument)
        results.append((receiver, kind, label, tuple(donor_keys), outcome, detail))
        if out is not None:
            print("\t".join([receiver, kind, label, ",".join(donor_keys), outcome,
                             f"{time.time() - start:.1f}s", detail[:600]]),
                  file=out, flush=True)
    return results


def counts(results):
    total = {}
    for receiver, kind, *_rest, outcome, _ in results:
        total.setdefault(kind, {}).setdefault(outcome, 0)
        total[kind][outcome] += 1
    return total


#: Why a case is refused, by the words of the refusal (first match).
CATEGORIES = (
    ("the measure compares another region family", ("compares [", "not a box",
                                                    "point-membership function")),
    ("no trained form reads a relation model", ("none of the forms",)),
    ("the relation model does not map these regions", ("maps [", "No relation model for",
                                                       "DL-Lite", "has no image",
                                                       "cannot be enlarged")),
    ("an operation the part does not provide", ("does not provide the operation",
                                                "has no volume", "cannot detach",
                                                "has no disjointness", "is not a Ball",
                                                "has no non-inclusion")),
    ("a margin the measure cannot read", ("margin",)),
    ("the method's training regime or solver needs its own part", ("PerColumnCorruption",
                                                                   "programme",
                                                                   "regularise")),
)


def category(detail):
    """The category of a refusal (:data:`CATEGORIES`), or ``"other"``."""
    for name, words in CATEGORIES:
        if any(word in detail for word in words):
            return name
    return "other"


def summary_rst(results):
    """The counts per part kind as a reStructuredText table."""
    by = counts(results)
    outcomes = (WORKS, REFUSED, SAME, NO_EFFECT, BUG)
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - part replaced"]
    lines += [f"     - {o}" for o in outcomes] + ["     - cases"]
    for kind in KINDS + ("all",):
        row = by.get(kind, {}) if kind != "all" else {
            o: sum(by.get(k, {}).get(o, 0) for k in KINDS) for o in outcomes}
        lines.append(f"   * - {kind}")
        lines += [f"     - {row.get(o, 0)}" for o in outcomes]
        lines.append(f"     - {sum(row.values())}")
    return "\n".join(lines) + "\n"


def refusals_rst(results):
    """The refusals by category and part kind, as a reStructuredText table."""
    table = {}
    for receiver, kind, label, donors_, outcome, detail in results:
        if outcome == REFUSED:
            row = table.setdefault(category(detail), {})
            row[kind] = row.get(kind, 0) + 1
    lines = [".. list-table::", "   :header-rows: 1", "", "   * - why refused"]
    lines += [f"     - {k}" for k in KINDS]
    for name in [c for c, _ in CATEGORIES] + ["other"]:
        if name in table:
            lines.append(f"   * - {name}")
            lines += [f"     - {table[name].get(k, 0)}" for k in KINDS]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import logging
    import warnings
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    receivers = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    with (open(path, "w") if path else sys.stdout) as out:
        results = sweep(out, receivers)
    for kind, row in counts(results).items():
        print(kind, row, file=sys.stderr)
