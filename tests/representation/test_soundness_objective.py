"""Does the default objective read every axiom form soundly, for every combination of parts?

1. Every combination of a class model, a relation model and a measure that ``compose``
   accepts reads every axiom form, positive and negative, as a cost of shape ``(B, 1)`` that
   is never negative, or refuses it with :class:`ConstructorUnavailable` (a negative loss of
   a role axiom, which no objective has, is the documented ``NotImplementedError``). It
   never fails inside a formula with another error.
2. For the combinations whose regions are sets of points and whose relation model states
   its pairs, a zero loss of ``Objective()`` (strict margin) implies the axiom in the sampled
   interpretation: the interpretation of a class is the set of sampled points in its region,
   that of a relation the set of pairs ``related`` says. The geometry is first trained on the
   axioms so that zero losses occur.
"""

import itertools

import pytest
import torch as th

import mowl.representation as R
from mowl.representation import (Ball, Box, ConstructorUnavailable, FORMS, Objective,
                                 ROLE_FORMS, compose)
from mowl.representation.library.ballfam_parts import ToleranceTranslation
from mowl.representation.library.boxrel_relations import AffineBoxes, TranslationBoxes

NC, NR, NI, DIM = 6, 3, 4, 4
COLUMNS = {"gci0": "cc", "gci1": "ccc", "gci2": "crc", "gci3": "rcc", "gci0_bot": "c",
           "gci1_bot": "cc", "gci3_bot": "rc", "class_assertion": "ic",
           "object_property_assertion": "iri", "role_inclusion": "rr", "role_chain": "rrr"}


def _rows(form, n, seed):
    g = th.Generator().manual_seed(seed)
    size = {"c": NC, "r": NR, "i": NI}
    return th.stack([th.randint(0, size[k], (n,), generator=g) for k in COLUMNS[form]], 1)


def _combinations():
    for c, r, m in itertools.product(R.CLASS_MODELS, list(R.RELATION_MODELS) + [None],
                                     [None] + list(R.MEASURES)):
        yield c, r, m


def test_every_accepted_combination_reads_or_refuses_every_form():
    th.manual_seed(0)
    accepted, bad = 0, []
    for c, r, m in _combinations():
        try:
            g = compose(NC, NR, NI, DIM, classes=c, relations=r, measure=m)
        except TypeError:
            continue           # refused when built: the measure does not compare the regions
        accepted += 1
        for form in FORMS + ROLE_FORMS:
            for neg in (False, True):
                rows = _rows(form, 3, 1)
                try:
                    cost = g.cost(form, rows, neg)
                except ConstructorUnavailable:
                    continue
                except NotImplementedError as error:
                    if "no negative loss" in str(error):
                        continue
                    bad.append((c, r, m, form, neg, repr(error)[:80]))
                    continue
                except Exception as error:      # noqa: BLE001 - the point of the test
                    bad.append((c, r, m, form, neg, repr(error)[:80]))
                    continue
                if cost.shape != (3, 1) or bool(th.isnan(cost).any()) \
                        or bool((cost < 0).any()):
                    bad.append((c, r, m, form, neg, tuple(cost.shape)))
    assert accepted > 1000
    assert not bad, bad[:10]


# -- semantics --------------------------------------------------------------------------------

def _points(n, seed, scale=3.0):
    g = th.Generator().manual_seed(seed)
    return (th.rand(n, DIM, generator=g) * 2 - 1) * scale


def _rows_of(region, i):
    if isinstance(region, Ball):
        return Ball(region.centre[i:i + 1], region.radius[i:i + 1])
    return Box(region.centre[i:i + 1], region.offset[i:i + 1])


def has_successor(relations, rid, x, region, family):
    """``(P,)``: whether each point of ``x`` has an r-successor in the region (one row)."""
    ids = th.tensor([rid])
    if isinstance(relations, ToleranceTranslation):
        centre = relations.successor(ids, x)[0]
        sigma = relations.tolerance(ids)[0]
        if family is Box:
            lo = th.maximum(centre - sigma, region.lower)
            hi = th.minimum(centre + sigma, region.upper)
            return (lo <= hi).all(dim=-1)
        return th.linalg.norm(centre - region.centre, dim=-1) <= region.radius[0] + sigma
    if relations.functional:
        return region.contains(relations.successor(ids, x)[0])[0]
    if isinstance(relations, TranslationBoxes):
        box = relations.box(ids)
        if family is Ball:
            reach = th.linalg.norm(box.offset)
            return th.linalg.norm(x + box.centre - region.centre, dim=-1) \
                <= region.radius[0] + reach
        lo = th.maximum(x + box.centre - box.offset, region.lower)
        hi = th.minimum(x + box.centre + box.offset, region.upper)
        return (lo <= hi).all(dim=-1)
    if isinstance(relations, AffineBoxes):
        m, c, e = relations.slope(ids), relations.vectors(ids), relations.error(ids)
        rng = relations.range(ids)
        lo = th.maximum(th.maximum((x - c - e) / m, rng.lower), region.lower)
        hi = th.minimum(th.minimum((x - c + e) / m, rng.upper), region.upper)
        return (lo <= hi).all(dim=-1)
    raise NotImplementedError(type(relations).__name__)


def _sample_pairs(relations, rid, n, seed, family):
    """Pairs ``(x, y)`` of the relation ``rid``."""
    ids = th.tensor([rid])
    x = _points(n, seed, 1.0)
    g = th.Generator().manual_seed(seed + 1)
    u = th.rand(n, DIM, generator=g) * 2 - 1
    if isinstance(relations, ToleranceTranslation):
        sigma = relations.tolerance(ids)[0]
        step = u * sigma if family is Box else u / DIM ** 0.5 * sigma
        return x, relations.successor(ids, x)[0] + step
    if relations.functional:
        return x, relations.successor(ids, x)[0]
    if isinstance(relations, TranslationBoxes):
        box = relations.box(ids)
        if family is Ball:
            return x, x + box.centre + u / DIM ** 0.5 * th.linalg.norm(box.offset)
        return x, x + box.centre + u * box.offset
    if isinstance(relations, AffineBoxes):
        rng = relations.range(ids)
        y = rng.centre + u * rng.offset
        e = relations.error(ids)
        v = th.rand(n, DIM, generator=g) * 2 - 1
        return y * relations.slope(ids) + relations.vectors(ids) + v * e, y
    raise NotImplementedError


def holds(g, form, row, family):
    """Whether the axiom ``row`` of ``form`` holds in the sampled interpretation of ``g``."""
    cls = g.classes
    rel = g.relation_model
    pts = _points(3000, int(row.sum()) + 7)

    def region(i):
        return cls(th.tensor([int(i)]))

    def members(r):
        return r.contains(pts)[0]
    if form == "gci0":
        return not bool((members(region(row[0])) & ~members(region(row[1]))).any())
    if form == "gci1":
        both = members(region(row[0])) & members(region(row[1]))
        return not bool((both & ~members(region(row[2]))).any())
    if form == "gci2":
        x = pts[members(region(row[0]))]
        return bool(has_successor(rel, int(row[1]), x, region(row[2]), family).all())
    if form == "gci3":
        succ = has_successor(rel, int(row[0]), pts, region(row[1]), family)
        return not bool((succ & ~members(region(row[2]))).any())
    if form == "gci0_bot":
        return not bool(members(region(row[0])).any())
    if form == "gci1_bot":
        return not bool((members(region(row[0])) & members(region(row[1]))).any())
    if form == "gci3_bot":
        return not bool(has_successor(rel, int(row[0]), pts, region(row[1]), family).any())
    if form == "role_inclusion":
        x, y = _sample_pairs(rel, int(row[0]), 500, 3, family)
        return bool(_related(rel, int(row[1]), x, y, family).all())
    if form == "role_chain":
        x, y = _sample_pairs(rel, int(row[0]), 500, 3, family)
        _, z = _sample_pairs(rel, int(row[1]), 500, 3, family)
        z = y + (z - _points(500, 3, 1.0)) if not isinstance(rel, AffineBoxes) else None
        if z is None:
            return True     # (pairs of a chain of range boxes are not sampled here)
        return bool(_related(rel, int(row[2]), x, z, family).all())
    raise KeyError(form)


def _related(rel, rid, x, y, family):
    ids = th.tensor([rid])
    if isinstance(rel, (ToleranceTranslation, TranslationBoxes)):
        return rel.related(ids, x, y, region=family)[0]
    if rel.functional:
        return rel.related(ids, x, y, tolerance=1e-4)[0]
    return rel.related(ids, x, y)[0]


SEMANTIC = [("ball", "translation"), ("box", "translation"),
            ("ball", "translation with tolerance"), ("box", "translation with tolerance"),
            ("box", "box of translations"), ("ball", "box of translations"),
            ("box", "affine map"), ("box", "affine map with range and error box")]
FORMS_CHECKED = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot")


@pytest.mark.parametrize("classes, relations", SEMANTIC)
def test_zero_loss_with_a_strict_margin_implies_the_axiom(classes, relations):
    th.manual_seed(0)
    g = compose(NC, NR, 0, DIM, classes=classes, relations=relations,
                objective=Objective(margin=0.05))
    family = g.class_model.region
    data = {form: _rows(form, 12, i) for i, form in enumerate(FORMS_CHECKED)}
    if g.role_axiom_capable:
        data.update({form: _rows(form, 6, 20 + i) for i, form in enumerate(ROLE_FORMS)})
    # train on each form separately (the forms contradict each other), so that zero
    # losses occur; then every row with zero loss must hold
    checked = 0
    for form, rows in data.items():
        model = compose(NC, NR, 0, DIM, classes=classes, relations=relations,
                        objective=Objective(margin=0.05))
        model.load_state_dict(g.state_dict())
        optimiser = th.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(150):
            optimiser.zero_grad()
            loss = model.cost(form, rows).sum()
            if float(loss) == 0:
                break
            loss.backward()
            optimiser.step()
        with th.no_grad():
            cost = model.cost(form, rows).reshape(-1)
            for row, c in zip(rows, cost):
                if float(c) == 0:
                    checked += 1
                    assert holds(model, form, row, family), (classes, relations, form, row)
    assert checked > 0


def _violations(g, form, rows, family, steps=150):
    optimiser = th.optim.Adam(g.parameters(), lr=0.05)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = g.cost(form, rows).sum()
        if float(loss) == 0:
            break
        loss.backward()
        optimiser.step()
    with th.no_grad():
        cost = g.cost(form, rows).reshape(-1)
    zero = [row for row, c in zip(rows, cost) if float(c) == 0]
    return zero, [row for row in zero if not holds(g, form, row, family)]


def test_the_published_readings_are_caught():
    """The check above finds the published readings that are not sound: the overlap
    reading of ``∃r.C ⊑ D`` (ELEmbeddings, ELBE) and the conjunction loss of ELEmbeddings."""
    from mowl.representation.measures import CentreDistance
    th.manual_seed(0)
    g = compose(NC, NR, 0, DIM, classes="ball", relations="translation",
                measure=CentreDistance(conjunction="published"),
                objective=Objective(margin=0.05, gci3="overlap"))
    found = 0
    for form in ("gci1", "gci3"):
        zero, wrong = _violations(g, form, _rows(form, 12, 3), Ball)
        found += len(wrong)
    assert found > 0
