"""Closed-form octagon geometry checked against brute-force point sampling."""
import itertools

import pytest
import torch as th

import mowl
mowl.init_jvm("2g")
import mowl.datasets  # noqa: F401,E402  (import order: avoids a circular import in mowl.nn)
from mowl.representation.library import octagon_geometry as G  # noqa: E402

LIM = 3.0
STEP = 0.01


def _grid(lim=LIM, step=STEP):
    xs = th.arange(-lim, lim + step / 2, step, dtype=th.float64)
    gx, gy = th.meshgrid(xs, xs, indexing="ij")
    return th.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


GRID = _grid()


def _inside(points, h):
    """Membership written out with the explicit inequalities, independent of ``geometry``."""
    x, y = points[:, 0], points[:, 1]
    conds = [x <= h[0], x + y <= h[1], y <= h[2], y - x <= h[3],
             -x <= h[4], -x - y <= h[5], -y <= h[6], x - y <= h[7]]
    return th.stack(conds, dim=0).all(dim=0)


def _random_nonempty(gen, n):
    c = (th.rand(n, 2, generator=gen, dtype=th.float64) - 0.5) * 2.0
    r = th.rand(n, 8, generator=gen, dtype=th.float64) * 1.5
    return G.from_center_radius(c, r)


def _random_raw(gen, n):
    """Raw bounds, about half of them describe empty octagons."""
    return (th.rand(n, 8, generator=gen, dtype=th.float64) - 0.35) * 2.5


def _paper_prop1(h):
    """Proposition 1 of Charpenay & Schockaert (IJCAI 2024), transcribed in their notation."""
    xm, xp, ym, yp, um, up, vm, vp = G.to_paper_bounds(h).unbind(-1)

    def mx(*a):
        return th.stack(a, -1).amax(-1)

    def mn(*a):
        return th.stack(a, -1).amin(-1)

    xm1 = mx(xm, vm - yp, ym - up, 0.5 * (vm - up))
    xp1 = mn(xp, vp - ym, yp - um, 0.5 * (vp - um))
    ym1 = mx(ym, um + xm, vm - xp, 0.5 * (um + vm))
    yp1 = mn(yp, up + xp, vp - xm, 0.5 * (up + vp))
    um1 = mx(um, ym - xp, vm - 2 * xp, 2 * ym - vp)
    up1 = mn(up, yp - xm, vp - 2 * xm, 2 * yp - vm)
    vm1 = mx(vm, xm + ym, um + 2 * xm, 2 * ym - up)
    vp1 = mn(vp, xp + yp, up + 2 * xp, 2 * yp - um)
    return G.from_paper_bounds(th.stack([xm1, xp1, ym1, yp1, um1, up1, vm1, vp1], -1))


def test_three_tightening_rules_per_direction():
    for k in range(8):
        assert sum(1 for r in G.TIGHTENING_RULES if r[0] == k) == 3


def test_tightening_equals_paper_proposition_1():
    gen = th.Generator().manual_seed(0)
    h = th.cat([_random_nonempty(gen, 200), _random_raw(gen, 200)])
    assert th.allclose(G.tighten(h), _paper_prop1(h), atol=1e-12)


def test_paper_bounds_roundtrip():
    gen = th.Generator().manual_seed(1)
    h = _random_raw(gen, 50)
    assert th.allclose(G.from_paper_bounds(G.to_paper_bounds(h)), h)


def test_tightening_idempotent_and_preserves_point_set():
    gen = th.Generator().manual_seed(2)
    for h in _random_nonempty(gen, 30):
        t = G.tighten(h)
        assert th.allclose(G.tighten(t), t, atol=1e-12)
        assert th.all(t <= h + 1e-12)
        assert th.equal(_inside(GRID, h), _inside(GRID, t))


def test_tight_bounds_are_the_support_function():
    """Compare with a linear programme (scipy) and with grid sampling."""
    from scipy.optimize import linprog
    gen = th.Generator().manual_seed(3)
    d = G.DIRECTIONS.double()
    for h in _random_nonempty(gen, 30):
        t = G.tighten(h)
        for k in range(8):
            res = linprog(-d[k].numpy(), A_ub=d.numpy(), b_ub=h.numpy(),
                          bounds=[(None, None)] * 2)
            assert res.status == 0
            assert abs(-res.fun - float(t[k])) < 1e-7
        pts = GRID[_inside(GRID, t)]
        assert th.all((pts @ d.T).amax(0) <= t + 1e-9)


def test_emptiness_agrees_with_sampling_and_witness():
    gen = th.Generator().manual_seed(4)
    h = _random_raw(gen, 400)
    empty = G.is_empty(h)
    assert 50 < int(empty.sum()) < 350
    for hi, ei in zip(h, empty):
        hits = _inside(GRID, hi).any()
        if ei:
            assert not hits
        else:
            w = G.witness_point(hi)
            assert _inside(w.unsqueeze(0), hi + 1e-9).all()


def test_intersection_agrees_with_sampling():
    gen = th.Generator().manual_seed(5)
    h1, h2 = _random_nonempty(gen, 30), _random_nonempty(gen, 30)
    for a, b in zip(h1, h2):
        inter = G.intersect(a, b)
        expected = _inside(GRID, a) & _inside(GRID, b)
        if G.is_empty(inter):
            assert not expected.any()
        else:
            assert th.equal(_inside(GRID, inter + 1e-9), expected)
            assert th.allclose(G.tighten(inter), inter, atol=1e-12)


def test_containment_agrees_with_sampling():
    gen = th.Generator().manual_seed(6)
    d = G.DIRECTIONS.double()
    checked_in = checked_out = 0
    for _ in range(300):
        outer = _random_nonempty(gen, 1)[0] + 0.5
        inner = _random_nonempty(gen, 1)[0] * 0.4
        gap = (G.tighten(inner) - outer).max()
        if abs(float(gap)) < 4 * STEP:
            continue  # too close to the boundary to be decided by the grid
        pts = GRID[_inside(GRID, inner)]
        if len(pts) == 0:
            continue
        sampled_inside = bool(((pts @ d.T) <= outer + 1e-12).all())
        assert bool(G.contains(outer, inner)) == sampled_inside
        checked_in += sampled_inside
        checked_out += not sampled_inside
    assert checked_in > 10 and checked_out > 10


def test_disjointness_agrees_with_sampling():
    gen = th.Generator().manual_seed(7)
    checked = {True: 0, False: 0}
    for _ in range(300):
        a = _random_nonempty(gen, 1)[0] * 0.6
        b = _random_nonempty(gen, 1)[0] * 0.6
        s = G.emptiness_slack(th.minimum(a, b))
        if abs(float(s)) < 4 * STEP:
            continue
        sampled = bool((_inside(GRID, a) & _inside(GRID, b)).any())
        assert bool(G.disjoint(a, b)) == (not sampled)
        checked[not sampled] += 1
    assert checked[True] > 10 and checked[False] > 10


def test_box_is_a_tight_octagon_and_box_intersection_is_a_box():
    gen = th.Generator().manual_seed(8)
    c = (th.rand(20, 2, generator=gen, dtype=th.float64) - 0.5) * 2
    o = th.rand(20, 2, generator=gen, dtype=th.float64)
    h = G.from_box(c, o)
    assert th.allclose(G.tighten(h), h)
    assert th.allclose(G.area(h), 4 * o[:, 0] * o[:, 1])
    lo = th.maximum(c[:10] - o[:10], c[10:] - o[10:])
    hi = th.minimum(c[:10] + o[:10], c[10:] + o[10:])
    ok = (lo <= hi).all(-1)
    inter = G.intersect(h[:10], h[10:])
    assert th.equal(~G.is_empty(inter), ok)
    box = G.from_box((lo + hi) / 2, (hi - lo) / 2)
    assert th.allclose(inter[ok], box[ok])


def test_translation_moves_the_point_set():
    gen = th.Generator().manual_seed(9)
    h = _random_nonempty(gen, 1)[0] * 0.5
    t = th.tensor([0.37, -0.21], dtype=th.float64)
    pts = GRID[_inside(GRID, h)]
    assert _inside(pts + t, G.translate(h, t) + 1e-9).all()


# --------------------------------------------------------------------------------------
# Helly: three octagons can pairwise intersect without a common point (T3) ...
# --------------------------------------------------------------------------------------

def t3_octagons():
    """X1, X2 (two thin bars) and X3 (a diagonal band): pairwise intersecting, no common point."""
    x1 = G.from_box(th.tensor([0.5, 0.05]), th.tensor([0.5, 0.05]))
    x2 = G.from_box(th.tensor([0.05, 0.5]), th.tensor([0.05, 0.5]))
    # 0 <= x, y <= 1 and 0.9 <= x + y <= 1
    x3 = th.tensor([1.0, 1.0, 1.0, 1.0, 0.0, -0.9, 0.0, 1.0])
    return x1.double(), x2.double(), x3.double()


def test_t3_three_pairwise_intersecting_octagons_without_common_point():
    x1, x2, x3 = t3_octagons()
    grid = _grid(lim=1.2, step=0.005)
    m = [_inside(grid, x) for x in (x1, x2, x3)]
    for i, j in itertools.combinations(range(3), 2):
        assert (m[i] & m[j]).any()
        assert not G.disjoint([x1, x2, x3][i], [x1, x2, x3][j])
    assert not (m[0] & m[1] & m[2]).any()
    triple = th.minimum(th.minimum(x1, x2), x3)
    assert float(G.emptiness_slack(triple)) < -0.5


# --------------------------------------------------------------------------------------
# ... but four octagon products with all triples intersecting share a point (Helly
# number 3; Helly's theorem in each factor).
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("pairs", [1, 2])
def test_helly_number_three_for_octagon_products(pairs):
    gen = th.Generator().manual_seed(10 + pairs)
    hits = 0
    for _ in range(3000):
        c = (th.rand(4 * pairs, 2, generator=gen, dtype=th.float64) - 0.5) * 2.0
        r = th.rand(4 * pairs, 8, generator=gen, dtype=th.float64) * 2.0
        h = G.from_center_radius(c, r).reshape(4, pairs, 8)
        triples_ok = all(
            not bool((G.emptiness_slack(th.minimum(th.minimum(h[i], h[j]), h[k])) < 0).any())
            for i, j, k in itertools.combinations(range(4), 3))
        if not triples_ok:
            continue
        hits += 1
        common = h.amin(0)  # (pairs, 8)
        assert bool((G.emptiness_slack(common) >= -1e-12).all())
        w = G.witness_point(common)  # a point in every factor of all four products
        for i in range(4):
            for p in range(pairs):
                assert _inside(w[p].unsqueeze(0), h[i, p] + 1e-9).all()
    assert hits > 20
