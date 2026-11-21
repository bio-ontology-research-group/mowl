"""Exact region checks of :mod:`mowl.evaluation.regions` and the readers of
:mod:`mowl.evaluation.exact_model`: against dense sampling / vertex enumeration, against
the modules' own margin-free losses, against :class:`mowl.geometric.checker.ModelChecker`
for single boxes, and on hand-built exact (and broken) models of the Helly ontology T_3.
"""

import itertools
import math
import types

import numpy as np
import pytest
import torch as th

import mowl  # noqa: F401  (the tests package starts the JVM)
import mowl.datasets  # noqa: F401  (import order: avoids a circular import in mowl.nn)
from mowl.evaluation.exact_model import ExactModelRegime, find_reader
from mowl.evaluation.regions import (BallIntersection, BallRegion, BoxRegion, EmptyRegion,
                                     OctagonRegion, PrismRegion, UnionRegion, Universe,
                                     box_minus_boxes, lens_max_distance, satisfies)
from mowl.geometric.normal_forms import BOTTOM, NF1, NF2

NS = "http://mowl.borg/geo#"
T3_NAMES = ["X1", "X2", "X3", "A1", "A2", "A3"]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def t3(tmp_path_factory):
    from mowl.datasets.builtin.helly import HellyTkDataset
    from mowl.geometric.completion import CanonicalModel
    ds = HellyTkDataset(3, directory=str(tmp_path_factory.mktemp("helly")))
    return ds, CanonicalModel.from_dataset(ds)


def fake_model(module):
    """A stand-in for a trained model: the module and the T3 class index."""
    return types.SimpleNamespace(module=module,
                                 class_index_dict={NS + n: i for i, n in enumerate(T3_NAMES)})


def run_regime(module, t3):
    regime = ExactModelRegime()
    res = regime.evaluate(fake_model(module), t3[0])["all"]
    return res, regime.info


def t3_axioms():
    x1, x2, x3, a1, a2, a3 = T3_NAMES
    gci0 = [NF1(a, x) for a, xs in ((a1, (x2, x3)), (a2, (x1, x3)), (a3, (x1, x2)))
            for x in xs]
    gci1 = [NF2(x2, x3, a1), NF2(x1, x3, a2), NF2(x1, x2, a3)]
    bot = [NF2(a, b, BOTTOM) for a, b in ((a1, a2), (a1, a3), (a2, a3))]
    return gci0 + gci1 + bot


# --------------------------------------------------------------------------------------
# Boxes and unions of boxes
# --------------------------------------------------------------------------------------

def test_box_minus_boxes_shared_faces():
    a = BoxRegion([0, 0], [2, 1])
    assert box_minus_boxes(a, [BoxRegion([0, 0], [1, 1]), BoxRegion([1, 0], [2, 1])]) == []
    gap = [BoxRegion([0, 0], [1, 1]), BoxRegion([1.001, 0], [2, 1])]
    assert box_minus_boxes(a, gap)
    assert box_minus_boxes(a, gap, tol=0.001) == []
    # A degenerate box on the shared face is covered; one on an uncovered face is not.
    assert box_minus_boxes(BoxRegion([1, 0], [1, 1]), gap[:1]) == []
    assert box_minus_boxes(BoxRegion([2, 0], [2, 1]), gap[:1])


def _lattice(lo, hi, step):
    axes = [np.arange(a, b + step / 2, step) for a, b in zip(lo, hi)]
    return np.stack([g.reshape(-1) for g in np.meshgrid(*axes, indexing="ij")], axis=1)


def _random_union(rng, n_max=3, size=5):
    boxes = []
    for _ in range(rng.integers(1, n_max + 1)):
        lo = rng.integers(0, 8, size=2) * 0.25
        boxes.append(BoxRegion(lo, lo + rng.integers(0, size, size=2) * 0.25))
    return UnionRegion(boxes)


def test_union_inclusion_matches_lattice_sampling():
    """Boxes with corners on the 1/4 lattice: every non-empty piece of a difference
    A \\ (B_1 u ... u B_n) contains a point of the 1/8 lattice, so lattice sampling is an
    exact independent check of union-in-union inclusion and of union intersection."""
    rng = np.random.default_rng(0)
    pts = _lattice([-0.25, -0.25], [4.25, 4.25], 0.125)
    n_true = n_multi = 0
    for trial in range(3000):
        a, b = _random_union(rng, 2, 3), _random_union(rng, 5, 8)
        if trial % 2:  # split a cover of a's first piece at a lattice line, maybe with a gap
            p = a.pieces[0]
            ax = int(rng.integers(2))
            cut = p.lo[ax] + rng.integers(0, 4) * 0.25
            gap = rng.integers(0, 2) * 0.25
            hi1, lo2 = p.hi.copy(), p.lo.copy()
            hi1[ax], lo2[ax] = cut, cut + gap
            cover = [BoxRegion(p.lo - 0.25, hi1), BoxRegion(lo2, p.hi + 0.25)]
            b = UnionRegion(cover + b.pieces[:2])
        in_a, in_b = a.contains_points(pts), b.contains_points(pts)
        inside = b.contains(a)
        assert inside == bool(np.all(in_b[in_a]))
        n_true += inside
        n_multi += inside and not all(any(q.contains(p) for q in b.pieces) for p in a.pieces)
        inter = a.intersect(b)
        assert np.array_equal(inter.contains_points(pts), in_a & in_b)
        assert inter.is_empty() == (not np.any(in_a & in_b))
    # not vacuous: also pieces that only the union of several boxes covers
    assert n_true > 50 and n_multi > 10


def test_box_region_agrees_with_model_checker_on_t3():
    from mowl.geometric.checker import DictInterpretation, ModelChecker
    from mowl.geometric.regions import Box
    rng = np.random.default_rng(1)
    axioms = t3_axioms()
    agree = 0
    for _ in range(300):
        boxes = {}
        for n in T3_NAMES:
            c = rng.uniform(-1, 1, 2)
            o = np.abs(rng.uniform(-1, 1, 2))
            boxes[n] = BoxRegion(c - o, c + o)
        lo = np.min([b.lo for b in boxes.values()], axis=0) - 1
        hi = np.max([b.hi for b in boxes.values()], axis=0) + 1
        checker = ModelChecker(DictInterpretation(
            "box", Box(lo, hi), classes={n: b.as_geometric_box() for n, b in boxes.items()}),
            tol=1e-6)
        for ax in axioms:
            ours = satisfies(ax, lambda n: EmptyRegion(2) if n == BOTTOM else boxes[n], 1e-6)
            assert ours == checker.satisfies(ax)
            agree += 1
    assert agree == 300 * len(axioms)


# --------------------------------------------------------------------------------------
# Balls
# --------------------------------------------------------------------------------------

def _lens_boundary_points(b1, b2, n=20000):
    """Dense points of the boundary of the lens of two discs (2-D)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = []
    for p, q in ((b1, b2), (b2, b1)):
        circ = p.center + p.radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        out.append(circ[q.contains_points(circ, 1e-12)])
    return np.concatenate(out)


def test_lens_max_distance_matches_dense_sampling():
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(300):
        b1 = BallRegion(rng.uniform(-1, 1, 2), rng.uniform(0.2, 1.2))
        b2 = BallRegion(rng.uniform(-1, 1, 2), rng.uniform(0.2, 1.2))
        e = rng.uniform(-2, 2, 2)
        far = lens_max_distance(b1.center, b1.radius, b2.center, b2.radius, e)
        pts = _lens_boundary_points(b1, b2)
        if len(pts) == 0:
            # empty lens, or one disc inside the other (then no boundary point of one
            # lies in the other except for the inner circle)
            continue
        sampled = np.linalg.norm(pts - e, axis=1).max()
        assert sampled <= far + 1e-9
        assert far - sampled < 1e-3
        checked += 1
    assert checked > 150


def test_lens_in_ball_matches_sampling_in_3d():
    rng = np.random.default_rng(3)
    n_in = n_out = 0
    for _ in range(200):
        b1 = BallRegion(rng.uniform(-1, 1, 3), rng.uniform(0.3, 1.2))
        b2 = BallRegion(rng.uniform(-1, 1, 3), rng.uniform(0.3, 1.2))
        lens = b1.intersect(b2)
        if lens.is_empty():
            continue
        sup = BallRegion(rng.uniform(-0.5, 0.5, 3), rng.uniform(0.5, 2.0))
        far = lens_max_distance(b1.center, b1.radius, b2.center, b2.radius, sup.center)
        if abs(far - sup.radius) < 0.02:
            continue  # too close to call by sampling
        pts = lens.sample(20000, rng)
        outside = ~sup.contains_points(pts)
        if sup.contains(lens):
            assert not outside.any()
            n_in += 1
        else:
            assert outside.any()
            n_out += 1
    assert n_in > 10 and n_out > 10


def test_ball_intersection_emptiness_and_witness():
    rng = np.random.default_rng(4)
    pts = _lattice([-2, -2], [2, 2], 0.01)
    seen = {True: 0, False: 0}
    for _ in range(150):
        m = int(rng.integers(3, 5))
        balls = [BallRegion(rng.uniform(-1, 1, 2), rng.uniform(0.3, 1.3)) for _ in range(m)]
        inter = BallIntersection(balls)
        empty = inter.is_empty()
        hit = inter.contains_points(pts)
        # Robust cases only: the intersection is either clearly non-empty (a lattice
        # point at depth 0.02) or empty even with radii enlarged by 0.02.
        deep = BallIntersection([BallRegion(b.center, b.radius - 0.02) for b in balls])
        wide = BallIntersection([BallRegion(b.center, b.radius + 0.02) for b in balls])
        if deep.contains_points(pts).any():
            assert not empty
        elif wide.is_empty():
            assert empty and not hit.any()
        else:
            continue
        if not empty:
            w = inter.witness()
            assert all(np.linalg.norm(w - b.center) <= b.radius + 1e-9 for b in balls)
        seen[empty] += 1
    assert seen[True] > 10 and seen[False] > 10


def test_ball_has_interior():
    assert not BallRegion([0, 0], 0.0).has_interior()
    assert BallRegion([0, 0], 1e-3).has_interior(tol=1e-6)
    tangent = BallIntersection([BallRegion([0, 0], 1), BallRegion([2, 0], 1)])
    assert not tangent.is_empty() and not tangent.has_interior()
    assert BallIntersection([BallRegion([0, 0], 1), BallRegion([1.9, 0], 1)]).has_interior()


# --------------------------------------------------------------------------------------
# Prisms and octagons (independent check by vertex enumeration)
# --------------------------------------------------------------------------------------

def _prism_vertices(p):
    corners = np.array(list(itertools.product(*zip(p.lo, p.hi))))
    return np.array([np.concatenate([v, c]) for v in p.triangle_vertices() for c in corners])


def _random_prism(rng, d=4):
    from mowl.nn.el.prism.losses import NORMALS
    c, r = rng.uniform(-1, 1, 2), rng.uniform(0, 1)
    h = NORMALS.double().numpy() @ c + r
    bc, bo = rng.uniform(-1, 1, d - 2), rng.uniform(0, 1, d - 2)
    return PrismRegion(h, bc - bo, bc + bo)


def test_prism_containment_matches_vertices():
    rng = np.random.default_rng(5)
    n_true = 0
    for _ in range(2000):
        a, b = _random_prism(rng), _random_prism(rng)
        inter = b.intersect(a)
        if inter.is_empty():
            continue
        # sub = a cap b is in b always; is it in a fresh prism c?
        c = _random_prism(rng)
        verts = _prism_vertices(inter)
        assert b.contains(inter, 1e-9) and a.contains(inter, 1e-9)
        assert c.contains(inter, 1e-9) == bool(c.contains_points(verts, 1e-9).all())
        n_true += c.contains(inter)
        pts = inter.sample(200, rng)
        assert a.contains_points(pts).all() and b.contains_points(pts).all()
    assert n_true > 5


def _octagon_vertices(h):
    from mowl.nn.el.octagon import geometry as G
    t = G.tighten(th.as_tensor(h, dtype=th.float64)).numpy()
    n = G.DIRECTIONS.double().numpy()
    return np.array([np.linalg.solve(n[[k, (k + 1) % 8]], t[[k, (k + 1) % 8]])
                     for k in range(8)])


def _random_octagon(rng):
    from mowl.nn.el.octagon import geometry as G
    c = th.as_tensor(rng.uniform(-1, 1, 2))
    r = th.as_tensor(rng.uniform(0, 1, 8))
    return OctagonRegion(G.from_center_radius(c, r).numpy().reshape(1, 8))


def test_octagon_containment_matches_vertices():
    rng = np.random.default_rng(6)
    n_true = 0
    for _ in range(2000):
        a, b, c = _random_octagon(rng), _random_octagon(rng), _random_octagon(rng)
        inter = a.intersect(b)
        if inter.is_empty():
            continue
        verts = _octagon_vertices(inter.h[0])
        assert c.contains(inter, 1e-9) == bool(c.contains_points(verts, 1e-9).all())
        n_true += c.contains(inter)
        assert inter.has_interior() == (inter.inradius() > 0)
    assert n_true > 5


# --------------------------------------------------------------------------------------
# Readers against the modules' own margin-free losses
# --------------------------------------------------------------------------------------

def _factories():
    from mowl.nn.el.boxel.module import BoxELModule
    from mowl.nn.el.boxsquaredel.module import BoxSquaredELModule
    from mowl.nn.el.elbe.module import ELBEModule
    from mowl.nn.el.elem.module import ELEmModule
    from mowl.nn.el.elkem.module import ELKEmModule
    from mowl.nn.el.emelpp.module import EmELPPModule, EmELVarModule
    from mowl.nn.el.multiboxel.module import MultiBoxELModule
    from mowl.nn.el.multielbe.module import MultiELBEModule
    from mowl.nn.el.octagon.module import OctagonELModule
    from mowl.nn.el.prism.module import PrismModule
    from mowl.nn.el.regd.module import RegDELBEModule, RegDELEmModule
    from mowl.nn.el.transbox.module import TransBoxModule
    n = 8
    return {
        "elbe": lambda: ELBEModule(n, 1, embed_dim=2, margin=0.0),
        "box2el": lambda: BoxSquaredELModule(n, 1, embed_dim=2, gamma=0.0),
        "transbox": lambda: TransBoxModule(n, 1, embed_dim=2, margin=0.0),
        "elkem": lambda: ELKEmModule(n, 1, embed_dim=2, margin=0.0).eval(),
        "regd": lambda: RegDELBEModule(n, 1, embed_dim=2, margin=0.0),
        "boxel": lambda: BoxELModule(n, 1, embed_dim=2),
        "elem": lambda: ELEmModule(n, 1, 0, embed_dim=2, margin=0.0),
        "emelpp": lambda: EmELPPModule(n, 1, embed_dim=2, margin=0.0),
        "emelvar": lambda: EmELVarModule(n, 1, embed_dim=2, margin=0.0),
        "regd_elem": lambda: RegDELEmModule(n, 1, embed_dim=2, margin=0.0),
        "multiboxel": lambda: MultiBoxELModule(n, 1, embed_dim=2, n_components=2,
                                               gamma=0.0, margin=0.0),
        "multielbe": lambda: MultiELBEModule(n, 1, embed_dim=2, n_components=2, margin=0.0),
        "prism": lambda: PrismModule(n, 1, embed_dim=3, margin=0.0),
        "octagon": lambda: OctagonELModule(n, 1, embed_dim=2, margin=0.0),
    }


#: How the margin-free loss of each normal form relates to the exact verdict:
#: "iff" -- zero loss exactly when the axiom holds; "sound" -- zero loss implies the axiom
#: holds (the loss is stronger, e.g. it also asks the intersection to be non-empty, or
#: disjointness in every coordinate). Normal forms whose loss is never exactly zero
#: (EmEL++'s centre-norm penalty is removed by normalising the centres below; RegD's
#: energies are positive) or not sound (ELEm's gci1 loss only asks the centre of E to lie
#: in C and D) are left out.
LOSS_SEMANTICS = {
    "elbe": {"gci0": "iff", "gci1": "sound", "gci1_bot": "sound"},
    "box2el": {"gci0": "iff", "gci1": "sound", "gci1_bot": "sound"},
    # TransBox's gci1_bot loss is its gci1 loss with the (trained) box of owl:Nothing
    # as superclass, so its zero set says nothing about disjointness.
    "transbox": {"gci0": "iff", "gci1": "sound"},
    "elkem": {"gci0": "iff", "gci1": "sound", "gci1_bot": "iff"},
    "regd": {"gci1_bot": "sound"},
    "boxel": {"gci0": "iff", "gci1": "sound"},
    "elem": {"gci0": "iff", "gci1_bot": "iff"},
    "emelpp": {"gci0": "iff", "gci1_bot": "iff"},
    "emelvar": {"gci0": "iff", "gci1_bot": "iff"},
    "regd_elem": {"gci1_bot": "iff"},
    "multiboxel": {"gci0": "sound", "gci1": "sound", "gci1_bot": "sound"},
    "multielbe": {"gci0": "sound", "gci1": "sound", "gci1_bot": "sound"},
    "prism": {"gci0": "iff", "gci1": "sound", "gci1_bot": "iff"},
    "octagon": {"gci0": "iff", "gci1": "sound", "gci1_bot": "iff"},
}


def _randomize(mod, seed, unit_centres=False):
    g = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in mod.parameters():
            p.copy_(th.rand(p.shape, generator=g, dtype=p.dtype) * 2 - 1)
        if unit_centres:
            w = mod.class_embed.weight
            w.div_(w.norm(dim=1, keepdim=True))


@pytest.mark.parametrize("name", sorted(LOSS_SEMANTICS))
def test_reader_agrees_with_margin_free_loss(name):
    make = _factories()[name]
    n = 8
    pairs = th.tensor(list(itertools.permutations(range(n), 2)))
    triples = th.tensor(list(itertools.permutations(range(n), 3)))
    bot_rows = th.cat([pairs, th.zeros(len(pairs), 1, dtype=th.long)], dim=1)
    counts = {g: [0, 0] for g in LOSS_SEMANTICS[name]}  # [zero loss, satisfied]
    for seed in range(40):
        mod = make().double()
        _randomize(mod, seed, unit_centres=name in ("emelpp", "emelvar"))
        if name in ("multiboxel", "multielbe"):  # smaller components: some disjoint pairs
            with th.no_grad():
                mod.class_offset.weight.mul_(0.3 if seed % 2 else 1.0)
        regions = find_reader(mod)(mod)

        def region(s):
            return EmptyRegion(regions[0].dim) if s == BOTTOM else regions[int(s)]
        for gci, kind in LOSS_SEMANTICS[name].items():
            rows = {"gci0": pairs, "gci1": triples, "gci1_bot": bot_rows}[gci]
            with th.no_grad():
                loss = getattr(mod, gci + "_loss")(rows).reshape(-1).tolist()
            for r, value in zip(rows.tolist(), loss):
                ax = (NF1(str(r[0]), str(r[1])) if gci == "gci0" else
                      NF2(str(r[0]), str(r[1]), str(r[2]) if gci == "gci1" else BOTTOM))
                sat = satisfies(ax, region, tol=1e-9)
                zero = value <= 1e-12
                if zero:
                    assert sat, f"{name} {gci} {r}: zero loss but violated"
                if kind == "iff" and sat:
                    assert zero, f"{name} {gci} {r}: satisfied but loss {value}"
                counts[gci][0] += zero
                counts[gci][1] += sat
    for gci, (n_zero, n_sat) in counts.items():
        assert n_zero > 0, f"{name} {gci}: no zero-loss case, the check is vacuous"


def test_multibox_cover_by_two_components_is_decided():
    """A box covered by two components of a union but by neither alone: the margin-based
    MultiBoxEL loss is positive, the exact check says the axiom holds."""
    from mowl.nn.el.multiboxel.module import MultiBoxELModule
    mod = MultiBoxELModule(2, 1, embed_dim=1, n_components=2, gamma=0.0, margin=0.0).double()
    with th.no_grad():
        mod.class_center.weight.copy_(th.tensor([[1.0, 1.0], [0.5, 1.5]]))
        mod.class_offset.weight.copy_(th.tensor([[1.0, 1.0], [0.5, 0.5]]))
    regions = find_reader(mod)(mod)
    assert regions[0].contains(regions[1]) and regions[1].contains(regions[0])
    with th.no_grad():
        assert float(mod.gci0_loss(th.tensor([[0, 1]])).max()) > 0


# --------------------------------------------------------------------------------------
# Hand-built exact models of T3 (and broken ones)
# --------------------------------------------------------------------------------------

# 1-D: X1 = [0,1] u [2,3], X2 = [0,1] u [4,5], X3 = [2,3] u [4,5]; A1 = X2 n X3 = [4,5],
# A2 = X1 n X3 = [2,3], A3 = X1 n X2 = [0,1] (second component repeated).
T3_INTERVALS = {"X1": [(0.5, 2.5)], "X2": [(0.5, 4.5)], "X3": [(2.5, 4.5)],
                "A1": [(4.5, 4.5)], "A2": [(2.5, 2.5)], "A3": [(0.5, 0.5)]}


def _set_multibox(weight_c, weight_o, k=2, d=2, broken=False):
    """Write T3_INTERVALS on axis 0 (half-width 0.5) and [0, 1] on axis 1."""
    with th.no_grad():
        for i, n in enumerate(T3_NAMES):
            c1, c2 = T3_INTERVALS[n][0]
            if broken and n == "A1":
                c1 = c2 = 2.5  # A1 := A2
            for j, c in enumerate((c1, c2)[:k]):
                weight_c[i, j * d:(j + 1) * d] = th.tensor([c, 0.5])
                weight_o[i, j * d:(j + 1) * d] = th.tensor([0.5, 0.5])


def _multiboxel(broken=False):
    from mowl.nn.el.multiboxel.module import MultiBoxELModule
    mod = MultiBoxELModule(6, 1, embed_dim=2, n_components=2).double()
    _set_multibox(mod.class_center.weight, mod.class_offset.weight, broken=broken)
    return mod


def _multielbe(broken=False):
    from mowl.nn.el.multielbe.module import MultiELBEModule
    mod = MultiELBEModule(6, 1, embed_dim=2, n_components=2).double()
    _set_multibox(mod.class_embed.weight, mod.class_offset.weight, broken=broken)
    return mod


def _mcmultibox(broken=False):
    from mowl.nn.el.mcmultibox.module import VolumeMultiBoxELModule
    mod = VolumeMultiBoxELModule(6, 1, embed_dim=2, num_boxes=2, num_points=64)
    w = th.zeros(6, 2, 4)
    for i, n in enumerate(T3_NAMES):
        c1, c2 = T3_INTERVALS[n][0]
        if broken and n == "A1":
            c1 = c2 = 2.5
        for j, c in enumerate((c1, c2)):
            w[i, j] = th.tensor([c, 0.5, 0.5, 0.5])
    with th.no_grad():
        mod.class_embed.copy_(w)
    return mod


def _prism(broken=False):
    """Triangle part: X_i has support -0.4 on normal i and 1 on the others; pairwise
    intersections have support sum 0.2 > 0, the triple one -1.2 < 0. Box part [0, 1]."""
    from mowl.nn.el.prism.losses import NORMALS
    from mowl.nn.el.prism.module import PrismModule
    xs = [np.array([1.0, 1.0, 1.0]) for _ in range(3)]
    for i in range(3):
        xs[i][i] = -0.4
    a = [np.minimum(xs[1], xs[2]), np.minimum(xs[0], xs[2]), np.minimum(xs[0], xs[1])]
    if broken:
        a[0] = xs[1]  # A1 := X2
    mod = PrismModule(6, 1, embed_dim=3, margin=0.0).double()
    normals = NORMALS.double().numpy()
    with th.no_grad():
        for i, h in enumerate(xs + a):
            r = h.sum() / 3.0
            c = np.linalg.lstsq(normals, h - r, rcond=None)[0]
            mod.tri_center.weight[i] = th.as_tensor(c)
            mod.tri_inradius.weight[i] = r
            mod.box_center.weight[i] = 0.5
            mod.box_offset.weight[i] = 0.5
    return mod


def _octagon(broken=False):
    """The exact T3 configuration of tests/nn/test_octagon_module.py."""
    from mowl.nn.el.octagon import geometry as G
    from mowl.nn.el.octagon.module import OctagonELModule
    x1 = G.from_box(th.tensor([0.5, 0.05]), th.tensor([0.5, 0.05])).double()
    x2 = G.from_box(th.tensor([0.05, 0.5]), th.tensor([0.05, 0.5])).double()
    x3 = th.tensor([1.0, 1.0, 1.0, 1.0, 0.0, -0.9, 0.0, 1.0], dtype=th.float64)
    a = [G.intersect(x2, x3), G.intersect(x1, x3), G.intersect(x1, x2)]
    if broken:
        a[2] = x1  # A3 := X1
    mod = OctagonELModule(6, 1, embed_dim=2).double()
    for i, h in enumerate([x1, x2, x3] + a):
        mod.set_class_bounds(i, h.reshape(1, 8))
    return mod


EXACT_T3 = {"multiboxel": _multiboxel, "multielbe": _multielbe, "mcmultibox": _mcmultibox,
            "prism": _prism, "octagon": _octagon}


@pytest.mark.parametrize("name", sorted(EXACT_T3))
def test_hand_built_exact_t3_model(name, t3):
    res, info = run_regime(EXACT_T3[name](), t3)
    assert info["unchecked_axioms"] == 0 and info["checked_axioms"] == 12, info
    assert res["is_model"] == 1.0, info["violated_axioms"]
    assert res["exact_model"] == 1.0 and res["exact_model_nonempty"] == 1.0
    assert res["no_interior_fraction"] == 0.0 and res["empty_fraction"] == 0.0
    assert res["helly_violation_rate"] == 0.0


@pytest.mark.parametrize("name", sorted(EXACT_T3))
def test_hand_built_broken_t3_model(name, t3):
    res, info = run_regime(EXACT_T3[name](broken=True), t3)
    assert res["exact_model"] == 0.0 and res["is_model"] == 0.0
    assert info["violated_axioms"]


# --------------------------------------------------------------------------------------
# Single boxes and balls cannot be exact non-degenerate models of T3
# --------------------------------------------------------------------------------------

def _regions_from(names_to_regions):
    return {NS + n: r for n, r in names_to_regions.items()}


def test_single_boxes_fail_t3_on_disjointness(t3):
    """Best case for boxes: random X_i and A_i := the intersection of the other two. Every
    inclusion holds, so the verdict must fail on a disjointness axiom (or on a degenerate
    A_i), never on an inclusion -- Helly: pairwise intersecting boxes share a point."""
    rng = np.random.default_rng(7)
    regime = ExactModelRegime()
    for _ in range(300):
        xs = []
        for _ in range(3):
            c, o = rng.uniform(-1, 1, 2), rng.uniform(0, 1, 2)
            xs.append(BoxRegion(c - o, c + o))
        a = [xs[1].intersect(xs[2]), xs[0].intersect(xs[2]), xs[0].intersect(xs[1])]
        # An empty intersection cannot be a name's region: put that A_i far away.
        far = BoxRegion([9, 9], [9.5, 9.5])
        regs = dict(zip(T3_NAMES, xs + [r if isinstance(r, BoxRegion) else far for r in a]))
        res = regime.evaluate_regions(_regions_from(regs), t3[1])["all"]
        assert res["exact_model"] == 0.0 and res["exact_model_nonempty"] == 0.0
        if all(isinstance(r, BoxRegion) for r in a):
            assert all("Nothing" in v for v in regime.info["violated_axioms"])
            assert res["helly_violation_rate"] == 1.0


def test_random_search_finds_no_single_box_or_ball_model(t3):
    from mowl.nn.el.elbe.module import ELBEModule
    from mowl.nn.el.elem.module import ELEmModule
    best = {}
    for name, make in (("elbe", lambda: ELBEModule(6, 1, embed_dim=2)),
                       ("elem", lambda: ELEmModule(6, 1, 0, embed_dim=2))):
        for seed in range(200):
            mod = make().double()
            _randomize(mod, seed)
            res, _ = run_regime(mod, t3)
            assert res["exact_model"] == 0.0 and res["exact_model_nonempty"] == 0.0
            best[name] = max(best.get(name, 0), res["axiom_satisfaction"])
    assert best["elbe"] < 1 and best["elem"] < 1


def test_lens_is_never_a_ball_with_interior():
    """A_3 = X_1 n X_2 needs the lens to equal a ball: random candidate balls never
    satisfy both X_1 n X_2 in A_3 and A_3 in X_1, A_3 in X_2 unless one disc contains the
    other (then the lens is a disc). This is why balls fail T3 (the A_i would have to be
    proper lenses), and the checker reports it through these axioms."""
    rng = np.random.default_rng(8)
    names = {"X1": None, "X2": None, "A3": None}
    axioms = [NF2("X1", "X2", "A3"), NF1("A3", "X1"), NF1("A3", "X2")]
    proper = 0
    for _ in range(200):
        x1 = BallRegion(rng.uniform(-1, 1, 2), rng.uniform(0.3, 1))
        x2 = BallRegion(rng.uniform(-1, 1, 2), rng.uniform(0.3, 1))
        lens = x1.intersect(x2)
        if not isinstance(lens, BallIntersection) or not lens.has_interior():
            continue
        proper += 1
        for _ in range(100):
            names.update(X1=x1, X2=x2, A3=BallRegion(rng.uniform(-1, 1, 2),
                                                     rng.uniform(0.01, 1.5)))
            assert not all(satisfies(ax, names.__getitem__, 1e-6) for ax in axioms)
    assert proper > 50


def test_tangent_balls_are_a_model_with_point_regions(t3):
    """Three pairwise externally tangent discs (radii 1, 2, 3 on a 3-4-5 triangle) with
    the tangency points as A_i: a model of T3 with non-empty names, but the A_i have no
    interior, so exact_model = 0 and exact_model_nonempty = 1."""
    from mowl.nn.el.elem.module import ELEmModule
    centres = [[0, 0], [3, 0], [0, 4], [1.8, 1.6], [0, 1], [1, 0]]
    radii = [1, 2, 3, 0, 0, 0]
    mod = ELEmModule(6, 1, 0, embed_dim=2).double()
    with th.no_grad():
        mod.class_embed.weight.copy_(th.tensor(centres, dtype=th.float64))
        mod.class_rad.weight.copy_(th.tensor(radii, dtype=th.float64).reshape(-1, 1))
    res, info = run_regime(mod, t3)
    assert res["is_model"] == 1.0, info["violated_axioms"]
    assert res["exact_model_nonempty"] == 1.0 and res["exact_model"] == 0.0
    assert res["no_interior_fraction"] == pytest.approx(0.5)
    assert res["helly_violation_rate"] == 0.0
    assert info["region_kind"] == "ball"


def test_universe_and_empty():
    u, e, b = Universe(2), EmptyRegion(2), BoxRegion([0, 0], [1, 1])
    assert u.contains(b) and not b.contains(u) and b.contains(e)
    assert u.intersect(b) is b and b.intersect(u) is b
    assert e.is_empty() and not u.is_empty() and math.isinf(u.inradius())
