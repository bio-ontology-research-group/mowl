"""Loss semantics of OctagonELModule, the exact T3 model, the ELBE special case and the
Helly limit (T4)."""
import itertools

import pytest
import torch as th

import mowl
mowl.init_jvm("2g")
import mowl.datasets  # noqa: F401,E402  (import order: avoids a circular import in mowl.nn)
from mowl.nn.el.octagon import geometry as G  # noqa: E402
from mowl.baselines.nn.el.octagon.module import OctagonELModule  # noqa: E402
from mowl.nn.el.elbe.module import ELBEModule  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")

STEP = 0.01
XS = th.arange(-3.0, 3.0 + STEP / 2, STEP, dtype=th.float64)
_GX, _GY = th.meshgrid(XS, XS, indexing="ij")
GRID = th.stack([_GX.reshape(-1), _GY.reshape(-1)], dim=-1)


def _inside(points, h, tol=0.0):
    """Explicit octagon inequalities (independent of the module and of ``geometry``)."""
    x, y = points[:, 0], points[:, 1]
    h = h + tol
    conds = [x <= h[0], x + y <= h[1], y <= h[2], y - x <= h[3],
             -x <= h[4], -x - y <= h[5], -y <= h[6], x - y <= h[7]]
    return th.stack(conds, dim=0).all(dim=0)


def _module(n, d=2, n_rels=1, **kw):
    mod = OctagonELModule(n, n_rels, embed_dim=d, **kw).double()
    return mod


def _rand_h(gen, scale=1.0, shift=0.0):
    c = (th.rand(2, generator=gen, dtype=th.float64) - 0.5) * 2.0 * scale + shift
    r = th.rand(8, generator=gen, dtype=th.float64) * 1.2 * scale
    return G.from_center_radius(c, r)


def _idx(*rows):
    return th.tensor(rows, dtype=th.long)


# --------------------------------------------------------------------------------------
# Basic properties
# --------------------------------------------------------------------------------------

def test_shapes_and_validation():
    mod = OctagonELModule(5, 2, nb_inds=3, embed_dim=6)
    assert mod.class_bounds(th.arange(5)).shape == (5, 3, 8)
    for name, data in [("gci0", _idx([0, 1])), ("gci1", _idx([0, 1, 2])),
                       ("gci2", _idx([0, 1, 2])), ("gci3", _idx([1, 0, 2])),
                       ("gci0_bot", _idx([0, 4])), ("gci1_bot", _idx([0, 1, 4])),
                       ("gci3_bot", _idx([1, 0, 4])), ("class_assertion", _idx([0, 1])),
                       ("object_property_assertion", _idx([0, 1, 2]))]:
        out = mod(data, name)
        assert out.shape == (1, 1) and th.isfinite(out).all() and (out >= 0).all()
    assert mod.role_axiom_capable is False
    assert mod.neg_capable_gcis == frozenset({"gci2"})
    for bad in (3, 0, 7):
        with pytest.raises(ValueError):
            OctagonELModule(2, 1, embed_dim=bad)


def test_concept_names_are_nonempty_by_construction():
    mod = OctagonELModule(50, 1, embed_dim=8)
    with th.no_grad():
        mod.class_radius.weight.normal_()
    assert (G.emptiness_slack(mod.all_class_bounds()) >= 0).all()


def test_set_class_bounds_roundtrip():
    gen = th.Generator().manual_seed(0)
    mod = _module(1, d=4)
    h = th.stack([_rand_h(gen), _rand_h(gen)])
    mod.set_class_bounds(0, h)
    assert th.allclose(mod.class_bounds(_idx(0)), h.unsqueeze(0), atol=1e-12)


# --------------------------------------------------------------------------------------
# Exactness: zero loss <=> geometric condition (checked by sampling)
# --------------------------------------------------------------------------------------

def _sampled_subset(h_sub, h_sup):
    pts = GRID[_inside(GRID, h_sub)]
    return bool(_inside(pts, h_sup, 1e-12).all())


def test_gci0_zero_iff_inclusion():
    gen = th.Generator().manual_seed(1)
    mod = _module(2)
    seen = {True: 0, False: 0}
    for _ in range(300):
        c, d = _rand_h(gen, 0.3), _rand_h(gen, 1.5, 0.1)
        gap = float((G.tighten(c) - d).max())
        if abs(gap) < 4 * STEP:
            continue
        mod.set_class_bounds(0, c.unsqueeze(0))
        mod.set_class_bounds(1, d.unsqueeze(0))
        zero = float(mod(_idx([0, 1]), "gci0")) <= 1e-9
        sampled = _sampled_subset(c, d)
        assert zero == sampled
        seen[sampled] += 1
    assert seen[True] > 10 and seen[False] > 10


def test_gci1_zero_iff_nonempty_intersection_included():
    gen = th.Generator().manual_seed(2)
    mod = _module(3)
    seen = {True: 0, False: 0}
    for _ in range(400):
        c, d = _rand_h(gen, 0.5), _rand_h(gen, 0.5)
        e = G.from_center_radius((th.rand(2, generator=gen, dtype=th.float64) - 0.5) * 0.6,
                                 0.6 + th.rand(8, generator=gen, dtype=th.float64))
        inter = th.minimum(c, d)
        s = float(G.emptiness_slack(inter))
        gap = float((G.tighten(inter) - e).max())
        if abs(s) < 4 * STEP or (s > 0 and abs(gap) < 4 * STEP):
            continue
        for i, h in enumerate((c, d, e)):
            mod.set_class_bounds(i, h.unsqueeze(0))
        zero = float(mod(_idx([0, 1, 2]), "gci1")) <= 1e-9
        both = _inside(GRID, c) & _inside(GRID, d)
        sampled = bool(both.any()) and bool(_inside(GRID[both], e, 1e-12).all())
        assert zero == sampled
        seen[sampled] += 1
    assert seen[True] > 10 and seen[False] > 10


@pytest.mark.parametrize("margin", [0.0, 0.05])
def test_gci1_bot_zero_iff_disjoint(margin):
    gen = th.Generator().manual_seed(3)
    mod = _module(2, disjoint_margin=margin)
    seen = {True: 0, False: 0}
    for _ in range(300):
        c, d = _rand_h(gen, 0.6), _rand_h(gen, 0.6)
        s = float(G.emptiness_slack(th.minimum(c, d)))
        if abs(s + margin) < 4 * STEP or abs(s) < 4 * STEP:
            continue
        mod.set_class_bounds(0, c.unsqueeze(0))
        mod.set_class_bounds(1, d.unsqueeze(0))
        zero = float(mod(_idx([0, 1, 1]), "gci1_bot")) <= 1e-12
        sampled_disjoint = not bool((_inside(GRID, c) & _inside(GRID, d)).any())
        if margin == 0.0:
            assert zero == sampled_disjoint
        else:
            # zero iff the smallest width of the intersection is <= -margin
            assert zero == (s <= -margin)
            assert (not zero) or sampled_disjoint
        seen[sampled_disjoint] += 1
    assert seen[True] > 10 and seen[False] > 10


def test_gci2_and_gci3_are_translated_inclusions():
    gen = th.Generator().manual_seed(4)
    mod = _module(2, n_rels=1)
    seen = {True: 0, False: 0}
    for _ in range(300):
        c, d = _rand_h(gen, 0.4), _rand_h(gen, 1.0, 0.3)
        t = (th.rand(2, generator=gen, dtype=th.float64) - 0.5)
        with th.no_grad():
            mod.rel_embed.weight[0] = t
        mod.set_class_bounds(0, c.unsqueeze(0))
        mod.set_class_bounds(1, d.unsqueeze(0))
        c_plus, c_minus = G.translate(c, t), G.translate(c, -t)
        if min(abs(float((G.tighten(c_plus) - d).max())),
               abs(float((G.tighten(c_minus) - d).max()))) < 4 * STEP:
            continue
        zero2 = float(mod(_idx([0, 0, 1]), "gci2")) <= 1e-9
        zero3 = float(mod(_idx([0, 0, 1]), "gci3")) <= 1e-9
        pts = GRID[_inside(GRID, c)]
        assert zero2 == bool(_inside(pts + t, d, 1e-9).all())   # C + t ⊆ D
        assert zero3 == bool(_inside(pts - t, d, 1e-9).all())   # ∃R.C = C - t ⊆ D
        neg_zero = float(mod(_idx([0, 0, 1]), "gci2", neg=True)) <= 1e-12
        if not zero2 and float((G.tighten(c_plus) - d).max()) > mod.neg_margin:
            assert neg_zero
        if zero2:
            assert not neg_zero
        seen[zero2] += 1
    assert seen[True] > 5 and seen[False] > 5


def test_bottom_losses_zero_iff_empty_interior():
    mod = _module(2)
    flat = G.from_box(th.tensor([0.0, 0.0]), th.tensor([1.0, 0.0])).double()
    full = G.from_box(th.tensor([0.0, 0.0]), th.tensor([1.0, 0.5])).double()
    mod.set_class_bounds(0, flat.unsqueeze(0))
    mod.set_class_bounds(1, full.unsqueeze(0))
    assert float(mod(_idx([0, 1]), "gci0_bot")) == 0.0
    assert float(mod(_idx([1, 1]), "gci0_bot")) > 0.0
    assert float(mod(_idx([0, 0, 1]), "gci3_bot")) == 0.0
    assert float(mod(_idx([0, 1, 1]), "gci3_bot")) > 0.0


def test_assertions():
    mod = OctagonELModule(1, 1, nb_inds=2, embed_dim=2).double()
    mod.set_class_bounds(0, G.from_box(th.zeros(2), th.ones(2)).double().unsqueeze(0))
    with th.no_grad():
        mod.ind_embed.weight[0] = th.tensor([0.5, -0.5])
        mod.ind_embed.weight[1] = th.tensor([1.5, 0.0])
        mod.rel_embed.weight[0] = th.tensor([1.0, 0.5])
    # columns [individual, class]
    assert float(mod(_idx([0, 0]), "class_assertion")) == 0.0
    assert float(mod(_idx([1, 0]), "class_assertion")) > 0.0
    assert float(mod(_idx([0, 0, 1]), "object_property_assertion")) < 1e-12
    assert float(mod(_idx([1, 0, 0]), "object_property_assertion")) > 0.0


# --------------------------------------------------------------------------------------
# Boxes are a special case: comparison with ELBE (mowl.nn.el.elbe)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("margin", [0.0, 0.1])
def test_box_special_case_matches_elbe(margin):
    gen = th.Generator().manual_seed(5)
    n, d = 40, 4
    centers = (th.rand(n, d, generator=gen, dtype=th.float64) - 0.5) * 2
    offsets = th.rand(n, d, generator=gen, dtype=th.float64) * 0.4
    offsets[: n // 4] += 1.5  # some large boxes, so that inclusions occur
    rels = (th.rand(3, d, generator=gen, dtype=th.float64) - 0.5)
    elbe = ELBEModule(n, 3, embed_dim=d, margin=margin).double()
    octa = OctagonELModule(n, 3, embed_dim=d, margin=margin).double()
    with th.no_grad():
        elbe.class_embed.weight.copy_(centers)
        elbe.class_offset.weight.copy_(offsets)
        elbe.rel_embed.weight.copy_(rels)
        octa.rel_embed.weight.copy_(rels)
    for i in range(n):
        octa.set_class_bounds(i, G.from_box(centers[i].view(2, 2), offsets[i].view(2, 2)))
    pairs = th.tensor(list(itertools.permutations(range(n), 2)))
    e0 = elbe(pairs, "gci0").reshape(-1)
    o0 = octa(pairs, "gci0").reshape(-1)
    assert th.equal(e0 <= 1e-12, o0 <= 1e-12)
    assert ((e0 > 1e-12).sum() > 10) and ((e0 <= 1e-12).sum() > 10)
    assert th.all(e0 <= o0 + 1e-9) and th.all(o0 <= 5 ** 0.5 * e0 + 1e-9)
    # the axis-parallel entries of the octagon loss are exactly ELBE's per-coordinate terms
    hc, hd = octa.class_bounds(pairs[:, 0]), octa.class_bounds(pairs[:, 1])
    v = G.tighten(hc) - hd + margin
    axis = th.relu(th.maximum(v[..., :4], v[..., 4:]))[..., [0, 2]].reshape(len(pairs), -1)
    dc = (centers[pairs[:, 0]] - centers[pairs[:, 1]]).abs()
    do = offsets[pairs[:, 0]] - offsets[pairs[:, 1]]
    elbe_terms = th.relu(dc + do + margin)
    assert th.allclose(axis, elbe_terms, atol=1e-12)

    triples = th.tensor([[i, r, j] for i, j in itertools.permutations(range(12), 2)
                         for r in range(3)])
    e2 = elbe(triples, "gci2").reshape(-1)
    o2 = octa(triples, "gci2").reshape(-1)
    assert th.equal(e2 <= 1e-12, o2 <= 1e-12)
    assert th.all(e2 <= o2 + 1e-9) and th.all(o2 <= 5 ** 0.5 * e2 + 1e-9)

    # gci1 (C ⊓ D ⊑ E): same zero set (non-empty intersection, contained in E)
    tri = th.tensor(list(itertools.permutations(range(12), 3)))
    assert th.equal(elbe(tri, "gci1").reshape(-1) <= 1e-12, octa(tri, "gci1").reshape(-1) <= 1e-12)


# --------------------------------------------------------------------------------------
# T3 = {A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3, A, B, C pairwise disjoint}: exact model in d=2
# --------------------------------------------------------------------------------------

def t3_axioms():
    """Classes: 0=X1, 1=X2, 2=X3, 3=A, 4=B, 5=C (as in tests/nn/test_prism_losses.py)."""
    gci0 = _idx([3, 0], [3, 1], [4, 1], [4, 2], [5, 0], [5, 2])
    gci1 = _idx([0, 1, 3], [1, 2, 4], [0, 2, 5])
    gci1_bot = _idx([3, 4, 6], [4, 5, 6], [3, 5, 6], [3, 2, 6], [4, 0, 6], [5, 1, 6])
    return gci0, gci1, gci1_bot


def t3_bounds():
    x1 = G.from_box(th.tensor([0.5, 0.05]), th.tensor([0.5, 0.05])).double()
    x2 = G.from_box(th.tensor([0.05, 0.5]), th.tensor([0.05, 0.5])).double()
    x3 = th.tensor([1.0, 1.0, 1.0, 1.0, 0.0, -0.9, 0.0, 1.0], dtype=th.float64)
    return [x1, x2, x3, G.intersect(x1, x2), G.intersect(x2, x3), G.intersect(x1, x3)]


def write_exact_t3(mod, extra_pairs=0):
    for i, h in enumerate(t3_bounds()):
        rows = [h] + [G.from_box(th.zeros(2), th.ones(2)).double()] * extra_pairs
        mod.set_class_bounds(i, th.stack(rows))


@pytest.mark.parametrize("d", [2, 4])
def test_t3_exact_zero_loss_configuration(d):
    mod = _module(7, d=d, margin=0.0, disjoint_margin=0.05)
    write_exact_t3(mod, extra_pairs=d // 2 - 1)
    gci0, gci1, gci1_bot = t3_axioms()
    assert float(mod(gci0, "gci0").max()) <= 1e-12
    assert float(mod(gci1, "gci1").max()) <= 1e-12
    assert float(mod(gci1_bot, "gci1_bot").max()) == 0.0  # disjoint with a gap >= 0.05


def test_t3_exact_configuration_is_a_model_by_sampling():
    """Independent of the losses: sample the plane and check every T3 axiom pointwise."""
    mod = _module(7, d=2)
    write_exact_t3(mod)
    h = mod.class_bounds(th.arange(6)).detach()[:, 0, :]
    xs = th.arange(-0.2, 1.2, 0.002, dtype=th.float64)
    gx, gy = th.meshgrid(xs, xs, indexing="ij")
    pts = th.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    m = [_inside(pts, hi, 1e-9) for hi in h]
    x1, x2, x3, a, b, c = m
    assert th.equal(a, x1 & x2) and th.equal(b, x2 & x3) and th.equal(c, x1 & x3)
    for s in (a, b, c):
        assert s.any()
    assert not (a & b).any() and not (b & c).any() and not (a & c).any()


# --------------------------------------------------------------------------------------
# The Helly limit: T4 has no model with non-empty concept names
# --------------------------------------------------------------------------------------

def t4_axioms():
    """X1..X4 = 0..3, A_i ≡ ⊓_{j≠i} X_j = 4..7, auxiliaries N_i ≡ first two of them = 8..11,
    and A_i ⊓ X_i ⊑ ⊥ (so X1 ⊓ X2 ⊓ X3 ⊓ X4 ⊑ ⊥)."""
    gci0, gci1, bot = [], [], []
    for i in range(4):
        j, k, m = [x for x in range(4) if x != i]
        a, n = 4 + i, 8 + i
        gci0 += [[a, j], [a, k], [a, m], [n, j], [n, k]]
        gci1 += [[j, k, n], [n, m, a]]
        bot += [[a, i, 12]]
    return _idx(*gci0), _idx(*gci1), _idx(*bot)


@pytest.mark.parametrize("d", [2, 4])
def test_t4_definitions_force_a_common_point(d):
    """Helly number 3. Take random X1..X4 (plus the T3 bars as adversarial X1..X3) and set
    every defined class to its definition (N_i = X_j ∩ X_k, A_i = N_i ∩ X_m), so that all
    gci0/gci1 losses are exactly zero. Whenever all A_i are non-empty (as a model needs),
    X1..X4 share a point; it lies in A_i ∩ X_i, so every disjointness loss is at least its
    margin: no octagon-product configuration is a model of T4."""
    gci0, gci1, bot = t4_axioms()
    pairs = d // 2
    gen = th.Generator().manual_seed(d)
    mod = _module(13, d=d, margin=0.0, disjoint_margin=0.1)
    t3 = t3_bounds()[:3]
    checked = 0
    for trial in range(400):
        xs = [th.stack([_rand_h(gen, 0.5) + 0.6 for _ in range(pairs)]) for _ in range(4)]
        if trial % 2:
            for i in range(3):
                xs[i][0] = t3[i]
        defs = {}
        ok = True
        for i in range(4):
            j, k, m = [x for x in range(4) if x != i]
            n_i = th.minimum(xs[j], xs[k])
            a_i = th.minimum(n_i, xs[m])
            if bool((G.emptiness_slack(a_i) < 0).any()):
                ok = False
                break
            defs[8 + i], defs[4 + i] = G.tighten(n_i), G.tighten(a_i)
        if not ok:
            continue
        for i in range(4):
            mod.set_class_bounds(i, xs[i])
        for idx, h in defs.items():
            mod.set_class_bounds(idx, h)
        with th.no_grad():
            assert float(mod(gci0, "gci0").max()) <= 1e-9
            assert float(mod(gci1, "gci1").max()) <= 1e-9
            assert float(mod(bot, "gci1_bot").min()) >= 0.1 - 1e-9
        common = th.stack(xs).amin(0)
        q = G.witness_point(common)
        for i in range(4):
            for p in range(pairs):
                assert _inside(q[p].unsqueeze(0), xs[i][p], 1e-9).all()
        checked += 1
    assert checked > 50


def test_t3_analogue_of_the_t4_check_is_satisfiable():
    """Contrast: for three concepts the same construction admits disjointness (T3)."""
    mod = _module(7, d=2, margin=0.0, disjoint_margin=0.05)
    write_exact_t3(mod)
    _, _, gci1_bot = t3_axioms()
    assert float(mod(gci1_bot, "gci1_bot").max()) == 0.0
