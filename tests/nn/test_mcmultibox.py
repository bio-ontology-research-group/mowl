"""Tests of the volume-based MultiBoxEL (Aharoni 2024): QMC volume, soft membership, losses."""
import pytest
import torch as th

import mowl.nn.el.mcmultibox.volume as V
from mowl.nn.el.mcmultibox.module import VolumeMultiBoxELModule


def boxes_tensor(boxes, n, lo_pad=0.0):
    """List of per-box ``(lo, hi)`` coordinate lists -> ``(n, 2k)`` centre/offset rows.

    Missing boxes are repeated copies of the first one (a union is unchanged by duplicates).
    """
    rows = []
    for lo, hi in boxes + [boxes[0]] * (n - len(boxes)):
        lo, hi = th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32)
        rows.append(th.cat([(lo + hi) / 2, (hi - lo) / 2]))
    return th.stack(rows)


def set_concept(module, idx, boxes):
    module.class_embed.data[idx] = boxes_tensor(boxes, module.num_boxes)


def set_role(module, idx, boxes):
    """``boxes``: list of ``(lo, hi)`` with ``2d`` coordinates (domain then codomain)."""
    module.rel_embed.data[idx] = boxes_tensor(boxes, module.num_boxes)


def make_module(n_classes=4, n_rels=2, d=2, n=2, points=4096, **kw):
    kw.setdefault("temperature", 0.02)
    kw.setdefault("bound", 4.0)
    return VolumeMultiBoxELModule(n_classes, n_rels, embed_dim=d, num_boxes=n,
                                  num_points=points, **kw)


def idx(*rows):
    return th.tensor([rows], dtype=th.long)


# ------------------------------------------------------------------------- point set / volume
def test_sobol_points_are_deterministic_and_in_range():
    a = V.sobol_points(256, 3, 2.0, seed=7)
    b = V.sobol_points(256, 3, 2.0, seed=7)
    c = V.sobol_points(256, 3, 2.0, seed=8)
    assert th.equal(a, b)
    assert not th.equal(a, c)
    assert a.shape == (256, 3) and a.abs().max() <= 2.0


def test_qmc_volume_of_union_of_boxes_matches_analytic():
    d, bound = 2, 4.0
    pts = V.sobol_points(2 ** 15, d, bound, seed=0)
    # two overlapping boxes: [0,2]x[0,1] and [1,3]x[0,2]; union area = 2 + 4 - 1 = 5
    boxes = boxes_tensor([([0.0, 0.0], [2.0, 1.0]), ([1.0, 0.0], [3.0, 2.0])], 2)
    c, o = boxes[None, :, :d], boxes[None, :, d:]
    vol = V.qmc_volume(V.hard_membership(pts, c, o).float(), bound, d)
    assert vol.item() == pytest.approx(5.0, rel=0.02)
    # a disjoint union adds up
    boxes = boxes_tensor([([-3.0, -3.0], [-2.0, -1.0]), ([1.0, 1.0], [3.0, 2.0])], 2)
    c, o = boxes[None, :, :d], boxes[None, :, d:]
    vol = V.qmc_volume(V.hard_membership(pts, c, o).float(), bound, d)
    assert vol.item() == pytest.approx(2.0 + 2.0, rel=0.03)


def test_soft_volume_of_sharp_membership_is_proportional_to_hard_volume():
    d, bound = 2, 4.0
    pts = V.sobol_points(2 ** 15, d, bound)
    boxes = boxes_tensor([([0.0, 0.0], [2.0, 2.0])], 1)
    c, o = boxes[None, :, :d], boxes[None, :, d:]
    soft = V.qmc_volume(V.soft_membership(pts, c, o, temperature=0.01), bound, d)
    # membership is about 0.5 inside the box (sigmoid of a margin / temperature, minus 0.5)
    assert soft.item() == pytest.approx(0.5 * 4.0, rel=0.1)


# --------------------------------------------------------------------------- soft membership
def test_soft_membership_positive_iff_inside_for_min_aggregate():
    d = 3
    gen = th.Generator().manual_seed(0)
    centre = th.rand(1, 3, d, generator=gen) * 2 - 1
    offset = 0.2 + th.rand(1, 3, d, generator=gen) * 0.5
    pts = th.rand(5000, d, generator=gen) * 4 - 2
    soft = V.soft_membership(pts, centre, offset, temperature=1.0, aggregate="min")[0]
    hard = V.hard_membership(pts, centre, offset)[0]
    assert hard.any() and (~hard).any()
    assert (soft[hard] > 0).all()
    assert (soft[~hard] == 0).all()


def test_mean_aggregate_is_positive_outside_but_larger_inside():
    """The thesis' literal formula: sigma(M-p) + sigma(p-m) > 1 always, so mean > 0.5."""
    centre, offset = th.zeros(1, 1, 1), th.ones(1, 1, 1)
    pts = th.tensor([[0.0], [0.9], [1.5], [10.0]])
    soft = V.soft_membership(pts, centre, offset, aggregate="mean")[0]
    assert (soft > 0).all()
    assert soft[0] > soft[2] > soft[3]


def test_intersection_is_union_of_pairwise_box_intersections():
    d, n = 2, 2
    gen = th.Generator().manual_seed(1)
    c1, c2 = (th.rand(1, n, d, generator=gen) * 2 - 1 for _ in range(2))
    o1, o2 = (0.3 + th.rand(1, n, d, generator=gen) for _ in range(2))
    pts = th.rand(3000, d, generator=gen) * 4 - 2
    # pairwise intersection boxes (possibly inverted = empty)
    lo = th.maximum((c1 - o1).unsqueeze(2), (c2 - o2).unsqueeze(1)).reshape(1, n * n, d)
    hi = th.minimum((c1 + o1).unsqueeze(2), (c2 + o2).unsqueeze(1)).reshape(1, n * n, d)
    # hard case
    pairwise = ((pts[None, :, None, :] >= lo[:, None]) & (pts[None, :, None, :] <= hi[:, None]))
    pairwise = pairwise.all(-1).any(-1)
    both = V.hard_membership(pts, c1, o1) & V.hard_membership(pts, c2, o2)
    assert th.equal(pairwise, both)
    # soft case: min of multibox memberships = max over pairwise intersections
    soft_pair = V.soft_box_membership(pts, (lo + hi) / 2, (hi - lo) / 2, 1.0, "min")
    soft_pair = th.where((hi >= lo).all(-1).unsqueeze(1), soft_pair, th.zeros_like(soft_pair))
    soft_pair = soft_pair.amax(-1)
    soft_min = th.minimum(V.soft_membership(pts, c1, o1, 1.0, "min"),
                          V.soft_membership(pts, c2, o2, 1.0, "min"))
    assert th.allclose(soft_pair, soft_min, atol=1e-6)


# --------------------------------------------------------------------------------- GCI losses
def test_gci0_zero_when_contained_and_one_when_disjoint():
    m = make_module()
    set_concept(m, 0, [([0.0, 0.0], [1.0, 1.0]), ([-3.0, -3.0], [-2.5, -2.5])])
    set_concept(m, 1, [([-0.5, -0.5], [1.5, 1.5]), ([-3.5, -3.5], [-2.0, -2.0])])  # contains 0
    set_concept(m, 2, [([2.0, 2.0], [3.0, 3.0]), ([2.0, -3.0], [3.0, -2.0])])  # disjoint
    contained = m.gci0_loss(idx(0, 1))
    disjoint = m.gci0_loss(idx(0, 2))
    reverse = m.gci0_loss(idx(1, 0))
    assert contained.item() == pytest.approx(0.0, abs=1e-4)
    assert disjoint.item() == pytest.approx(1.0, abs=1e-4)
    assert 0.1 < reverse.item() < 0.9


def test_gci0_loss_is_positive_for_partial_overlap_and_has_gradient():
    m = make_module()
    set_concept(m, 0, [([0.0, 0.0], [2.0, 2.0])])
    set_concept(m, 1, [([1.0, 0.0], [3.0, 2.0])])  # half of concept 0 inside
    loss = m.gci0_loss(idx(0, 1))
    assert loss.item() == pytest.approx(0.5, abs=0.05)
    loss.sum().backward()
    assert m.class_embed.grad[0].abs().sum() > 0 and m.class_embed.grad[1].abs().sum() > 0


def test_chunking_does_not_change_the_loss():
    m1 = make_module(point_chunk_size=64, seed=3)
    m2 = make_module(point_chunk_size=100000, seed=3)
    data = th.tensor([[0, 1], [2, 3]])
    assert th.allclose(m1.gci0_loss(data), m2.gci0_loss(data), atol=1e-6)


def test_gci1_uses_the_intersection_of_the_left_multiboxes():
    m = make_module()
    set_concept(m, 0, [([0.0, 0.0], [2.0, 2.0])])
    set_concept(m, 1, [([1.0, 1.0], [3.0, 3.0])])  # intersection [1,2]^2
    set_concept(m, 2, [([0.5, 0.5], [2.5, 2.5])])  # contains the intersection, not concept 0
    set_concept(m, 3, [([-3.0, -3.0], [-2.0, -2.0])])
    assert m.gci1_loss(idx(0, 1, 2)).item() == pytest.approx(0.0, abs=1e-4)
    assert m.gci1_loss(idx(0, 1, 3)).item() == pytest.approx(1.0, abs=1e-4)
    assert m.gci0_loss(idx(0, 2)).item() > 0.1  # contrast: concept 0 alone is not inside 2


# --------------------------------------------------------------------- existentials / projection
def test_projection_matches_brute_force_on_a_hand_built_role():
    d, bound = 1, 8.0
    m = make_module(n_classes=2, n_rels=1, d=d, n=2, bound=bound, temperature=0.05)
    # role: ([0,1] x [2,3]) union ([4,5] x [6,7])
    set_role(m, 0, [([0.0, 2.0], [1.0, 3.0]), ([4.0, 6.0], [5.0, 7.0])])
    # filler meets only the codomain of the first role box
    set_concept(m, 0, [([2.5], [2.8]), ([-7.0], [-6.0])])
    rc, ro = m.role_boxes(th.tensor([0]))
    fc, fo = m.concept_boxes(th.tensor([0]))

    xs = th.linspace(-bound, bound, 801).unsqueeze(1)
    # brute force: x in projection iff some y with (x, y) in R, y in filler and y in S
    ys = th.linspace(-bound, bound, 4001)
    in_role_x = V.hard_box_membership(xs, rc[..., :d], ro[..., :d])[0]  # (X, n)
    in_role_y = V.hard_box_membership(ys.unsqueeze(1), rc[..., d:], ro[..., d:])[0]  # (Y, n)
    in_fill = V.hard_membership(ys.unsqueeze(1), fc, fo)[0]  # (Y,)
    has_y = (in_role_y & in_fill.unsqueeze(1)).any(0)  # per role box: exists y
    brute = (in_role_x & has_y.unsqueeze(0)).any(-1)

    # exact box-level projection (hard gate)
    hard_gate = V.hard_overlap(rc[..., d:], ro[..., d:], fc, fo, bound).any(-1)  # (1, n)
    hard_proj = (in_role_x & hard_gate).any(-1)
    assert th.equal(brute, hard_proj)
    assert brute.any() and (~brute).any()
    assert hard_gate.tolist() == [[True, False]]

    soft_gate = V.project_gate(rc, ro, fc, fo, bound, 0.05, "min")
    soft_proj = V.projection_membership(xs, rc, ro, soft_gate, 0.05, "min")[0] > 0
    assert th.equal(soft_proj, brute)


def test_gci2_and_gci3_zero_for_a_satisfying_configuration():
    m = make_module(n_classes=3, n_rels=1, d=1, n=1, bound=8.0, temperature=0.05)
    set_role(m, 0, [([0.0, 2.0], [1.0, 3.0])])
    set_concept(m, 0, [([0.25], [0.75])])  # C inside the domain
    set_concept(m, 1, [([2.2], [2.8])])  # D inside the codomain
    set_concept(m, 2, [([-0.5], [1.5])])  # E contains the domain
    assert m.gci2_loss(idx(0, 0, 1)).item() == pytest.approx(0.0, abs=1e-4)  # C <= ER.D
    assert m.gci3_loss(idx(0, 1, 2)).item() == pytest.approx(0.0, abs=1e-4)  # ER.D <= E
    # D' outside the codomain: C <= ER.D' fails completely
    set_concept(m, 1, [([5.0], [6.0])])
    assert m.gci2_loss(idx(0, 0, 1)).item() == pytest.approx(1.0, abs=1e-4)
    # E' not containing the domain: ER.D <= E' fails
    set_concept(m, 1, [([2.2], [2.8])])
    set_concept(m, 2, [([-3.0], [-2.0])])
    assert m.gci3_loss(idx(0, 1, 2)).item() == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------------------------- role axioms
def test_role_inclusion_and_chain():
    m = make_module(n_classes=1, n_rels=4, d=1, n=1, bound=4.0)
    set_role(m, 0, [([0.0, 0.0], [1.0, 1.0])])
    set_role(m, 1, [([-0.5, -0.5], [1.5, 1.5])])  # contains role 0
    set_role(m, 2, [([2.0, 2.0], [3.0, 3.0])])  # disjoint from role 0
    assert m.role_inclusion_loss(idx(0, 1)).item() == pytest.approx(0.0, abs=1e-4)
    assert m.role_inclusion_loss(idx(0, 2)).item() == pytest.approx(1.0, abs=1e-4)
    # chain r0 o r2 <= r3: domain of r0 is [0,1], codomain of r2 is [2,3]; the simplified loss
    # compares [0,1]x[2,3] with r3
    set_role(m, 3, [([-1.0, 1.5], [2.0, 3.5])])
    assert m.role_chain_loss(idx(0, 2, 3)).item() == pytest.approx(0.0, abs=1e-4)
    set_role(m, 3, [([-1.0, -1.0], [0.5, 0.5])])
    assert m.role_chain_loss(idx(0, 2, 3)).item() == pytest.approx(1.0, abs=1e-4)


def test_role_chain_loss_is_the_simplified_one():
    """Documented limitation: the middle element is not constrained.

    R1 = [0,1] x [5,6] and R2 = [7,8] x [2,3] cannot be chained (the codomain of R1 and the
    domain of R2 are disjoint) but the simplified loss only looks at domain(R1) x codomain(R2).
    """
    m = make_module(n_classes=1, n_rels=3, d=1, n=1, bound=9.0)
    set_role(m, 0, [([0.0, 5.0], [1.0, 6.0])])
    set_role(m, 1, [([7.0, 2.0], [8.0, 3.0])])
    set_role(m, 2, [([-0.5, 1.5], [1.5, 3.5])])
    assert m.role_chain_loss(idx(0, 1, 2)).item() == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------------------- T3
def build_t3(d, bound=4.0):
    """Hand-built exact configuration of the Helly pattern T3 with n = 2 boxes.

    A = X1 and X2, B = X2 and X3, C = X1 and X3, with A, B, C pairwise disjoint. Classes:
    0 A, 1 B, 2 C, 3 X1, 4 X2, 5 X3. Along the first axis A = [-3,-2], B = [-0.5,0.5], C = [2,3];
    X1 = A u C, X2 = A u B, X3 = B u C. Other coordinates span [-1, 1].
    """
    m = make_module(n_classes=6, n_rels=1, d=d, n=2, bound=bound, points=2 ** 13,
                    temperature=0.05)

    def box(a, b):
        return ([a] + [-1.0] * (d - 1), [b] + [1.0] * (d - 1))

    A, B, C = box(-3.0, -2.0), box(-0.5, 0.5), box(2.0, 3.0)
    for i, bxs in enumerate([[A], [B], [C], [A, C], [A, B], [B, C]]):
        set_concept(m, i, bxs)
    return m


@pytest.mark.parametrize("d", [1, 2])
def test_t3_admits_an_exact_two_box_configuration(d):
    m = build_t3(d)
    A, B, C, X1, X2, X3 = range(6)
    axioms = []
    for x, (p, q) in {A: (X1, X2), B: (X2, X3), C: (X1, X3)}.items():
        axioms.append(("gci0", [x, p]))  # X <= P
        axioms.append(("gci0", [x, q]))  # X <= Q
        axioms.append(("gci1", [p, q, x]))  # P and Q <= X
    for name, row in axioms:
        data = th.tensor([row])
        loss = (m.gci0_loss if name == "gci0" else m.gci1_loss)(data)
        assert loss.item() == pytest.approx(0.0, abs=1e-4), (name, row)

    # A, B, C pairwise disjoint: the (hard) volume of every pairwise intersection is zero
    pts = m.points
    for p, q in [(A, B), (B, C), (A, C)]:
        pc, po = m.concept_boxes(th.tensor([p, q]))
        both = V.hard_membership(pts, pc, po)
        assert not (both[0] & both[1]).any()
    # while the intersections X1 and X2 etc. are non-empty (equal A, B, C)
    pc, po = m.concept_boxes(th.tensor([X1, X2]))
    both = V.hard_membership(pts, pc, po)
    assert (both[0] & both[1]).any()

    # and a single box per concept cannot do this: three boxes with pairwise intersections
    # in the pattern would need the boxes X1, X2, X3 to pairwise meet in A, B, C while A, B, C
    # are disjoint, which axis-parallel boxes cannot (Helly number 2). Check one concrete
    # failure: replacing X1 by the box hull of A u C makes X1 and X2 meet in more than A.
    set_concept(m, X1, [([-3.0] + [-1.0] * (d - 1), [3.0] + [1.0] * (d - 1))])
    assert m.gci1_loss(th.tensor([[X1, X2, A]])).item() > 0.1


def test_t3_distance_ranks_the_true_subsumers_first():
    m = build_t3(2)
    A, B, C, X1, X2, X3 = range(6)
    queries = th.tensor([[A, c] for c in range(6)])
    dist = m.distance(queries, "gci0")
    # A is contained in A, X1 and X2 only
    assert dist[A].item() == 0.0 and dist[X1].item() == 0.0 and dist[X2].item() == 0.0
    assert dist[B].item() == 1.0 and dist[C].item() == 1.0 and dist[X3].item() == 1.0
    d1 = m.distance(th.tensor([[X1, X2, A], [X1, X2, B]]), "gci1")
    assert d1[0].item() == 0.0 and d1[1].item() == 1.0


# -------------------------------------------------------------------------- module interface
def test_weight_shapes_and_capabilities():
    m = VolumeMultiBoxELModule(7, 3, nb_inds=4, embed_dim=5, num_boxes=5, num_points=128)
    assert m.class_embed.shape == (7, 5, 10)
    assert m.rel_embed.shape == (3, 5, 20)
    assert m.ind_embed.shape == (4, 1, 10)
    assert m.neg_capable_gcis == frozenset()
    assert m.role_axiom_capable is True
    assert "points" not in m.state_dict()
    with pytest.raises(NotImplementedError):
        m.gci0_bot_loss(th.zeros(1, 2, dtype=th.long))
    with pytest.raises(NotImplementedError):
        m.gci0_loss(th.zeros(1, 2, dtype=th.long), neg=True)


def test_same_seed_same_module():
    a = VolumeMultiBoxELModule(5, 2, embed_dim=3, num_boxes=2, num_points=64, seed=5)
    b = VolumeMultiBoxELModule(5, 2, embed_dim=3, num_boxes=2, num_points=64, seed=5)
    assert th.equal(a.class_embed, b.class_embed) and th.equal(a.points, b.points)


_FORM_ROWS = {"gci0": (0, 0), "gci1": (0, 0, 0), "gci2": (0, 1, 0), "gci3": (1, 0, 0),
              "role_inclusion": (1, 1), "role_chain": (1, 1, 1)}


def _random_rows(form, n_classes, n_rels, count, gen):
    kinds = _FORM_ROWS[form]
    return th.stack([th.randint(n_rels if k else n_classes, (count,), generator=gen)
                     for k in kinds], 1)


@pytest.mark.parametrize("form", sorted(_FORM_ROWS))
def test_row_chunks_give_the_numbers_and_gradients_of_the_whole_batch(form):
    """The axioms of a batch are processed a few at a time (bounded memory); losses and
    gradients are those of the whole batch at once."""
    whole = make_module(n_classes=6, n_rels=3, points=300, point_chunk_size=128, seed=5,
                        temperature=0.5, max_chunk_elements=None)
    chunked = make_module(n_classes=6, n_rels=3, points=300, point_chunk_size=128, seed=5,
                          temperature=0.5, max_chunk_elements=128 * 2 * 4 * 3)
    assert chunked.rows_per_chunk("concept") == 6 and chunked.rows_per_chunk("role") == 3
    assert whole.rows_per_chunk() is None
    data = _random_rows(form, 6, 3, 17, th.Generator().manual_seed(1))
    a, b = whole(data, form), chunked(data, form)
    assert a.shape == b.shape == (17, 1)
    assert th.allclose(a, b, atol=1e-6, rtol=0)
    a.sum().backward()
    b.sum().backward()
    for p, q in ((whole.class_embed, chunked.class_embed), (whole.rel_embed, chunked.rel_embed)):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert p.grad.abs().sum() > 0
            assert th.allclose(p.grad, q.grad, atol=1e-6, rtol=0)


def test_row_chunk_size_follows_the_element_budget():
    m = VolumeMultiBoxELModule(10, 2, embed_dim=5, num_boxes=5, num_points=20000)
    # 2**25 elements / (2048 points x 5 boxes x 5 coordinates) = 655 axioms per chunk
    assert m.rows_per_chunk("concept") == 655 and m.rows_per_chunk("role") == 327
    small = VolumeMultiBoxELModule(10, 2, embed_dim=5, num_boxes=5, num_points=100)
    assert small.rows_per_chunk("concept") == 2 ** 25 // (100 * 25)
