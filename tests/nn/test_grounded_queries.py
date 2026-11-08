"""Unit tests for mowl.nn.el.grounded.queries (MultiBox query compiler and scorers)."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pytest
import torch as th

from mowl.nn import GroundedELModule
from mowl.nn.el.grounded import (
    MultiBox,
    anchor,
    class_boxes,
    compile_query,
    depth,
    intersect,
    overlap_depth,
    project,
    project_exact,
    score_concepts,
    score_entities,
    union,
)

ALL_STRUCTURES: Tuple[Tuple[str, Any], ...] = (
    ("1p", ("e", ("r",))),
    ("2p", ("e", ("r", "r"))),
    ("3p", ("e", ("r", "r", "r"))),
    ("2i", (("e", ("r",)), ("e", ("r",)))),
    ("3i", (("e", ("r",)), ("e", ("r",)), ("e", ("r",)))),
    ("2u", (("e", ("r",)), ("e", ("r",)), ("u",))),
    ("ip", ((("e", ("r",)), ("e", ("r",))), ("r",))),
    ("pi", (("e", ("r", "r")), ("e", ("r",)))),
    ("up", ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",))),
)


def _make_translation_module(
    seed: int = 42,
    inverse_of: Sequence[int] = (1, 0, -1),
) -> GroundedELModule:
    """Build a tiny translation-mode module (d=2, 30 elements, 3 roles with K_r=2, 4 classes) with non-trivial overlaps."""
    gen = th.Generator().manual_seed(seed)
    mod = GroundedELModule(
        nb_elements=30,
        nb_classes=4,
        nb_rels=3,
        embed_dim=2,
        n_components=2,
        n_role_components=2,
        role_mode="translation",
        inverse_of=inverse_of,
    )
    with th.no_grad():
        # Spread elements in [-1.0, 1.0]^2 and give roles moderate widths so multi-hop
        # intersections, projections, and element witnesses are well-populated.
        mod.elem_pos.weight.copy_(
            th.rand((30, 2), generator=gen) * 2.0 - 1.0
        )
        mod.role_centers.copy_(
            th.rand((3, 2, 2), generator=gen) * 0.8 - 0.4
        )
        # Softplus raw widths around 0.0 -> width ~ 0.69
        mod.role_raw_widths.copy_(
            th.rand((3, 2, 2), generator=gen) * 0.6 - 0.3
        )
        mod.class_centers.copy_(
            th.rand((4, 2, 2), generator=gen) * 1.2 - 0.6
        )
        mod.class_raw_widths.copy_(
            th.rand((4, 2, 2), generator=gen) * 0.6 - 0.3
        )
    return mod


def _extract_numpy_geometry(
    mod: GroundedELModule,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract element positions and role translation boxes using raw parameter tensors and numpy math."""
    pos = mod.elem_pos.weight.detach().cpu().numpy().copy()  # (N, d)
    raw_c = mod.role_centers.detach().cpu().numpy().copy()  # (R, K_r, d)
    raw_w = mod.role_raw_widths.detach().cpu().numpy().copy()  # (R, K_r, d)
    act_buf = mod.role_active.detach().cpu().numpy().copy()  # (R, K_r)
    inv_of = mod.inverse_of.detach().cpu().numpy().copy()  # (R,)

    nb_rels, kr, dim = raw_c.shape
    role_c = np.zeros_like(raw_c)
    role_w = np.zeros_like(raw_w)
    role_act = np.zeros_like(act_buf, dtype=bool)

    for r in range(nb_rels):
        p = int(inv_of[r])
        if 0 <= p < r:
            c_r = -raw_c[p]
            rw_r = raw_w[p]
            a_r = act_buf[p]
        else:
            c_r = raw_c[r]
            rw_r = raw_w[r]
            a_r = act_buf[r]
        w_pos = np.log1p(np.exp(rw_r))
        w_r = np.where(a_r[:, None], w_pos, 0.0)
        eff_a = a_r & (np.min(w_r, axis=-1) > 0.0)
        role_c[r] = c_r
        role_w[r] = w_r
        role_act[r] = eff_a

    return pos, role_c, role_w, role_act


def _np_region_eval(
    structure: Any,
    query: Any,
    pos: np.ndarray,
    role_c: np.ndarray,
    role_w: np.ndarray,
    role_act: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Brute-force region semantics in pure numpy: list of non-empty [lo, hi] boxes."""
    if isinstance(structure, tuple) and len(structure) == 2 and structure[0] == "e":
        e_id = int(query[0])
        rels = [int(r) for r in query[1]]
        boxes: List[Tuple[np.ndarray, np.ndarray]] = [(pos[e_id].copy(), pos[e_id].copy())]
        for r in rels:
            next_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
            for lo, hi in boxes:
                for k in range(role_c.shape[1]):
                    if role_act[r, k]:
                        r_lo = role_c[r, k] - role_w[r, k]
                        r_hi = role_c[r, k] + role_w[r, k]
                        next_boxes.append((lo + r_lo, hi + r_hi))
            boxes = next_boxes
        return boxes

    if (
        isinstance(structure, tuple)
        and len(structure) == 2
        and isinstance(structure[1], tuple)
        and all(x == "r" for x in structure[1])
    ):
        boxes = _np_region_eval(structure[0], query[0], pos, role_c, role_w, role_act)
        rels = [int(r) for r in query[1]]
        for r in rels:
            next_boxes = []
            for lo, hi in boxes:
                for k in range(role_c.shape[1]):
                    if role_act[r, k]:
                        r_lo = role_c[r, k] - role_w[r, k]
                        r_hi = role_c[r, k] + role_w[r, k]
                        next_boxes.append((lo + r_lo, hi + r_hi))
            boxes = next_boxes
        return boxes

    if isinstance(structure, tuple) and structure[-1] == ("u",):
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(structure) - 1):
            out.extend(
                _np_region_eval(structure[i], query[i], pos, role_c, role_w, role_act)
            )
        return out

    # Intersection
    acc = _np_region_eval(structure[0], query[0], pos, role_c, role_w, role_act)
    for i in range(1, len(structure)):
        rhs = _np_region_eval(structure[i], query[i], pos, role_c, role_w, role_act)
        inter: List[Tuple[np.ndarray, np.ndarray]] = []
        for lo1, hi1 in acc:
            for lo2, hi2 in rhs:
                lo = np.maximum(lo1, lo2)
                hi = np.minimum(hi1, hi2)
                if np.all(hi > lo):
                    inter.append((lo, hi))
        acc = inter
    return acc


def _np_answers_from_boxes(
    boxes: List[Tuple[np.ndarray, np.ndarray]],
    pos: np.ndarray,
) -> np.ndarray:
    """Return boolean mask (N,) of elements strictly inside at least one [lo, hi] box."""
    n_elem = pos.shape[0]
    ans = np.zeros((n_elem,), dtype=bool)
    for lo, hi in boxes:
        inside = np.all((pos > lo[None, :]) & (pos < hi[None, :]), axis=-1)
        ans |= inside
    return ans


def _np_element_witness_answers(
    structure: Any,
    query: Any,
    pos: np.ndarray,
    role_c: np.ndarray,
    role_w: np.ndarray,
    role_act: np.ndarray,
) -> np.ndarray:
    """Brute-force element-witness semantics in numpy: intermediate variables range over elements."""
    n_elem = pos.shape[0]
    nb_rels = role_c.shape[0]
    # Precompute binary role adjacency matrices adj[r, x, y] in numpy
    adj = np.zeros((nb_rels, n_elem, n_elem), dtype=bool)
    diff = pos[None, :, :] - pos[:, None, :]  # (N_x, N_y, d) = p_y - p_x
    for r in range(nb_rels):
        for k in range(role_c.shape[1]):
            if role_act[r, k]:
                in_box = np.all(
                    np.abs(diff - role_c[r, k][None, None, :]) < role_w[r, k][None, None, :],
                    axis=-1,
                )
                adj[r] |= in_box

    def _eval(struct: Any, q: Any) -> np.ndarray:
        if isinstance(struct, tuple) and len(struct) == 2 and struct[0] == "e":
            e_id = int(q[0])
            rels = [int(r) for r in q[1]]
            cur = adj[rels[0], e_id].copy()  # first hop from anchor e_id
            for r in rels[1:]:
                # Witness elements z with cur[z] == True
                next_cur = np.zeros((n_elem,), dtype=bool)
                for z in np.flatnonzero(cur):
                    next_cur |= adj[r, z]
                cur = next_cur
            return cur

        if (
            isinstance(struct, tuple)
            and len(struct) == 2
            and isinstance(struct[1], tuple)
            and all(x == "r" for x in struct[1])
        ):
            cur = _eval(struct[0], q[0])
            rels = [int(r) for r in q[1]]
            for r in rels:
                next_cur = np.zeros((n_elem,), dtype=bool)
                for z in np.flatnonzero(cur):
                    next_cur |= adj[r, z]
                cur = next_cur
            return cur

        if isinstance(struct, tuple) and struct[-1] == ("u",):
            out = np.zeros((n_elem,), dtype=bool)
            for i in range(len(struct) - 1):
                out |= _eval(struct[i], q[i])
            return out

        out = _eval(struct[0], q[0])
        for i in range(1, len(struct)):
            out = out & _eval(struct[i], q[i])
        return out

    return _eval(structure, query)


def _sample_random_query_instance(
    structure: Any,
    rng: np.random.Generator,
    nb_elements: int,
    nb_rels: int,
) -> Any:
    """Sample a random query instance matching ``structure``."""
    if isinstance(structure, tuple) and len(structure) == 2 and structure[0] == "e":
        e_id = int(rng.integers(0, nb_elements))
        rels = tuple(int(rng.integers(0, nb_rels)) for _ in structure[1])
        return (e_id, rels)
    if (
        isinstance(structure, tuple)
        and len(structure) == 2
        and isinstance(structure[1], tuple)
        and all(x == "r" for x in structure[1])
    ):
        sub_q = _sample_random_query_instance(structure[0], rng, nb_elements, nb_rels)
        rels = tuple(int(rng.integers(0, nb_rels)) for _ in structure[1])
        return (sub_q, rels)
    if isinstance(structure, tuple) and structure[-1] == ("u",):
        parts = [
            _sample_random_query_instance(s, rng, nb_elements, nb_rels)
            for s in structure[:-1]
        ]
        parts.append((-1,))
        return tuple(parts)
    return tuple(
        _sample_random_query_instance(s, rng, nb_elements, nb_rels)
        for s in structure
    )


# ---------------------------------------------------------------------------
# 1. Brute-force equivalence on all Query2Box structures (region semantics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,structure", ALL_STRUCTURES)
def test_compile_query_matches_numpy_region_brute_force(
    name: str,
    structure: Any,
) -> None:
    """score_entities(compile_query(...)) > 0 equals explicit numpy box arithmetic on every structure."""
    mod = _make_translation_module(seed=101)
    pos, role_c, role_w, role_act = _extract_numpy_geometry(mod)
    rng = np.random.default_rng(2026)

    batch_size = 24
    queries = [
        _sample_random_query_instance(structure, rng, mod.nb_elements, mod.nb_rels)
        for _ in range(batch_size)
    ]

    with th.no_grad():
        compiled = compile_query(mod, structure, queries, max_boxes=64)
        scores = score_entities(mod, compiled)  # (B, 30)

    assert scores.shape == (batch_size, mod.nb_elements)
    got_mask = (scores.cpu().numpy() > 0.0)

    expected_mask = np.zeros_like(got_mask)
    for b, q in enumerate(queries):
        boxes = _np_region_eval(structure, q, pos, role_c, role_w, role_act)
        expected_mask[b] = _np_answers_from_boxes(boxes, pos)

    np.testing.assert_array_equal(
        got_mask,
        expected_mask,
        err_msg=f"Mismatch against numpy region brute force on structure {name}",
    )


# ---------------------------------------------------------------------------
# 2. project_exact & compile_query(..., exact=True) vs element-witness brute force
# ---------------------------------------------------------------------------


def test_project_exact_and_compile_query_exact_match_element_witness_brute_force() -> None:
    """project_exact and compile_query(..., exact=True) equal brute-force over element witnesses."""
    mod = _make_translation_module(seed=303)
    pos, role_c, role_w, role_act = _extract_numpy_geometry(mod)
    rng = np.random.default_rng(404)

    # 2a. Direct project_exact test from 1p regions
    batch_size = 20
    q_1p = [
        (int(rng.integers(0, mod.nb_elements)), (int(rng.integers(0, mod.nb_rels)),))
        for _ in range(batch_size)
    ]
    hop2_rels = [int(rng.integers(0, mod.nb_rels)) for _ in range(batch_size)]

    with th.no_grad():
        r1 = compile_query(mod, ("e", ("r",)), q_1p)
        r2_exact = project_exact(
            mod,
            r1,
            hop2_rels,
            element_ids=th.arange(mod.nb_elements),
            max_witnesses=256,
            max_boxes=64,
        )
        got_direct = (score_entities(mod, r2_exact).cpu().numpy() > 0.0)

    for b in range(batch_size):
        q_2p = (q_1p[b][0], (q_1p[b][1][0], hop2_rels[b]))
        exp_b = _np_element_witness_answers(
            ("e", ("r", "r")), q_2p, pos, role_c, role_w, role_act
        )
        np.testing.assert_array_equal(got_direct[b], exp_b)

    # 2b. compile_query(..., exact=True) on all structures
    for name, structure in ALL_STRUCTURES:
        queries = [
            _sample_random_query_instance(structure, rng, mod.nb_elements, mod.nb_rels)
            for _ in range(batch_size)
        ]
        with th.no_grad():
            compiled_ex = compile_query(
                mod,
                structure,
                queries,
                exact=True,
                max_witnesses=256,
                max_boxes=64,
            )
            got_ex = (score_entities(mod, compiled_ex).cpu().numpy() > 0.0)

        exp_ex = np.zeros_like(got_ex)
        for b, q in enumerate(queries):
            exp_ex[b] = _np_element_witness_answers(
                structure, q, pos, role_c, role_w, role_act
            )
        np.testing.assert_array_equal(
            got_ex,
            exp_ex,
            err_msg=f"Exact witness mismatch on structure {name}",
        )


# ---------------------------------------------------------------------------
# 3. intersect drops empties & keeps exact box; union concatenates; max_boxes cap
# ---------------------------------------------------------------------------


def test_intersect_union_and_max_boxes_volume_cap() -> None:
    """intersect drops empty pairs and keeps exact box; union concatenates; max_boxes keeps largest volume."""
    # R1 has 2 boxes in 2D:
    #   box 0: cen=[0.0, 0.0], w=[1.0, 1.0] -> [-1, 1] x [-1, 1]
    #   box 1: cen=[10.0, 10.0], w=[0.5, 0.5] -> [9.5, 10.5]^2 (disjoint from R2)
    # R2 has 1 box in 2D:
    #   box 0: cen=[0.5, -0.5], w=[1.0, 1.0] -> [-0.5, 1.5] x [-1.5, 0.5]
    # Exact intersection of R1[0] and R2[0]:
    #   lo = [-0.5, -1.0], hi = [1.0, 0.5] -> cen = [0.25, -0.25], w = [0.75, 0.75]
    r1 = MultiBox(
        cen=th.tensor([[[0.0, 0.0], [10.0, 10.0]]]),
        w=th.tensor([[[1.0, 1.0], [0.5, 0.5]]]),
        active=th.tensor([[True, True]]),
    )
    r2 = MultiBox(
        cen=th.tensor([[[0.5, -0.5]]]),
        w=th.tensor([[[1.0, 1.0]]]),
        active=th.tensor([[True]]),
    )

    inter = intersect(r1, r2, max_boxes=64)
    assert int(inter.active.sum()) == 1
    assert bool(inter.active[0, 0]) is True
    assert bool(inter.active[0, 1]) is False
    th.testing.assert_close(inter.cen[0, 0], th.tensor([0.25, -0.25]))
    th.testing.assert_close(inter.w[0, 0], th.tensor([0.75, 0.75]))
    # With max_boxes=1, the empty intersection is dropped completely
    inter_cap1 = intersect(r1, r2, max_boxes=1)
    assert inter_cap1.cen.shape == (1, 1, 2)
    assert bool(inter_cap1.active[0, 0]) is True
    th.testing.assert_close(inter_cap1.cen[0, 0], th.tensor([0.25, -0.25]))
    th.testing.assert_close(inter_cap1.w[0, 0], th.tensor([0.75, 0.75]))

    # Union concatenates when M1 + M2 <= max_boxes
    u_full = union(r1, r2, max_boxes=64)
    assert u_full.cen.shape == (1, 3, 2)
    assert u_full.active.tolist() == [[True, True, True]]
    th.testing.assert_close(
        u_full.cen,
        th.tensor([[[0.0, 0.0], [10.0, 10.0], [0.5, -0.5]]]),
    )
    th.testing.assert_close(
        u_full.w,
        th.tensor([[[1.0, 1.0], [0.5, 0.5], [1.0, 1.0]]]),
    )

    # Cap at 2 with 3 boxes of distinct volumes:
    #   box A: w=[0.2, 0.2] (area = 0.16, smallest -> dropped)
    #   box B: w=[2.0, 1.5] (area = 12.0, largest -> kept first)
    #   box C: w=[0.8, 1.0] (area = 3.2, second largest -> kept second)
    b_a = MultiBox(
        cen=th.tensor([[[1.0, 1.0], [2.0, 2.0]]]),
        w=th.tensor([[[0.2, 0.2], [2.0, 1.5]]]),
        active=th.tensor([[True, True]]),
    )
    b_c = MultiBox(
        cen=th.tensor([[[3.0, 3.0]]]),
        w=th.tensor([[[0.8, 1.0]]]),
        active=th.tensor([[True]]),
    )
    u_cap2 = union(b_a, b_c, max_boxes=2)
    assert u_cap2.cen.shape == (1, 2, 2)
    assert u_cap2.active.tolist() == [[True, True]]
    th.testing.assert_close(u_cap2.cen, th.tensor([[[2.0, 2.0], [3.0, 3.0]]]))
    th.testing.assert_close(u_cap2.w, th.tensor([[[2.0, 1.5], [0.8, 1.0]]]))


# ---------------------------------------------------------------------------
# 4. type_inv_rel and type_rel handling
# ---------------------------------------------------------------------------


def test_type_inv_rel_and_type_rel_handling() -> None:
    """Anchor at a class entity + type^- gives that class's boxes; region + type_rel gives union of overlapping class boxes."""
    mod = _make_translation_module(seed=505)
    # Dataset entity mapping:
    #   dataset entities 0..3 -> class entities for classes 0..3 (class_of_entity = [0, 1, 2, 3], entity_to_element = [-1, -1, -1, -1])
    #   dataset entities 4..9 -> individual elements 10..15
    class_of_entity = th.tensor([0, 1, 2, 3, -1, -1, -1, -1, -1, -1], dtype=th.long)
    entity_to_element = th.tensor([-1, -1, -1, -1, 10, 11, 12, 13, 14, 15], dtype=th.long)
    type_rel = 98
    type_inv_rel = 99

    # 4a. Anchor at class entity + type_inv_rel gives exact class_boxes(A)
    q_inv = [(0, (type_inv_rel,)), (2, (type_inv_rel,))]
    compiled_inv = compile_query(
        mod,
        ("e", ("r",)),
        q_inv,
        type_rel=type_rel,
        type_inv_rel=type_inv_rel,
        class_of_entity=class_of_entity,
        entity_to_element=entity_to_element,
    )
    expected_cls = class_boxes(mod, [0, 2])
    th.testing.assert_close(compiled_inv.cen, expected_cls.cen)
    th.testing.assert_close(compiled_inv.w, expected_cls.w)
    assert compiled_inv.active.tolist() == expected_cls.active.tolist()

    # 4b. Region followed by type_rel gives union of overlapping class boxes
    with th.no_grad():
        # Set class boxes explicitly so classes 1 and 3 overlap the projected region of entity 4 along role 0,
        # while classes 0 and 2 are far away.
        r1 = compile_query(
            mod,
            ("e", ("r",)),
            [(4, (0,))],
            entity_to_element=entity_to_element,
        )
        c_anchor = r1.cen[0, 0].clone()
        mod.class_centers[0].fill_(50.0)
        mod.class_centers[1, 0].copy_(c_anchor)
        mod.class_centers[1, 1].copy_(c_anchor + 0.1)
        mod.class_centers[2].fill_(-50.0)
        mod.class_centers[3, 0].copy_(c_anchor - 0.1)
        mod.class_centers[3, 1].copy_(c_anchor + 0.2)

        compiled_type = compile_query(
            mod,
            ("e", ("r", "r")),
            [(4, (0, type_rel))],
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
            entity_to_element=entity_to_element,
        )
        expected_union = union(
            class_boxes(mod, [1]),
            class_boxes(mod, [3]),
        )
        assert int(compiled_type.active.sum()) == int(expected_union.active.sum())
        th.testing.assert_close(
            compiled_type.cen[compiled_type.active],
            expected_union.cen[expected_union.active],
        )
        th.testing.assert_close(
            compiled_type.w[compiled_type.active],
            expected_union.w[expected_union.active],
        )


# ---------------------------------------------------------------------------
# 5. score_concepts positive iff the boxes overlap (hand-built example)
# ---------------------------------------------------------------------------


def test_score_concepts_positive_iff_boxes_overlap() -> None:
    """score_concepts and overlap_depth are > 0 iff query box and class box strictly overlap."""
    mod = GroundedELModule(
        nb_elements=6,
        nb_classes=4,
        nb_rels=1,
        embed_dim=2,
        n_components=1,
        n_role_components=1,
        role_mode="translation",
    )
    # Set half-widths of all 4 classes to exact 1.0
    raw_w_one = float(np.log(np.expm1(1.0)))
    with th.no_grad():
        mod.class_raw_widths.fill_(raw_w_one)
        # Class 0: cen=[0.0, 0.0], w=[1.0, 1.0] -> [-1, 1] x [-1, 1] (overlaps R)
        # Class 1: cen=[1.5, 0.5], w=[1.0, 1.0] -> [0.5, 2.5] x [-0.5, 1.5] (overlaps R)
        # Class 2: cen=[2.0, 0.0], w=[1.0, 1.0] -> [1.0, 3.0] x [-1, 1] (touches boundary of R at x=1.0 -> depth == 0)
        # Class 3: cen=[3.0, 0.0], w=[1.0, 1.0] -> [2.0, 4.0] x [-1, 1] (separated by 1.0 -> depth == -1.0 < 0)
        mod.class_centers[0, 0] = th.tensor([0.0, 0.0])
        mod.class_centers[1, 0] = th.tensor([1.5, 0.5])
        mod.class_centers[2, 0] = th.tensor([2.0, 0.0])
        mod.class_centers[3, 0] = th.tensor([3.0, 0.0])

    # Query region R: single box at [0.0, 0.0] with w=[1.0, 1.0] -> [-1, 1] x [-1, 1]
    R = MultiBox(
        cen=th.tensor([[[0.0, 0.0]]]),
        w=th.tensor([[[1.0, 1.0]]]),
        active=th.tensor([[True]]),
    )
    scores = score_concepts(mod, R).detach()  # (1, 4)
    assert scores.shape == (1, 4)
    assert float(scores[0, 0]) == pytest.approx(2.0, abs=1e-5)
    assert float(scores[0, 1]) == pytest.approx(0.5, abs=1e-5)
    assert float(scores[0, 2]) == pytest.approx(0.0, abs=1e-5)
    assert float(scores[0, 3]) == pytest.approx(-1.0, abs=1e-5)
    assert (scores > 0.0).tolist() == [[True, True, False, False]]

    # Also verify overlap_depth directly against class_boxes
    for c_id, expected_pos in enumerate([True, True, False, False]):
        ov = overlap_depth(R, class_boxes(mod, [c_id]))
        assert bool((ov > 0.0)[0]) is expected_pos


# ---------------------------------------------------------------------------
# 6. Differentiability, chunking, edge cases, and static R1 / no-.item() check
# ---------------------------------------------------------------------------


def test_mixed_special_relations_and_inactive_floor() -> None:
    """Mixed batches with type_inv_rel / type_rel and normal relations match individual compilations; inactive -> -1e6."""
    mod = _make_translation_module(seed=606)
    class_of_entity = th.tensor([0, 1, 2, 3, -1, -1], dtype=th.long)
    entity_to_element = th.tensor([-1, -1, -1, -1, 10, 11], dtype=th.long)
    type_rel = 98
    type_inv_rel = 99

    q_mixed = [(1, (type_inv_rel,)), (4, (0,))]
    with th.no_grad():
        r_mixed = compile_query(
            mod,
            ("e", ("r",)),
            q_mixed,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
            entity_to_element=entity_to_element,
        )
        r_q0 = compile_query(
            mod,
            ("e", ("r",)),
            [q_mixed[0]],
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
            entity_to_element=entity_to_element,
        )
        r_q1 = compile_query(
            mod,
            ("e", ("r",)),
            [q_mixed[1]],
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
            entity_to_element=entity_to_element,
        )
        s_mixed = score_entities(mod, r_mixed)
        s_q0 = score_entities(mod, r_q0)
        s_q1 = score_entities(mod, r_q1)
    th.testing.assert_close(s_mixed[0:1], s_q0)
    th.testing.assert_close(s_mixed[1:2], s_q1)

    # All-inactive MultiBox returns -1e6 for both depth and overlap_depth
    empty_mb = MultiBox(
        cen=th.zeros((2, 2, 2)),
        w=th.zeros((2, 2, 2)),
        active=th.zeros((2, 2), dtype=th.bool),
    )
    pts = th.zeros((5, 2))
    dep_empty = depth(pts, empty_mb, crisp=True)
    assert dep_empty.shape == (2, 5)
    th.testing.assert_close(dep_empty, th.full((2, 5), -1e6))

    ov_empty = overlap_depth(empty_mb, class_boxes(mod, [0, 1]))
    assert ov_empty.shape == (2,)
    th.testing.assert_close(ov_empty.detach(), th.full((2,), -1e6))

    # Product-mode module is rejected by project / project_exact / compile_query
    prod_mod = GroundedELModule(nb_elements=6, nb_classes=2, nb_rels=2, embed_dim=2, role_mode="product")
    with pytest.raises(ValueError, match="translation"):
        compile_query(prod_mod, ("e", ("r",)), [(0, (0,))])


def test_differentiability_and_static_hygiene() -> None:
    """compile_query + depth/score_entities/score_concepts are differentiable and queries.py has no .item() or reasoner imports."""
    mod = _make_translation_module(seed=777)
    queries = [((0, (0,)), (1, (1,))), ((2, (1,)), (3, (2,)))]
    R = compile_query(mod, (("e", ("r",)), ("e", ("r",))), queries)
    ent_scores = score_entities(mod, R, chunk=8)
    con_scores = score_concepts(mod, R)
    smooth_dep = depth(mod.positions([4, 5]), R, crisp=False)

    loss = ent_scores.sum() + con_scores.sum() + smooth_dep.sum()
    loss.backward()
    assert mod.elem_pos.weight.grad is not None
    assert mod.role_centers.grad is not None
    assert mod.class_centers.grad is not None

    # Static AST check on mowl/nn/el/grounded/queries.py
    queries_path = (
        Path(__file__).resolve().parents[2]
        / "mowl"
        / "nn"
        / "el"
        / "grounded"
        / "queries.py"
    )
    tree = ast.parse(queries_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "item", "queries.py must not call .item()"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod_name = getattr(node, "module", "") or ""
            assert "completion" not in mod_name
            assert "reasoner" not in mod_name
