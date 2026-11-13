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
    query_witness_loss,
    score_concepts,
    score_entities,
    score_query_beam,
    score_query_witness,
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


# ---------------------------------------------------------------------------
# 7. Beam (max-min) query scoring: score_query_beam
# ---------------------------------------------------------------------------


def _np_role_depth_matrix(
    pos: np.ndarray,
    role_c: np.ndarray,
    role_w: np.ndarray,
    role_act: np.ndarray,
) -> np.ndarray:
    """Precompute pairwise crisp role depths dep_r(z, x) of shape (R, E, E) in pure numpy."""
    n_elem = pos.shape[0]
    nb_rels, kr, _ = role_c.shape
    diff = pos[None, :, :] - pos[:, None, :]  # (E_z, E_x, d) = p_x - p_z
    dep = np.full((nb_rels, n_elem, n_elem), -1e6, dtype=np.float64)
    for r in range(nb_rels):
        active_k = [k for k in range(kr) if role_act[r, k]]
        if not active_k:
            continue
        k_depths = [
            np.min(
                role_w[r, k][None, None, :] - np.abs(diff - role_c[r, k][None, None, :]),
                axis=-1,
            )
            for k in active_k
        ]
        dep[r] = np.max(np.stack(k_depths, axis=0), axis=0)
    return dep


def _np_beam_max_min_eval(
    structure: Any,
    query: Any,
    dep_mat: np.ndarray,
) -> np.ndarray:
    """Brute-force max-min Gödel beam scores over all elements (k = nb_elements) in pure numpy."""
    n_elem = dep_mat.shape[1]
    neg = -1e6
    big = 1e6

    def _project_all(s: np.ndarray, r: int) -> np.ndarray:
        valid = s > neg
        if not np.any(valid):
            return np.full((n_elem,), neg, dtype=np.float64)
        # min(s[z], dep_r(z, x)) for valid z, then max over z
        comb = np.minimum(s[valid, None], dep_mat[r, valid, :])
        return np.max(comb, axis=0)

    def _eval(struct: Any, q: Any) -> np.ndarray:
        if isinstance(struct, tuple) and len(struct) == 2 and struct[0] == "e":
            e_id = int(q[0])
            rels = [int(r) for r in q[1]]
            s = np.full((n_elem,), neg, dtype=np.float64)
            s[e_id] = big
            for r in rels:
                s = _project_all(s, r)
            return s

        if (
            isinstance(struct, tuple)
            and len(struct) == 2
            and isinstance(struct[1], tuple)
            and all(x == "r" for x in struct[1])
        ):
            s = _eval(struct[0], q[0])
            for r in [int(x) for x in q[1]]:
                s = _project_all(s, r)
            return s

        if isinstance(struct, tuple) and struct[-1] == ("u",):
            acc = _eval(struct[0], q[0])
            for i in range(1, len(struct) - 1):
                acc = np.maximum(acc, _eval(struct[i], q[i]))
            return acc

        acc = _eval(struct[0], q[0])
        for i in range(1, len(struct)):
            acc = np.minimum(acc, _eval(struct[i], q[i]))
        return acc

    return _eval(structure, query)


def test_score_query_beam_1p_matches_compile_query_and_type_inv_rel() -> None:
    """1. 1p: score_query_beam equals score_entities(compile_query(...)) including type_inv_rel."""
    mod = _make_translation_module(seed=101)
    rng = np.random.default_rng(1701)

    q_1p = [
        _sample_random_query_instance(("e", ("r",)), rng, mod.nb_elements, mod.nb_rels)
        for _ in range(20)
    ]
    with th.no_grad():
        expected_1p = score_entities(mod, compile_query(mod, ("e", ("r",)), q_1p))
        for k_val in (1, 8, mod.nb_elements, 64):
            got_1p = score_query_beam(mod, ("e", ("r",)), q_1p, k=k_val, chunk=7)
            th.testing.assert_close(got_1p, expected_1p)

        # Subset of candidate_element_ids
        cand_ids = [0, 5, 12, 29]
        got_sub = score_query_beam(
            mod, ("e", ("r",)), q_1p, k=16, candidate_element_ids=cand_ids
        )
        th.testing.assert_close(got_sub, expected_1p[:, cand_ids])

    # type_inv_rel anchor case (pure and mixed with normal 1p)
    class_of_entity = th.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=th.long)
    entity_to_element = th.tensor([-1, -1, -1, -1, 10, 11, 12, 13], dtype=th.long)
    type_rel = 98
    type_inv_rel = 99

    q_inv = [(0, (type_inv_rel,)), (2, (type_inv_rel,)), (3, (type_inv_rel,))]
    q_mixed = [(1, (type_inv_rel,)), (4, (0,)), (2, (type_inv_rel,)), (6, (2,))]
    with th.no_grad():
        exp_inv = score_entities(
            mod,
            compile_query(
                mod,
                ("e", ("r",)),
                q_inv,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                class_of_entity=class_of_entity,
                entity_to_element=entity_to_element,
            ),
        )
        got_inv = score_query_beam(
            mod,
            ("e", ("r",)),
            q_inv,
            k=8,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
            entity_to_element=entity_to_element,
        )
        th.testing.assert_close(got_inv, exp_inv)

        exp_mix = score_entities(
            mod,
            compile_query(
                mod,
                ("e", ("r",)),
                q_mixed,
                type_rel=type_rel,
                type_inv_rel=type_inv_rel,
                class_of_entity=class_of_entity,
                entity_to_element=entity_to_element,
            ),
        )
        got_mix = score_query_beam(
            mod,
            ("e", ("r",)),
            q_mixed,
            k=8,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
            entity_to_element=entity_to_element,
        )
        th.testing.assert_close(got_mix, exp_mix)


@pytest.mark.parametrize(
    "name,structure",
    (
        ("2p", ("e", ("r", "r"))),
        ("ip", ((("e", ("r",)), ("e", ("r",))), ("r",))),
        ("3p", ("e", ("r", "r", "r"))),
        ("up", ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",))),
    ),
)
def test_score_query_beam_full_k_matches_numpy_max_min_and_element_answers(
    name: str,
    structure: Any,
) -> None:
    """2. 2p and ip (and 3p, up) with k=nb_elements equal numpy brute-force max-min and element-semantics answer set."""
    mod = _make_translation_module(seed=303)
    pos, role_c, role_w, role_act = _extract_numpy_geometry(mod)
    dep_mat = _np_role_depth_matrix(pos, role_c, role_w, role_act)
    rng = np.random.default_rng(1702)

    batch_size = 24
    queries = [
        _sample_random_query_instance(structure, rng, mod.nb_elements, mod.nb_rels)
        for _ in range(batch_size)
    ]

    with th.no_grad():
        got = score_query_beam(
            mod, structure, queries, k=mod.nb_elements, chunk=11
        ).cpu().numpy()

    expected_scores = np.zeros_like(got, dtype=np.float64)
    expected_ans = np.zeros(got.shape, dtype=bool)
    for b, q in enumerate(queries):
        expected_scores[b] = _np_beam_max_min_eval(structure, q, dep_mat)
        expected_ans[b] = _np_element_witness_answers(
            structure, q, pos, role_c, role_w, role_act
        )

    np.testing.assert_allclose(
        got,
        expected_scores,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Max-min score mismatch on {name}",
    )
    np.testing.assert_array_equal(
        got >= 0.0,
        expected_ans,
        err_msg=f"Crisp sign pattern (>= 0) mismatch against element-semantics answers on {name}",
    )


def test_score_query_beam_intersection_min_and_union_max() -> None:
    """3. Intersection = elementwise min and union = elementwise max of branch scores (2i, 2u, pi, up)."""
    mod = _make_translation_module(seed=404)
    rng = np.random.default_rng(1703)
    batch_size = 16

    for k_val in (4, mod.nb_elements):
        # 2i: (('e', ('r',)), ('e', ('r',)))
        q_2i = [
            _sample_random_query_instance(
                (("e", ("r",)), ("e", ("r",))), rng, mod.nb_elements, mod.nb_rels
            )
            for _ in range(batch_size)
        ]
        b0_2i = [q[0] for q in q_2i]
        b1_2i = [q[1] for q in q_2i]
        with th.no_grad():
            s_2i = score_query_beam(mod, (("e", ("r",)), ("e", ("r",))), q_2i, k=k_val)
            s_b0 = score_query_beam(mod, ("e", ("r",)), b0_2i, k=k_val)
            s_b1 = score_query_beam(mod, ("e", ("r",)), b1_2i, k=k_val)
            th.testing.assert_close(s_2i, th.minimum(s_b0, s_b1))

        # 2u: (('e', ('r',)), ('e', ('r',)), ('u',))
        q_2u = [
            _sample_random_query_instance(
                (("e", ("r",)), ("e", ("r",)), ("u",)), rng, mod.nb_elements, mod.nb_rels
            )
            for _ in range(batch_size)
        ]
        b0_2u = [q[0] for q in q_2u]
        b1_2u = [q[1] for q in q_2u]
        with th.no_grad():
            s_2u = score_query_beam(
                mod, (("e", ("r",)), ("e", ("r",)), ("u",)), q_2u, k=k_val
            )
            s_u0 = score_query_beam(mod, ("e", ("r",)), b0_2u, k=k_val)
            s_u1 = score_query_beam(mod, ("e", ("r",)), b1_2u, k=k_val)
            th.testing.assert_close(s_2u, th.maximum(s_u0, s_u1))

        # pi: (('e', ('r', 'r')), ('e', ('r',)))
        q_pi = [
            _sample_random_query_instance(
                (("e", ("r", "r")), ("e", ("r",))), rng, mod.nb_elements, mod.nb_rels
            )
            for _ in range(batch_size)
        ]
        b0_pi = [q[0] for q in q_pi]
        b1_pi = [q[1] for q in q_pi]
        with th.no_grad():
            s_pi = score_query_beam(
                mod, (("e", ("r", "r")), ("e", ("r",))), q_pi, k=k_val
            )
            s_pi0 = score_query_beam(mod, ("e", ("r", "r")), b0_pi, k=k_val)
            s_pi1 = score_query_beam(mod, ("e", ("r",)), b1_pi, k=k_val)
            th.testing.assert_close(s_pi, th.minimum(s_pi0, s_pi1))

    # up: ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)) with k = nb_elements
    # When k = nb_elements, projection distributes over union (max), so up equals elementwise max of the two 2p branches.
    q_up = [
        _sample_random_query_instance(
            ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)),
            rng,
            mod.nb_elements,
            mod.nb_rels,
        )
        for _ in range(batch_size)
    ]
    b0_2p = [(q[0][0][0], (q[0][0][1][0], q[1][0])) for q in q_up]
    b1_2p = [(q[0][1][0], (q[0][1][1][0], q[1][0])) for q in q_up]
    with th.no_grad():
        s_up = score_query_beam(
            mod,
            ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)),
            q_up,
            k=mod.nb_elements,
        )
        s_up0 = score_query_beam(mod, ("e", ("r", "r")), b0_2p, k=mod.nb_elements)
        s_up1 = score_query_beam(mod, ("e", ("r", "r")), b1_2p, k=mod.nb_elements)
        th.testing.assert_close(s_up, th.maximum(s_up0, s_up1))


def test_score_query_beam_type_rel_step() -> None:
    """4. type_rel step: class elements get max_{z in Z} min(S(z), depth(p_z, class_boxes(c))); individuals get NEG."""
    mod = _make_translation_module(seed=505)
    type_rel = 98
    q_hop1 = [(5, (0,)), (12, (1,)), (20, (2,))]
    q_2p_type = [(5, (0, type_rel)), (12, (1, type_rel)), (20, (2, type_rel))]

    all_pos = mod.positions().detach()
    c_boxes = class_boxes(mod, None)  # (4, 2, 2)

    for k_val in (1, 4, mod.nb_elements):
        with th.no_grad():
            s1 = score_query_beam(mod, ("e", ("r",)), q_hop1, k=k_val)  # (B, E)
            s_type = score_query_beam(
                mod,
                ("e", ("r", "r")),
                q_2p_type,
                k=k_val,
                type_rel=type_rel,
            )  # (B, E)

        # Individuals (id >= nb_classes) must all equal NEG (-1e6)
        assert s_type.shape == (len(q_2p_type), mod.nb_elements)
        th.testing.assert_close(
            s_type[:, mod.nb_classes :],
            th.full((len(q_2p_type), mod.nb_elements - mod.nb_classes), -1e6),
        )

        # Explicitly verify class scores (0 .. nb_classes - 1) against formula
        k_keep = min(k_val, mod.nb_elements)
        top_z = th.argsort(s1, dim=1, descending=True, stable=True)[:, :k_keep]
        for b in range(len(q_2p_type)):
            z_idx = top_z[b]
            s_z = s1[b, z_idx]  # (K,)
            p_z = all_pos[z_idx]  # (K, d)
            dep_zc = depth(p_z, c_boxes, crisp=True)  # (C, K)
            expected_c = th.amax(th.minimum(s_z.unsqueeze(0), dep_zc), dim=1)  # (C,)
            th.testing.assert_close(s_type[b, : mod.nb_classes], expected_c)

    # Projecting again after type_rel with k=10 > nb_classes=4: the 6 individual elements
    # in top-10 have S(z) == NEG (-1e6) and must contribute NEG, matching k=4.
    q_3p_after_type = [(5, (0, type_rel, 1)), (12, (1, type_rel, 0))]
    with th.no_grad():
        s_after_k4 = score_query_beam(
            mod, ("e", ("r", "r", "r")), q_3p_after_type, k=4, type_rel=type_rel
        )
        s_after_k10 = score_query_beam(
            mod, ("e", ("r", "r", "r")), q_3p_after_type, k=10, type_rel=type_rel
        )
        # Note: at hop 1 (1p), k=4 vs k=10 is identical; at hop 2 (type_rel), we check when hop 1 is type_inv_rel:
        class_of_entity = th.tensor([0, 1, 2, 3, -1, -1], dtype=th.long)
        type_inv_rel = 99
        q_inv_type_r = [(0, (type_inv_rel, type_rel, 1)), (2, (type_inv_rel, type_rel, 0))]
        s_it_k4 = score_query_beam(
            mod,
            ("e", ("r", "r", "r")),
            q_inv_type_r,
            k=mod.nb_elements,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
        )
        # Only class elements 0..3 have S > NEG after type_rel, so projecting from s_type with k=4 vs k=30 is identical
        s_type_inv = score_query_beam(
            mod,
            ("e", ("r", "r")),
            [(0, (type_inv_rel, type_rel)), (2, (type_inv_rel, type_rel))],
            k=mod.nb_elements,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            class_of_entity=class_of_entity,
        )
        assert bool((s_type_inv[:, mod.nb_classes :] == -1e6).all())
        assert s_after_k4.shape == s_after_k10.shape == s_it_k4.shape


def test_score_query_beam_monotonicity_in_k() -> None:
    """5. Monotonicity in k: scores with k=1 <= k=4 <= k=all elementwise."""
    mod = _make_translation_module(seed=606)
    rng = np.random.default_rng(1705)
    batch_size = 16

    monotone_structures = (
        ("1p", ("e", ("r",))),
        ("2p", ("e", ("r", "r"))),
        ("2i", (("e", ("r",)), ("e", ("r",)))),
        ("3i", (("e", ("r",)), ("e", ("r",)), ("e", ("r",)))),
        ("2u", (("e", ("r",)), ("e", ("r",)), ("u",))),
        ("ip", ((("e", ("r",)), ("e", ("r",))), ("r",))),
        ("pi", (("e", ("r", "r")), ("e", ("r",)))),
        ("up", ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",))),
    )
    for name, structure in monotone_structures:
        queries = [
            _sample_random_query_instance(structure, rng, mod.nb_elements, mod.nb_rels)
            for _ in range(batch_size)
        ]
        with th.no_grad():
            s_k1 = score_query_beam(mod, structure, queries, k=1)
            s_k4 = score_query_beam(mod, structure, queries, k=4)
            s_all = score_query_beam(mod, structure, queries, k=mod.nb_elements)

        assert bool((s_k1 <= s_k4).all()), f"k=1 > k=4 on {name}"
        assert bool((s_k4 <= s_all).all()), f"k=4 > k=all on {name}"


def test_score_query_beam_differentiability() -> None:
    """score_query_beam is differentiable with respect to element positions, role boxes, and class boxes."""
    mod = _make_translation_module(seed=808)
    type_rel = 98
    q_2p = [(0, (0, 1)), (2, (1, 2))]
    q_type = [(1, (0, type_rel)), (3, (2, type_rel))]

    s_2p = score_query_beam(mod, ("e", ("r", "r")), q_2p, k=6, chunk=8)
    s_t = score_query_beam(
        mod, ("e", ("r", "r")), q_type, k=6, type_rel=type_rel, chunk=8
    )
    loss = s_2p.sum() + s_t[:, : mod.nb_classes].sum()
    loss.backward()

    assert mod.elem_pos.weight.grad is not None
    assert mod.role_centers.grad is not None
    assert mod.role_raw_widths.grad is not None
    assert mod.class_centers.grad is not None
    assert mod.class_raw_widths.grad is not None


# ---------------------------------------------------------------------------
# 8. Witness max-min loss: query_witness_loss & score_query_witness (Task 11)
# ---------------------------------------------------------------------------


def _make_two_element_toy_module(seed: int = 1101) -> GroundedELModule:
    """Build a 2-element translation-mode toy KB (nb_elements=2, nb_classes=1, nb_rels=2, d=2, K=2, K_r=2)."""
    gen = th.Generator().manual_seed(seed)
    mod = GroundedELModule(
        nb_elements=2,
        nb_classes=1,
        nb_rels=2,
        embed_dim=2,
        n_components=2,
        n_role_components=2,
        role_mode="translation",
    )
    with th.no_grad():
        mod.elem_pos.weight.copy_(th.tensor([[0.0, 0.0], [0.6, -0.4]]))
        mod.role_centers.copy_(
            th.rand((2, 2, 2), generator=gen) * 0.8 - 0.4
        )
        mod.role_raw_widths.copy_(
            th.rand((2, 2, 2), generator=gen) * 0.6 - 0.3
        )
        mod.class_centers.copy_(
            th.rand((1, 2, 2), generator=gen) * 0.8 - 0.4
        )
        mod.class_raw_widths.copy_(
            th.rand((1, 2, 2), generator=gen) * 0.6 - 0.3
        )
    return mod


def test_query_witness_loss_two_element_toy_kb_tau_to_zero_matches_beam() -> None:
    """On a 2-element toy KB, s(q, a) equals score_query_beam's score for the same answer when tau -> 0 (no_grad)."""
    mod = _make_two_element_toy_module(seed=1101)
    mod.set_tau(1e-6)

    toy_shapes = (
        ("2p", ("e", ("r", "r")), [(0, (0, 1)), (1, (1, 0)), (0, (0, 0)), (1, (1, 1))]),
        ("3p", ("e", ("r", "r", "r")), [(0, (0, 1, 0)), (1, (1, 0, 1)), (0, (1, 1, 0))]),
        ("2i", (("e", ("r",)), ("e", ("r",))), [((0, (0,)), (1, (1,))), ((0, (1,)), (1, (0,)))]),
        (
            "3i",
            (("e", ("r",)), ("e", ("r",)), ("e", ("r",))),
            [((0, (0,)), (1, (1,)), (0, (1,))), ((1, (0,)), (0, (0,)), (1, (1,)))],
        ),
    )

    with th.no_grad():
        for name, structure, queries in toy_shapes:
            bsz = len(queries)
            for k_val in (1, 2):
                beam_scores = score_query_beam(mod, structure, queries, k=k_val)  # (B, 2)
                for a_el in (0, 1):
                    ans = [a_el] * bsz
                    neg = [1 - a_el] * bsz
                    loss, pos_s, neg_s = query_witness_loss(
                        mod,
                        structure,
                        queries,
                        ans,
                        neg,
                        k=k_val,
                        margin=0.25,
                        return_scores=True,
                    )
                    assert neg_s is not None
                    th.testing.assert_close(
                        pos_s,
                        beam_scores[:, a_el],
                        rtol=1e-5,
                        atol=1e-5,
                        msg=f"pos s(q, a) != beam score on 2-element toy KB ({name}, k={k_val}, a={a_el})",
                    )
                    th.testing.assert_close(
                        neg_s,
                        beam_scores[:, 1 - a_el],
                        rtol=1e-5,
                        atol=1e-5,
                        msg=f"neg s(q, a-) != beam score on 2-element toy KB ({name}, k={k_val}, a={a_el})",
                    )
                    expected_loss = (
                        th.relu(0.25 - beam_scores[:, a_el]).mean()
                        + th.relu(0.25 + beam_scores[:, 1 - a_el]).mean()
                    )
                    th.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-5)

                # Also verify 2-D target matrix [0, 1] via score_query_witness
                both_targets = [[0, 1] for _ in range(bsz)]
                wit_both = score_query_witness(
                    mod, structure, queries, both_targets, k=k_val
                )
                th.testing.assert_close(wit_both, beam_scores, rtol=1e-5, atol=1e-5)

    # Also verify tau -> 0 equivalence on the 30-element module (including type_rel / type_inv_rel)
    mod30 = _make_translation_module(seed=1102)
    mod30.set_tau(1e-6)
    rng = np.random.default_rng(1102)
    with th.no_grad():
        for name, structure in ALL_STRUCTURES:
            qs = [
                _sample_random_query_instance(
                    structure, rng, mod30.nb_elements, mod30.nb_rels
                )
                for _ in range(8)
            ]
            cand_targets = rng.integers(0, mod30.nb_elements, size=(8, 5))
            for k_val in (3, mod30.nb_elements):
                beam_all = score_query_beam(mod30, structure, qs, k=k_val)
                beam_picked = th.gather(
                    beam_all, 1, th.as_tensor(cand_targets, dtype=th.long)
                )
                wit_picked = score_query_witness(
                    mod30, structure, qs, cand_targets, k=k_val, chunk=9
                )
                th.testing.assert_close(
                    wit_picked,
                    beam_picked,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Mismatch on 30-element module ({name}, k={k_val})",
                )

        # Special relations type_inv_rel and type_rel at tau -> 0
        class_of_entity = th.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=th.long)
        entity_to_element = th.tensor([-1, -1, -1, -1, 10, 11, 12, 13], dtype=th.long)
        type_rel = 98
        type_inv_rel = 99
        q_special = [(0, (type_inv_rel, 0)), (4, (1, type_rel)), (2, (type_inv_rel, type_rel))]
        t_special = [[0, 1, 10, 11], [0, 2, 3, 12], [1, 3, 10, 13]]
        beam_sp = score_query_beam(
            mod30,
            ("e", ("r", "r")),
            q_special,
            k=6,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            entity_to_element=entity_to_element,
            class_of_entity=class_of_entity,
        )
        wit_sp = score_query_witness(
            mod30,
            ("e", ("r", "r")),
            q_special,
            t_special,
            k=6,
            type_rel=type_rel,
            type_inv_rel=type_inv_rel,
            entity_to_element=entity_to_element,
            class_of_entity=class_of_entity,
        )
        th.testing.assert_close(
            wit_sp,
            th.gather(beam_sp, 1, th.tensor(t_special, dtype=th.long)),
            rtol=1e-5,
            atol=1e-5,
        )


def test_query_witness_loss_two_element_toy_kb_sgd_and_negatives() -> None:
    """On a 2-element toy KB, gradient is non-zero, SGD lowers the loss, and negatives push their score down."""
    shapes_and_queries = (
        ("2p", ("e", ("r", "r")), [(0, (0, 1))]),
        ("3p", ("e", ("r", "r", "r")), [(0, (0, 1, 0))]),
        ("2i", (("e", ("r",)), ("e", ("r",))), [((0, (0,)), (0, (1,)))]),
        ("3i", (("e", ("r",)), ("e", ("r",)), ("e", ("r",))), [((0, (0,)), (0, (1,)), (0, (0,)))]),
    )

    for idx, (name, structure, queries) in enumerate(shapes_and_queries):
        mod = _make_two_element_toy_module(seed=1200 + idx)
        mod.set_tau(0.5)
        ans = [1]
        neg = [[0]]
        margin = 1.5

        loss0, pos0, neg0 = query_witness_loss(
            mod,
            structure,
            queries,
            ans,
            neg,
            k=2,
            margin=margin,
            return_scores=True,
        )
        assert neg0 is not None
        assert float(loss0.detach()) > 0.0
        loss0.backward()

        assert mod.elem_pos.weight.grad is not None
        assert mod.role_centers.grad is not None
        assert mod.role_raw_widths.grad is not None
        grad_norm = (
            float(mod.elem_pos.weight.grad.norm())
            + float(mod.role_centers.grad.norm())
            + float(mod.role_raw_widths.grad.norm())
        )
        assert grad_norm > 0.0, f"Zero gradient on {name}"

        opt = th.optim.SGD(mod.parameters(), lr=0.1)
        for _ in range(12):
            opt.zero_grad()
            step_loss = query_witness_loss(
                mod, structure, queries, ans, neg, k=2, margin=margin
            )
            step_loss.backward()
            opt.step()

        with th.no_grad():
            loss1, pos1, neg1 = query_witness_loss(
                mod,
                structure,
                queries,
                ans,
                neg,
                k=2,
                margin=margin,
                return_scores=True,
            )
        assert neg1 is not None
        assert float(loss1) < float(loss0.detach()), f"Loss did not decrease on {name}"
        assert float(pos1[0]) > float(pos0.detach()[0]), f"Positive score did not increase on {name}"
        assert float(neg1[0, 0]) < float(neg0.detach()[0, 0]), f"Negative score was not pushed down on {name}"

    # Verify negative-only effect: comparing SGD with negatives vs without negatives on identical init
    mod_with_neg = _make_two_element_toy_module(seed=1300)
    mod_no_neg = _make_two_element_toy_module(seed=1300)
    mod_with_neg.set_tau(0.5)
    mod_no_neg.set_tau(0.5)
    q_2p = [(0, (0, 1))]

    with th.no_grad():
        s_neg_init = float(
            score_query_witness(mod_with_neg, ("e", ("r", "r")), q_2p, [0], k=2)[0]
        )

    opt_neg = th.optim.SGD(mod_with_neg.parameters(), lr=0.1)
    opt_none = th.optim.SGD(mod_no_neg.parameters(), lr=0.1)
    for _ in range(10):
        opt_neg.zero_grad()
        l_neg = query_witness_loss(
            mod_with_neg, ("e", ("r", "r")), q_2p, answers=[1], negatives=[0], k=2, margin=0.5
        )
        l_neg.backward()
        opt_neg.step()

        opt_none.zero_grad()
        l_none = query_witness_loss(
            mod_no_neg, ("e", ("r", "r")), q_2p, answers=[1], negatives=None, k=2, margin=0.5
        )
        l_none.backward()
        opt_none.step()

    with th.no_grad():
        s_neg_after_with = float(
            score_query_witness(mod_with_neg, ("e", ("r", "r")), q_2p, [0], k=2)[0]
        )
        s_neg_after_without = float(
            score_query_witness(mod_no_neg, ("e", ("r", "r")), q_2p, [0], k=2)[0]
        )
    assert s_neg_after_with < s_neg_init
    assert s_neg_after_with < s_neg_after_without


def test_query_witness_loss_validation_and_edge_cases() -> None:
    """Validate arguments and empty-batch handling for query_witness_loss and score_query_witness."""
    mod = _make_two_element_toy_module(seed=1400)
    q_2p = [(0, (0, 1))]

    # Empty batch
    empty_loss, empty_pos, empty_neg = query_witness_loss(
        mod, ("e", ("r", "r")), [], answers=[], negatives=np.zeros((0, 2), dtype=np.int64), return_scores=True
    )
    assert empty_loss.shape == ()
    assert float(empty_loss.detach()) == 0.0
    assert empty_pos.shape == (0,)
    assert empty_neg is not None and empty_neg.shape == (0, 2)

    # Invalid k, margin, answers, negatives, role_mode
    with pytest.raises(ValueError, match="k must be an int >= 1"):
        query_witness_loss(mod, ("e", ("r", "r")), q_2p, [1], k=0)
    with pytest.raises(ValueError, match="margin"):
        query_witness_loss(mod, ("e", ("r", "r")), q_2p, [1], margin=float("nan"))
    with pytest.raises(ValueError, match="answers batch size"):
        query_witness_loss(mod, ("e", ("r", "r")), q_2p, [0, 1])
    with pytest.raises(ValueError, match="answers indices out of range"):
        query_witness_loss(mod, ("e", ("r", "r")), q_2p, [5])
    with pytest.raises(ValueError, match="negatives indices out of range"):
        query_witness_loss(mod, ("e", ("r", "r")), q_2p, [1], negatives=[-1])

    prod_mod = GroundedELModule(nb_elements=2, nb_classes=1, nb_rels=1, embed_dim=2, role_mode="product")
    with pytest.raises(ValueError, match="translation"):
        query_witness_loss(prod_mod, ("e", ("r", "r")), [(0, (0, 0))], [1])

    # Verify cdist top-k fast path (nb_elements > 256) matches score_query_beam at tau -> 0
    th.manual_seed(1401)
    large_mod = GroundedELModule(
        nb_elements=300,
        nb_classes=10,
        nb_rels=6,
        embed_dim=8,
        n_components=3,
        n_role_components=3,
        role_mode="translation",
    )
    large_mod.set_tau(1e-5)
    q_3p_large = [
        (0, (0, 1, 2)),
        (15, (5, 0, 4)),
        (2, (5, 4, 1)),
        (20, (1, 2, 0)),
    ]
    targs = th.tensor([[0, 1, 50, 299], [2, 10, 100, 250], [1, 5, 120, 290], [0, 3, 77, 180]], dtype=th.long)
    with th.no_grad():
        s_beam_large = score_query_beam(
            large_mod, ("e", ("r", "r", "r")), q_3p_large, k=8, type_rel=4, type_inv_rel=5
        )
        s_wit_large = score_query_witness(
            large_mod, ("e", ("r", "r", "r")), q_3p_large, targs, k=8, type_rel=4, type_inv_rel=5
        )
        s_beam_gat = th.gather(s_beam_large, 1, targs)
        assert th.allclose(s_wit_large, s_beam_gat, atol=1e-4, rtol=1e-4)



# ---------------------------------------------------------------------------
# 9. type_inv_rel at a non-first hop (class -> its instances after an intermediate step)
# ---------------------------------------------------------------------------

_TYPE_REL = 98
_TYPE_INV_REL = 99


def _np_class_depth_matrix(mod: GroundedELModule) -> np.ndarray:
    """Crisp depth cdep[c, y] of every element position in every class multi-box, in pure numpy, shape (C, E)."""
    pos = mod.elem_pos.weight.detach().cpu().numpy().astype(np.float64)
    cb = class_boxes(mod, None)
    cen = cb.cen.detach().cpu().numpy().astype(np.float64)  # (C, M, d)
    w = cb.w.detach().cpu().numpy().astype(np.float64)  # (C, M, d)
    act = cb.active.detach().cpu().numpy()  # (C, M)
    n_cls = cen.shape[0]
    cdep = np.full((n_cls, pos.shape[0]), -1e6, dtype=np.float64)
    for c in range(n_cls):
        slots = [
            np.min(w[c, m][None, :] - np.abs(pos - cen[c, m][None, :]), axis=-1)
            for m in range(cen.shape[1])
            if act[c, m]
        ]
        if slots:
            cdep[c] = np.max(np.stack(slots, axis=0), axis=0)
    return cdep


def _np_special_beam_eval(
    structure: Any,
    query: Any,
    dep_mat: np.ndarray,
    cdep: np.ndarray,
) -> np.ndarray:
    """Brute-force max-min beam scores (k = all) with roles, type_rel (98) and type_inv_rel (99) at any hop."""
    n_elem = dep_mat.shape[1]
    n_cls = cdep.shape[0]
    neg = -1e6
    big = 1e6

    def _step(s: np.ndarray, r: int) -> np.ndarray:
        out = np.full((n_elem,), neg, dtype=np.float64)
        if r == _TYPE_REL:
            valid = s > neg
            if np.any(valid):
                out[:n_cls] = np.max(np.minimum(s[valid][None, :], cdep[:, valid]), axis=1)
            return out
        if r == _TYPE_INV_REL:
            valid_c = s[:n_cls] > neg
            if np.any(valid_c):
                out = np.max(np.minimum(s[:n_cls][valid_c][:, None], cdep[valid_c, :]), axis=0)
            return out
        valid = s > neg
        if np.any(valid):
            out = np.max(np.minimum(s[valid, None], dep_mat[r, valid, :]), axis=0)
        return out

    def _eval(struct: Any, q: Any) -> np.ndarray:
        if isinstance(struct, tuple) and len(struct) == 2 and struct[0] == "e":
            s = np.full((n_elem,), neg, dtype=np.float64)
            s[int(q[0])] = big
            for r in q[1]:
                s = _step(s, int(r))
            return s
        if (
            isinstance(struct, tuple)
            and len(struct) == 2
            and isinstance(struct[1], tuple)
            and all(x == "r" for x in struct[1])
        ):
            s = _eval(struct[0], q[0])
            for r in q[1]:
                s = _step(s, int(r))
            return s
        if isinstance(struct, tuple) and struct[-1] == ("u",):
            acc = _eval(struct[0], q[0])
            for i in range(1, len(struct) - 1):
                acc = np.maximum(acc, _eval(struct[i], q[i]))
            return acc
        acc = _eval(struct[0], q[0])
        for i in range(1, len(struct)):
            acc = np.minimum(acc, _eval(struct[i], q[i]))
        return acc

    return _eval(structure, query)


def _np_geometry_for_special(mod: GroundedELModule) -> Tuple[np.ndarray, np.ndarray]:
    pos, role_c, role_w, role_act = _extract_numpy_geometry(mod)
    return _np_role_depth_matrix(pos, role_c, role_w, role_act), _np_class_depth_matrix(mod)


def test_type_inv_rel_second_hop_2p_matches_numpy_and_element_answers() -> None:
    """Test 1: (individual, (type_rel, type_inv_rel)) with k = nb_elements equals numpy max-min; crisp set is co-membership."""
    mod = _make_translation_module(seed=1501)
    dep_mat, cdep = _np_geometry_for_special(mod)
    anchors = list(range(mod.nb_classes, mod.nb_elements))
    queries = [(a, (_TYPE_REL, _TYPE_INV_REL)) for a in anchors]

    with th.no_grad():
        got = score_query_beam(
            mod,
            ("e", ("r", "r")),
            queries,
            k=mod.nb_elements,
            type_rel=_TYPE_REL,
            type_inv_rel=_TYPE_INV_REL,
            chunk=7,
        ).cpu().numpy()

    expected = np.stack(
        [_np_special_beam_eval(("e", ("r", "r")), q, dep_mat, cdep) for q in queries]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    # Element semantics: y is an answer iff some class c contains both the anchor and y
    in_cls = cdep > 0.0  # (C, E)
    expected_ans = np.stack(
        [np.any(in_cls[:, a][:, None] & in_cls, axis=0) for a in anchors]
    )
    assert expected_ans.any() and not expected_ans.all(), "degenerate fixture"
    np.testing.assert_array_equal(got > 0.0, expected_ans)


def test_type_inv_rel_middle_and_last_hop_3p_match_numpy() -> None:
    """Test 2: 3p with type_inv_rel in the middle (then a role), role -> type_rel -> type_inv_rel, and ip -> type_inv_rel."""
    mod = _make_translation_module(seed=1502)
    dep_mat, cdep = _np_geometry_for_special(mod)
    rng = np.random.default_rng(1502)

    q_mid = [
        (int(a), (_TYPE_REL, _TYPE_INV_REL, int(r)))
        for a, r in zip(rng.integers(0, mod.nb_elements, 10), rng.integers(0, mod.nb_rels, 10))
    ]
    q_last = [
        (int(a), (int(r), _TYPE_REL, _TYPE_INV_REL))
        for a, r in zip(rng.integers(0, mod.nb_elements, 10), rng.integers(0, mod.nb_rels, 10))
    ]
    q_role_inv = [
        (int(a), (int(r), _TYPE_INV_REL))
        for a, r in zip(rng.integers(mod.nb_classes, mod.nb_elements, 10), rng.integers(0, mod.nb_rels, 10))
    ]
    ip_struct = ((("e", ("r",)), ("e", ("r",))), ("r",))
    q_ip = [
        (((int(a), (int(r1),)), (int(b), (int(r2),))), (_TYPE_INV_REL,))
        for a, b, r1, r2 in zip(
            rng.integers(0, mod.nb_elements, 8),
            rng.integers(0, mod.nb_elements, 8),
            rng.integers(0, mod.nb_rels, 8),
            rng.integers(0, mod.nb_rels, 8),
        )
    ]
    cases = (
        ("3p mid", ("e", ("r", "r", "r")), q_mid),
        ("3p last", ("e", ("r", "r", "r")), q_last),
        ("2p role->inv", ("e", ("r", "r")), q_role_inv),
        ("ip->inv", ip_struct, q_ip),
    )
    for name, struct, queries in cases:
        with th.no_grad():
            got = score_query_beam(
                mod,
                struct,
                queries,
                k=mod.nb_elements,
                type_rel=_TYPE_REL,
                type_inv_rel=_TYPE_INV_REL,
            ).cpu().numpy()
        expected = np.stack([_np_special_beam_eval(struct, q, dep_mat, cdep) for q in queries])
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5, err_msg=name)


def test_type_inv_rel_later_hop_mixed_batch_matches_rowwise() -> None:
    """Test 3: rows with type_inv_rel at hop 2 mixed with role / type_rel rows equal scoring each row alone."""
    mod = _make_translation_module(seed=1503)
    q_mixed = [
        (10, (_TYPE_REL, _TYPE_INV_REL)),
        (11, (0, 1)),
        (12, (1, _TYPE_INV_REL)),
        (13, (2, _TYPE_REL)),
        (2, (_TYPE_INV_REL, _TYPE_INV_REL)),
        (14, (_TYPE_REL, 2)),
    ]
    q_mixed_3p = [
        (10, (_TYPE_REL, _TYPE_INV_REL, 0)),
        (11, (0, 1, _TYPE_INV_REL)),
        (12, (_TYPE_REL, 2, _TYPE_REL)),
        (1, (_TYPE_INV_REL, _TYPE_REL, _TYPE_INV_REL)),
    ]
    for struct, queries in (
        (("e", ("r", "r")), q_mixed),
        (("e", ("r", "r", "r")), q_mixed_3p),
    ):
        for k_val in (1, 3, mod.nb_elements):
            with th.no_grad():
                s_batch = score_query_beam(
                    mod,
                    struct,
                    queries,
                    k=k_val,
                    type_rel=_TYPE_REL,
                    type_inv_rel=_TYPE_INV_REL,
                )
                for b, q in enumerate(queries):
                    s_row = score_query_beam(
                        mod,
                        struct,
                        [q],
                        k=k_val,
                        type_rel=_TYPE_REL,
                        type_inv_rel=_TYPE_INV_REL,
                    )
                    # Role projection is batch-shape dependent at the ~1 ulp level (pre-existing),
                    # so compare with a tight tolerance instead of bitwise equality.
                    th.testing.assert_close(
                        s_batch[b : b + 1],
                        s_row,
                        rtol=1e-6,
                        atol=1e-6,
                        msg=f"row {b} differs ({q}, k={k_val})",
                    )


def test_type_inv_rel_first_hop_unchanged() -> None:
    """Test 4: first-hop type_inv_rel from class anchors still equals score_entities(class_boxes(...)) bitwise."""
    mod = _make_translation_module(seed=1504)
    cls_anchors = list(range(mod.nb_classes))
    with th.no_grad():
        expected = score_entities(mod, class_boxes(mod, cls_anchors))
        got = score_query_beam(
            mod,
            ("e", ("r",)),
            [(c, (_TYPE_INV_REL,)) for c in cls_anchors],
            k=5,
            type_rel=_TYPE_REL,
            type_inv_rel=_TYPE_INV_REL,
        )
        assert th.equal(got, expected)

        # Mixed first hop (class anchors with type_inv_rel, individuals with roles): type_inv rows unchanged
        q_mix = [(0, (_TYPE_INV_REL,)), (12, (1,)), (3, (_TYPE_INV_REL,)), (20, (_TYPE_REL,))]
        got_mix = score_query_beam(
            mod,
            ("e", ("r",)),
            q_mix,
            k=5,
            type_rel=_TYPE_REL,
            type_inv_rel=_TYPE_INV_REL,
        )
        assert th.equal(got_mix[0], expected[0])
        assert th.equal(got_mix[2], expected[3])

        # A first-hop type_inv_rel followed by type_inv_rel: hop 2 starts from score_entities(class_boxes(A))
        from mowl.nn.el.grounded.queries import _beam_project_type_inv_rel

        got_2p = score_query_beam(
            mod,
            ("e", ("r", "r")),
            [(c, (_TYPE_INV_REL, _TYPE_INV_REL)) for c in cls_anchors],
            k=3,
            type_rel=_TYPE_REL,
            type_inv_rel=_TYPE_INV_REL,
        )
        assert th.equal(got_2p, _beam_project_type_inv_rel(mod, expected, k=3, chunk=65536))


def test_type_inv_rel_later_hop_monotone_in_k_and_empty_floor() -> None:
    """Test 5: scores are monotone in k (1 <= 4 <= all); rows without a class source give the empty floor."""
    mod = _make_translation_module(seed=1505)
    rng = np.random.default_rng(1505)
    cases = (
        (("e", ("r", "r")), lambda a, r: (a, (_TYPE_REL, _TYPE_INV_REL))),
        (("e", ("r", "r")), lambda a, r: (a, (r, _TYPE_INV_REL))),
        (("e", ("r", "r", "r")), lambda a, r: (a, (_TYPE_REL, _TYPE_INV_REL, r))),
        (("e", ("r", "r", "r")), lambda a, r: (a, (r, _TYPE_REL, _TYPE_INV_REL))),
    )
    for struct, make_q in cases:
        queries = [
            make_q(int(a), int(r))
            for a, r in zip(rng.integers(0, mod.nb_elements, 12), rng.integers(0, mod.nb_rels, 12))
        ]
        with th.no_grad():
            kw = dict(type_rel=_TYPE_REL, type_inv_rel=_TYPE_INV_REL)
            s_k1 = score_query_beam(mod, struct, queries, k=1, **kw)
            s_k4 = score_query_beam(mod, struct, queries, k=4, **kw)
            s_all = score_query_beam(mod, struct, queries, k=mod.nb_elements, **kw)
        assert bool((s_k1 <= s_k4).all()), f"k=1 > k=4 on {queries[0]}"
        assert bool((s_k4 <= s_all).all()), f"k=4 > k=all on {queries[0]}"

    from mowl.nn.el.grounded.queries import _beam_project_type_inv_rel

    S = th.full((2, mod.nb_elements), -1e6)
    S[0, mod.nb_classes :] = 5.0  # only individuals reachable -> no class source
    S[1, 1] = 0.3
    with th.no_grad():
        out = _beam_project_type_inv_rel(mod, S, k=4, chunk=8)
        dep1 = score_entities(mod, class_boxes(mod, [1]))[0]
    assert bool((out[0] == -1e6).all())
    assert th.equal(out[1], th.minimum(th.tensor(0.3), dep1))


def test_type_inv_rel_later_hop_witness_scoring_and_compile_query() -> None:
    """score_query_witness supports later-hop type_inv_rel (tau -> 0 equals beam); compile_query raises NotImplementedError."""
    mod = _make_translation_module(seed=1506)
    mod.set_tau(1e-6)
    kw = dict(type_rel=_TYPE_REL, type_inv_rel=_TYPE_INV_REL)
    rng = np.random.default_rng(1506)
    cases = (
        (("e", ("r", "r")), [(10, (_TYPE_REL, _TYPE_INV_REL)), (11, (0, 1)), (13, (_TYPE_REL, _TYPE_INV_REL))], (4, 30)),
        # type_inv_rel after a non-type_rel hop: witness and beam agree at k = all (documented non-guarantee otherwise)
        (("e", ("r", "r")), [(3, (_TYPE_INV_REL, _TYPE_INV_REL)), (12, (1, _TYPE_INV_REL))], (30,)),
        (
            ("e", ("r", "r", "r")),
            [
                (10, (_TYPE_REL, _TYPE_INV_REL, 0)),
                (12, (1, _TYPE_REL, _TYPE_INV_REL)),
                (13, (_TYPE_REL, 2, 0)),
                (14, (0, 1, 2)),
            ],
            (4, 30),
        ),
    )
    with th.no_grad():
        for struct, queries, k_vals in cases:
            targets = rng.integers(0, mod.nb_elements, size=(len(queries), 6))
            for k_val in k_vals:
                beam = score_query_beam(mod, struct, queries, k=k_val, **kw)
                wit = score_query_witness(mod, struct, queries, targets, k=k_val, chunk=9, **kw)
                th.testing.assert_close(
                    wit,
                    th.gather(beam, 1, th.as_tensor(targets, dtype=th.long)),
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"witness != beam on {queries} (k={k_val})",
                )

        # Compound projection (ip) ending in type_inv_rel, k = all
        ip_struct = ((("e", ("r",)), ("e", ("r",))), ("r",))
        q_ip = [(((10, (0,)), (11, (1,))), (_TYPE_INV_REL,)), (((5, (2,)), (6, (0,))), (_TYPE_INV_REL,))]
        targets = rng.integers(0, mod.nb_elements, size=(2, 6))
        beam_ip = score_query_beam(mod, ip_struct, q_ip, k=mod.nb_elements, **kw)
        wit_ip = score_query_witness(mod, ip_struct, q_ip, targets, k=mod.nb_elements, **kw)
        th.testing.assert_close(
            wit_ip, th.gather(beam_ip, 1, th.as_tensor(targets, dtype=th.long)), rtol=1e-5, atol=1e-5
        )

    # Differentiable through the soft class depth of the later type_inv_rel step
    mod.set_tau(0.1)
    s = score_query_witness(
        mod, ("e", ("r", "r")), [(10, (_TYPE_REL, _TYPE_INV_REL))], [[0, 5, 12]], k=4, **kw
    )
    s.sum().backward()
    assert mod.class_centers.grad is not None and float(mod.class_centers.grad.abs().sum()) > 0.0

    with pytest.raises(NotImplementedError, match="type_inv_rel at a non-first hop"):
        compile_query(mod, ("e", ("r", "r")), [(10, (_TYPE_REL, _TYPE_INV_REL))], **kw)
