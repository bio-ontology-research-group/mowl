"""Tests for exact geometric model checker in mowl.geometric.checker."""
from __future__ import annotations

import subprocess
import sys
import numpy as np
import pytest

from mowl.geometric.checker import (
    DictInterpretation,
    GeometricInterpretation,
    ModelChecker,
)
from mowl.geometric.errors import UnsupportedAxiom, UnsupportedRegion
from mowl.geometric.normal_forms import (
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
)
from mowl.geometric.regions import (
    EMPTY_1D,
    Box,
    Face,
    IntervalUnion,
    PairHull,
    RectUnion,
    simplex_points,
)


def test_interval_gci1_and_bot() -> None:
    universe = IntervalUnion([(0.0, 2.0)])
    reg_a = IntervalUnion([(0.0, 1.0)])
    reg_b = IntervalUnion([(0.5, 2.0)])
    reg_c = IntervalUnion([(0.5, 1.0)])

    eta = DictInterpretation(
        family="interval",
        universe=universe,
        classes={"A": reg_a, "B": reg_b, "C": reg_c},
        roles={},
    )
    assert isinstance(eta, GeometricInterpretation)
    assert eta.class_region(BOTTOM) == EMPTY_1D
    assert eta.class_region("UNKNOWN") == EMPTY_1D
    assert eta.class_region(TOP) == universe

    checker = ModelChecker(eta)
    assert checker.satisfies(NF2("A", "B", "C")) is True
    assert checker.satisfies(NF2("A", "B", BOTTOM)) is False
    assert checker.satisfies(NF1("C", "A")) is True
    assert checker.satisfies(NF1("A", "C")) is False
    assert checker.satisfies(NF1(BOTTOM, "A")) is True
    assert checker.satisfies(NF1("A", TOP)) is True

    axioms = [NF2("A", "B", "C"), NF1("C", "A"), NF2("A", "B", BOTTOM)]
    assert checker.violations(axioms) == [NF2("A", "B", BOTTOM)]
    assert checker.is_model(axioms) is False
    assert checker.is_model([NF2("A", "B", "C"), NF1("C", "A")]) is True


def test_interval_exists() -> None:
    universe = IntervalUnion([(0.0, 10.0)])
    r = RectUnion([((0.0, 1.0), (5.0, 6.0))])
    reg_a = IntervalUnion([(0.0, 1.0)])
    reg_b = IntervalUnion([(5.0, 6.0)])

    eta = DictInterpretation(
        family="interval",
        universe=universe,
        classes={"A": reg_a, "B": reg_b},
        roles={"r": r},
    )
    checker = ModelChecker(eta)
    assert checker.satisfies(NF3("A", "r", "B")) is True
    assert checker.satisfies(NF4("r", "B", "A")) is True

    # With B=[7,8], NF3(A, r, B) is False
    eta_disjoint = DictInterpretation(
        family="interval",
        universe=universe,
        classes={"A": reg_a, "B": IntervalUnion([(7.0, 8.0)])},
        roles={"r": r},
    )
    checker_disjoint = ModelChecker(eta_disjoint)
    assert checker_disjoint.satisfies(NF3("A", "r", "B")) is False


def test_interval_role_inclusions_and_chains() -> None:
    universe = IntervalUnion([(0.0, 10.0)])
    r = RectUnion([((0.0, 1.0), (5.0, 6.0))])
    s = RectUnion([((0.0, 2.0), (4.0, 7.0))])
    q = RectUnion([((5.5, 6.5), (8.0, 9.0))])
    t_ok = RectUnion([((0.0, 1.0), (8.0, 9.0))])
    t_bad = RectUnion([((0.0, 0.5), (8.0, 9.0))])

    eta = DictInterpretation(
        family="interval",
        universe=universe,
        classes={},
        roles={"r": r, "s": s, "q": q, "t_ok": t_ok, "t_bad": t_bad},
    )
    checker = ModelChecker(eta)
    assert checker.satisfies(NF6("r", "s")) is True
    assert checker.satisfies(NF6("s", "r")) is False
    assert checker.satisfies(NF7("r", "q", "t_ok")) is True
    assert checker.satisfies(NF7("r", "q", "t_bad")) is False


def test_polyhedral_face_subset_and_exists() -> None:
    pts = simplex_points(4)
    universe = Face.from_simplex(pts, {0, 1, 2, 3})
    f01 = Face.from_simplex(pts, {0, 1})
    f2 = Face.from_simplex(pts, {2})
    f0 = Face.from_simplex(pts, {0})

    # PairHull from pairs (0->2), (1->2)
    pair_pts = np.vstack(
        [
            np.concatenate([pts[0], pts[2]]),
            np.concatenate([pts[1], pts[2]]),
        ]
    )
    r = PairHull(pair_pts)

    eta = DictInterpretation(
        family="polyhedral",
        universe=universe,
        classes={"F01": f01, "F2": f2, "F0": f0},
        roles={"r": r},
    )
    checker = ModelChecker(eta)

    assert checker.satisfies(NF1("F0", "F01")) is True
    assert checker.satisfies(NF1("F01", "F0")) is False
    assert checker.satisfies(NF2("F01", "F0", "F0")) is True
    assert checker.satisfies(NF2("F01", "F2", BOTTOM)) is True
    assert checker.satisfies(NF2("F01", "F0", BOTTOM)) is False

    assert checker.satisfies(NF3("F01", "r", "F2")) is True
    assert checker.satisfies(NF4("r", "F2", "F01")) is True
    assert checker.satisfies(NF4("r", "F2", "F0")) is False
    assert checker.satisfies(NF3("F01", "r", "F0")) is False
    assert checker.satisfies(NF4("r", "F0", BOTTOM)) is True
    assert checker.satisfies(NF4("r", "F2", BOTTOM)) is False


def test_polyhedral_role_hierarchy_and_chains_and_unsupported_region() -> None:
    pts = simplex_points(4)
    universe = Face.from_simplex(pts, {0, 1, 2, 3})
    r = PairHull(
        np.vstack(
            [
                np.concatenate([pts[0], pts[1]]),
                np.concatenate([pts[1], pts[2]]),
            ]
        )
    )
    s = PairHull(
        np.vstack(
            [
                np.concatenate([pts[0], pts[1]]),
                np.concatenate([pts[1], pts[2]]),
                np.concatenate([pts[2], pts[3]]),
            ]
        )
    )
    t_ok = PairHull(
        np.vstack(
            [
                np.concatenate([pts[0], pts[2]]),
                np.concatenate([pts[1], pts[3]]),
            ]
        )
    )
    t_incomplete = PairHull(np.vstack([np.concatenate([pts[0], pts[2]])]))
    t_empty = PairHull(np.zeros((0, 6)))

    eta = DictInterpretation(
        family="polyhedral",
        universe=universe,
        classes={"F0": Face.from_simplex(pts, {0}), "F1": Face.from_simplex(pts, {1})},
        roles={
            "r": r,
            "s": s,
            "t_ok": t_ok,
            "t_incomplete": t_incomplete,
            "t_empty": t_empty,
        },
    )
    checker = ModelChecker(eta)
    assert checker.satisfies(NF6("r", "s")) is True
    assert checker.satisfies(NF6("s", "r")) is False
    assert checker.satisfies(NF7("r", "s", "t_ok")) is True
    assert checker.satisfies(NF7("r", "s", "t_incomplete")) is False
    assert checker.satisfies(NF7("r", "s", "t_empty")) is False

    # Non-Face class region on NF3 raises UnsupportedRegion
    class HRepOnlyPolyhedron:
        is_empty = False
        dim = 3

        def h_rep(self):
            return Face.from_simplex(pts, {0}).h_rep()

    eta_bad_cls = DictInterpretation(
        family="polyhedral",
        universe=universe,
        classes={"H": HRepOnlyPolyhedron(), "F1": Face.from_simplex(pts, {1})},
        roles={"r": r},
    )
    with pytest.raises(UnsupportedRegion):
        ModelChecker(eta_bad_cls).satisfies(NF3("H", "r", "F1"))

    # Role region without points attribute raises UnsupportedRegion on NF6/NF7
    classPointlessRole: object = object()
    eta_bad_role = DictInterpretation(
        family="polyhedral",
        universe=universe,
        classes={},
        roles={"bad": classPointlessRole, "s": s},
    )
    with pytest.raises(UnsupportedRegion):
        ModelChecker(eta_bad_role).satisfies(NF6("bad", "s"))


def test_box_unsupported_roles_and_propositional() -> None:
    universe = Box([0.0, 0.0], [10.0, 10.0])
    box_a = Box([1.0, 1.0], [3.0, 3.0])
    box_b = Box([2.0, 2.0], [4.0, 4.0])
    box_c = Box([2.0, 2.0], [3.0, 3.0])

    eta = DictInterpretation(
        family="box",
        universe=universe,
        classes={"A": box_a, "B": box_b, "C": box_c},
        roles={},
    )
    checker = ModelChecker(eta)

    assert checker.satisfies(NF1("C", "A")) is True
    assert checker.satisfies(NF1("A", "C")) is False
    assert checker.satisfies(NF2("A", "B", "C")) is True
    assert checker.satisfies(NF2("A", "B", BOTTOM)) is False
    assert checker.satisfies(NF1("UNSAT", BOTTOM)) is True

    with pytest.raises(UnsupportedAxiom):
        checker.satisfies(NF3("A", "r", "B"))
    with pytest.raises(UnsupportedAxiom):
        checker.satisfies(NF4("r", "A", "B"))
    with pytest.raises(UnsupportedAxiom):
        checker.satisfies(NF6("r", "s"))
    with pytest.raises(UnsupportedAxiom):
        checker.satisfies(NF7("r", "s", "t"))


def test_nondegenerate_and_strongly_faithful_on_names() -> None:
    class StubCanonicalModel:
        def __init__(self, names, entailments):
            self.names = list(names)
            self._entailments = set(entailments)

        def entails(self, a: str, b: str) -> bool:
            return a == b or b == TOP or (a, b) in self._entailments

    cm = StubCanonicalModel(
        names=["A", "B", TOP],
        entailments=[("A", "B")],
    )
    universe = IntervalUnion([(0.0, 3.0)])
    eta_faithful = DictInterpretation(
        family="interval",
        universe=universe,
        classes={
            "A": IntervalUnion([(0.0, 1.0)]),
            "B": IntervalUnion([(0.0, 2.0)]),
        },
        roles={},
        names=["A", "B", TOP],
    )
    checker = ModelChecker(eta_faithful)
    assert checker.is_nondegenerate(cm) is True
    assert checker.is_strongly_faithful_on_names(cm) is True

    # Collapsing A and B to the same region breaks strong faithfulness (B <= A not entailed)
    eta_collapsed = DictInterpretation(
        family="interval",
        universe=universe,
        classes={
            "A": IntervalUnion([(0.0, 2.0)]),
            "B": IntervalUnion([(0.0, 2.0)]),
        },
        roles={},
        names=["A", "B", TOP],
    )
    checker_collapsed = ModelChecker(eta_collapsed)
    assert checker_collapsed.is_nondegenerate(cm) is True
    assert checker_collapsed.is_strongly_faithful_on_names(cm) is False

    # Making A empty breaks non-degeneracy
    eta_degenerate = DictInterpretation(
        family="interval",
        universe=universe,
        classes={
            "A": EMPTY_1D,
            "B": IntervalUnion([(0.0, 2.0)]),
        },
        roles={},
        names=["A", "B", TOP],
    )
    assert ModelChecker(eta_degenerate).is_nondegenerate(cm) is False


def test_polyhedral_0d_and_hrep_fallback() -> None:
    # 0-D simplex (n = 1, k = 0)
    pts1 = simplex_points(1)
    u0 = Face.from_simplex(pts1, {0})
    f_emp0 = Face.from_simplex(pts1, set())
    r_full0 = PairHull(np.zeros((1, 0)))
    r_emp0 = PairHull(np.zeros((0, 0)))

    eta0 = DictInterpretation(
        family="polyhedral",
        universe=u0,
        classes={"A": u0, "E": f_emp0},
        roles={"r": r_full0, "r_emp": r_emp0},
    )
    chk0 = ModelChecker(eta0)
    assert chk0.satisfies(NF1("A", TOP)) is True
    assert chk0.satisfies(NF1("A", "E")) is False
    assert chk0.satisfies(NF3("A", "r", "A")) is True
    assert chk0.satisfies(NF3("A", "r_emp", "A")) is False
    assert chk0.satisfies(NF4("r", "A", "A")) is True
    assert chk0.satisfies(NF4("r", "A", "E")) is False
    assert chk0.satisfies(NF6("r", "r")) is True
    assert chk0.satisfies(NF7("r", "r", "r")) is True

    # H-rep fallback for thickened-style Face subclasses on NF1/NF2
    pts3 = simplex_points(3)
    u3 = Face.from_simplex(pts3, {0, 1, 2})

    class ThickFaceStub(Face):
        pass

    tf0 = ThickFaceStub.from_simplex(pts3, {0})
    tf01 = ThickFaceStub.from_simplex(pts3, {0, 1})
    tf2 = ThickFaceStub.from_simplex(pts3, {2})

    eta_thick = DictInterpretation(
        family="polyhedral",
        universe=u3,
        classes={"T0": tf0, "T01": tf01, "T2": tf2},
        roles={},
    )
    chk_thick = ModelChecker(eta_thick)
    assert chk_thick.satisfies(NF1("T0", "T01")) is True
    assert chk_thick.satisfies(NF1("T01", "T0")) is False
    assert chk_thick.satisfies(NF2("T01", "T0", "T0")) is True
    assert chk_thick.satisfies(NF2("T01", "T2", BOTTOM)) is True
    assert chk_thick.satisfies(NF2("T01", "T0", BOTTOM)) is False


def test_checker_imports_without_jvm() -> None:
    cmd = [
        sys.executable,
        "-c",
        "import mowl.geometric.checker, mowl.geometric.normal_forms; from mowl.geometric.normal_forms import NF1, TOP; print('ok')",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"stderr: {res.stderr}"
    assert "ok" in res.stdout


