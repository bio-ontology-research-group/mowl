"""Tests for BoxModel and BoxModelBuilder (MILP) in mowl.geometric.boxes."""
from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.boxes import BoxModel, BoxModelBuilder
from mowl.geometric.checker import ModelChecker
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.errors import (
    BoxModelNotFound,
    InconsistentOntology,
    InternalConsistencyError,
    MissingDependency,
    NotBoxRepresentable,
    UnsupportedAxiom,
)
from mowl.geometric.helly import HellyCertificate, HellyReport, helly_check
from mowl.geometric.normal_forms import (
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
    Axiom,
    NormalisedOntology,
)
from mowl.geometric.regions import Box

logger = logging.getLogger(__name__)

NS = "http://mowl.borg/geo#"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


def _assert_certificate_confirmed(
    tc: TestCase, cm: CanonicalModel, cert: HellyCertificate
) -> None:
    """Confirm that cert's pairwise conjunctions are satisfiable, triple is unsatisfiable, and pairwise_names match."""
    p12 = cert.c1 + cert.c2
    p13 = cert.c1 + cert.c3
    p23 = cert.c2 + cert.c3
    triple = cert.c1 + cert.c2 + cert.c3

    tc.assertTrue(cm.joint_satisfiable(p12), f"Expected {p12} to be satisfiable")
    tc.assertTrue(cm.joint_satisfiable(p23), f"Expected {p23} to be satisfiable")
    tc.assertTrue(cm.joint_satisfiable(p13), f"Expected {p13} to be satisfiable")
    tc.assertFalse(
        cm.joint_satisfiable(triple), f"Expected {triple} to be unsatisfiable"
    )

    for pair, pname in zip((p12, p13, p23), cert.pairwise_names):
        if pname is not None:
            tc.assertTrue(cm.is_satisfiable(pname))
            for x in pair:
                tc.assertTrue(cm.entails(pname, x))
            tc.assertTrue(cm.entails_conjunction(pair, pname))


class TestBoxes(TestCase):

    def test_helly_ok_box_model(self) -> None:
        ds = PathDataset(_fixture_path("helly_ok.owl"))
        cm = CanonicalModel.from_dataset(ds)

        model = BoxModelBuilder(cm).build()
        self.assertIsInstance(model, BoxModel)
        self.assertEqual(model.family, "box")
        self.assertGreaterEqual(model.dimension, 1)
        self.assertLessEqual(model.dimension, 2)
        self.assertTrue(model.is_model())
        self.assertTrue(model.is_nondegenerate())

        checker = ModelChecker(model)
        self.assertTrue(checker.is_model(cm.axioms))
        self.assertTrue(checker.is_nondegenerate(cm))
        self.assertIs(model.check(), model.check())

        # Universe is [0, M]^d
        d = model.dimension
        self.assertEqual(model.universe, Box(np.zeros(d), np.full(d, 100.0)))
        self.assertEqual(model.class_region(TOP), model.universe)
        self.assertEqual(model.class_region(BOTTOM), Box.empty(d))
        self.assertEqual(model.class_region(NS + "UnknownConcept"), Box.empty(d))
        with self.assertRaises(UnsupportedAxiom):
            model.role_region(NS + "r")

        # boxes() returns {name: (lo, hi)} for names with a box
        boxes_dict = model.boxes()
        self.assertIsInstance(boxes_dict, dict)
        for name in cm.names:
            self.assertIn(name, boxes_dict)
            lo, hi = boxes_dict[name]
            self.assertEqual(lo.shape, (d,))
            self.assertEqual(hi.shape, (d,))
            self.assertTrue(np.all(lo <= hi))
            self.assertEqual(model.class_region(name), Box(lo, hi))
        self.assertNotIn(BOTTOM, boxes_dict)

        for ax in cm.axioms:
            self.assertTrue(model.satisfies(ax))

    def test_t3_not_representable(self) -> None:
        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)

        with self.assertRaises(NotBoxRepresentable) as ctx:
            BoxModelBuilder(cm).build()
        rep = ctx.exception.report
        self.assertIsInstance(rep, HellyReport)
        self.assertEqual(len(rep.certificates_nondegenerate), 1)
        _assert_certificate_confirmed(self, cm, rep.certificates_nondegenerate[0])

        # With faithful="strong", also raises NotBoxRepresentable carrying strong certificates
        with self.assertRaises(NotBoxRepresentable) as ctx_strong:
            BoxModelBuilder(cm, faithful="strong").build()
        self.assertGreaterEqual(len(ctx_strong.exception.report.certificates_strong), 1)

        # One-way T3 has a strong certificate but no nondegenerate certificate:
        # raises NotBoxRepresentable for faithful="strong", not for faithful="weak"
        X1, X2, X3 = NS + "X1", NS + "X2", NS + "X3"
        A, B, C = NS + "A", NS + "B", NS + "C"
        no_oneway = NormalisedOntology(
            axioms=[
                NF2(X1, X2, A),
                NF2(X2, X3, B),
                NF2(X1, X3, C),
                NF2(A, B, BOTTOM),
                NF2(B, C, BOTTOM),
                NF2(A, C, BOTTOM),
            ],
            concept_names={A, B, C, X1, X2, X3, TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm_oneway = CanonicalModel.from_normalised(no_oneway)
        report_ow = helly_check(cm_oneway)
        self.assertEqual(len(report_ow.certificates_nondegenerate), 0)
        self.assertGreaterEqual(len(report_ow.certificates_strong), 1)
        # Strong certificates exclude only conjunction-faithful box models, so
        # faithful="strong" is not rejected: the builder searches and the model it
        # returns is strongly faithful on names.
        strong_model = BoxModelBuilder(cm_oneway, faithful="strong").build()
        self.assertTrue(strong_model.is_model())
        self.assertTrue(strong_model.is_nondegenerate())
        self.assertTrue(strong_model.is_strongly_faithful_on_names())

        # Under faithful="weak", one-way T3 is box-representable (X1, X2, X3 pairwise disjoint)
        weak_model = BoxModelBuilder(cm_oneway, faithful="weak").build()
        self.assertTrue(weak_model.is_model())
        self.assertTrue(weak_model.is_nondegenerate())

    def test_roles_unsupported(self) -> None:
        ds = PathDataset(_fixture_path("roles.owl"))
        cm = CanonicalModel.from_dataset(ds)

        with self.assertRaises(UnsupportedAxiom):
            BoxModelBuilder(cm).build()

        A, B = NS + "A", NS + "B"
        r, s, t = NS + "r", NS + "s", NS + "t"
        for role_ax in (
            NF3(A, r, B),
            NF4(r, A, B),
            NF6(r, s),
            NF7(r, s, t),
        ):
            with self.subTest(role_axiom=type(role_ax).__name__):
                no = NormalisedOntology(
                    axioms=[role_ax],
                    concept_names={A, B, TOP, BOTTOM},
                    role_names={r, s, t},
                    dropped_axioms=0,
                )
                cm_role = CanonicalModel.from_normalised(no)
                with self.assertRaises(UnsupportedAxiom):
                    BoxModelBuilder(cm_role).build()

    def test_strong_faithful_small(self) -> None:
        ds = PathDataset(_fixture_path("helly_ok.owl"))
        cm = CanonicalModel.from_dataset(ds)

        try:
            model = BoxModelBuilder(cm, faithful="strong", d_max=4, time_limit=30.0).build()
        except BoxModelNotFound as exc:
            logger.info("test_strong_faithful_small outcome: BoxModelNotFound (%s)", exc)
        else:
            logger.info(
                "test_strong_faithful_small outcome: BoxModel found in d=%d",
                model.dimension,
            )
            self.assertTrue(model.is_model())
            self.assertTrue(model.is_nondegenerate())
            self.assertTrue(model.is_strongly_faithful_on_names())
            checker = ModelChecker(model)
            self.assertTrue(checker.is_model(cm.axioms))
            self.assertTrue(checker.is_nondegenerate(cm))
            self.assertTrue(checker.is_strongly_faithful_on_names(cm))

    def test_strong_certificate_does_not_block_strongly_faithful_model(self) -> None:
        # Reviewer counterexample (final review C1): O = {X1 n X2 <= Y, Y n X3 <= bot}
        # has the strong certificate (X1, X2, X3) with no pairwise names, yet the
        # 1-D boxes X1=[0,1], X2=[2,3], X3=[4,5], Y=[6,7] are a non-degenerate
        # model that is strongly faithful on names.
        X1, X2, X3, Y = NS + "X1", NS + "X2", NS + "X3", NS + "Y"
        no = NormalisedOntology(
            axioms=[NF2(X1, X2, Y), NF2(Y, X3, BOTTOM)],
            concept_names={X1, X2, X3, Y, TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm = CanonicalModel.from_normalised(no)
        report = helly_check(cm)
        self.assertTrue(report.complete)
        self.assertEqual(len(report.certificates_nondegenerate), 0)
        self.assertEqual(len(report.certificates_strong), 1)
        cert = report.certificates_strong[0]
        self.assertEqual(set(cert.c1 + cert.c2 + cert.c3), {X1, X2, X3})
        self.assertEqual(cert.pairwise_names, (None, None, None))
        self.assertIs(report.box_representable_strong, False)

        hand = BoxModel(
            cm,
            1,
            {
                X1: Box([0.0], [1.0]),
                X2: Box([2.0], [3.0]),
                X3: Box([4.0], [5.0]),
                Y: Box([6.0], [7.0]),
            },
        )
        self.assertTrue(hand.is_model())
        self.assertTrue(hand.is_nondegenerate())
        self.assertTrue(hand.is_strongly_faithful_on_names())

        model = BoxModelBuilder(cm, faithful="strong").build()
        self.assertEqual(model.dimension, 1)
        self.assertTrue(model.is_model())
        self.assertTrue(model.is_nondegenerate())
        self.assertTrue(model.is_strongly_faithful_on_names())
        self.assertTrue(ModelChecker(model).is_strongly_faithful_on_names(cm))

    def test_time_limited_incumbent_is_post_checked(self) -> None:
        from scipy.optimize import milp as real_milp

        A, B, C = NS + "A", NS + "B", NS + "C"
        no = NormalisedOntology(
            axioms=[NF1(A, B), NF2(B, C, BOTTOM)],
            concept_names={A, B, C, TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm = CanonicalModel.from_normalised(no)

        def incumbent_milp(*args, **kwargs):
            # A real feasible solution, reported as "time limit reached" (status 1).
            res = real_milp(*args, **kwargs)
            self.assertEqual(res.status, 0)
            out = MagicMock()
            out.success = False
            out.status = 1
            out.x = res.x
            out.message = "Time limit reached. (HiGHS Status 13)"
            return out

        with patch("mowl.geometric.boxes._scipy_milp", side_effect=incumbent_milp):
            model = BoxModelBuilder(cm, d_max=2, time_limit=10.0).build()
        self.assertEqual(model.dimension, 1)
        self.assertTrue(model.is_model())
        self.assertTrue(model.is_nondegenerate())

        # An incumbent that fails the post-check (all boxes the point 0, so B and C
        # intersect) is not returned; the schedule continues and ends inconclusive.
        def bad_incumbent_milp(*args, **kwargs):
            n_vars = len(kwargs["c"])
            out = MagicMock()
            out.success = False
            out.status = 1
            out.x = np.zeros(n_vars)
            out.message = "Time limit reached. (HiGHS Status 13)"
            return out

        with patch(
            "mowl.geometric.boxes._scipy_milp", side_effect=bad_incumbent_milp
        ) as mocked:
            with self.assertRaises(BoxModelNotFound):
                BoxModelBuilder(cm, d_max=2, time_limit=10.0).build()
        self.assertEqual(mocked.call_count, 2)

        # Solver status 4 is logged at INFO, then treated as no model at that d.
        other_res = MagicMock()
        other_res.success = False
        other_res.status = 4
        other_res.x = None
        other_res.message = "Solver error"
        with patch("mowl.geometric.boxes._scipy_milp", return_value=other_res):
            with self.assertLogs("mowl.geometric.boxes", level="INFO") as logs:
                with self.assertRaises(BoxModelNotFound):
                    BoxModelBuilder(cm, d_max=1, time_limit=10.0).build()
        self.assertTrue(any("status 4" in line for line in logs.output))

    def test_random_propositional(self) -> None:
        names = [f"{NS}N{i}" for i in range(6)]
        counts: Dict[str, int] = {
            "inconsistent": 0,
            "not_box_representable": 0,
            "box_model_found": 0,
            "box_model_not_found": 0,
        }
        dims_used: List[int] = []

        for seed in range(20):
            rng = random.Random(seed)
            n_axioms = rng.randint(1, 8)
            axioms: List[Axiom] = []
            for _ in range(n_axioms):
                kind = rng.choice(("NF1", "NF2"))
                if kind == "NF1":
                    sub = rng.choice(names)
                    sup = (
                        BOTTOM
                        if rng.random() < 0.2
                        else rng.choice([c for c in names if c != sub])
                    )
                    axioms.append(NF1(sub, sup))
                else:
                    sub1, sub2 = rng.sample(names, 2)
                    sup = (
                        BOTTOM
                        if rng.random() < 0.2
                        else rng.choice([c for c in names if c not in (sub1, sub2)])
                    )
                    axioms.append(NF2(sub1, sub2, sup))

            no = NormalisedOntology(
                axioms=axioms,
                concept_names=set(names) | {TOP, BOTTOM},
                role_names=set(),
                dropped_axioms=0,
            )
            try:
                cm = CanonicalModel.from_normalised(no)
            except InconsistentOntology:
                counts["inconsistent"] += 1
                logger.info("seed=%d outcome=inconsistent", seed)
                continue

            report = helly_check(cm)
            for cert in report.certificates_strong:
                _assert_certificate_confirmed(self, cm, cert)
            for cert in report.certificates_nondegenerate:
                self.assertIn(cert, report.certificates_strong)
                _assert_certificate_confirmed(self, cm, cert)

            try:
                model = BoxModelBuilder(cm, d_max=4, time_limit=15.0).build()
            except NotBoxRepresentable as exc:
                counts["not_box_representable"] += 1
                self.assertGreaterEqual(
                    len(exc.report.certificates_nondegenerate), 1
                )
                for cert in exc.report.certificates_nondegenerate:
                    _assert_certificate_confirmed(self, cm, cert)
                logger.info(
                    "seed=%d outcome=NotBoxRepresentable certs=%d",
                    seed,
                    len(exc.report.certificates_nondegenerate),
                )
            except BoxModelNotFound as exc:
                counts["box_model_not_found"] += 1
                logger.info("seed=%d outcome=BoxModelNotFound (%s)", seed, exc)
            else:
                counts["box_model_found"] += 1
                dims_used.append(model.dimension)
                checker = ModelChecker(model)
                self.assertTrue(checker.is_model(cm.axioms))
                self.assertTrue(checker.is_nondegenerate(cm))
                self.assertTrue(model.is_model())
                self.assertTrue(model.is_nondegenerate())
                logger.info(
                    "seed=%d outcome=BoxModel d=%d", seed, model.dimension
                )

        logger.info(
            "test_random_propositional summary: %s, dims=%s", counts, dims_used
        )
        self.assertGreater(counts["box_model_found"], 0)

    def test_edge_cases_and_errors(self) -> None:
        A, B, C = NS + "A", NS + "B", NS + "C"
        no = NormalisedOntology(
            axioms=[
                NF1(TOP, A),
                NF1(B, BOTTOM),
                NF2(A, C, TOP),
                NF2(A, B, C),
            ],
            concept_names={A, B, C, TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm = CanonicalModel.from_normalised(no)
        model = BoxModelBuilder(cm, faithful="strong", d_max=2).build()
        self.assertTrue(model.is_model())
        self.assertTrue(model.is_nondegenerate())
        self.assertTrue(model.is_strongly_faithful_on_names())
        self.assertTrue(model.class_region(B).is_empty)
        self.assertNotIn(B, model.boxes())
        self.assertEqual(model.class_region(A), model.universe)

        # Invalid arguments
        with self.assertRaises(ValueError):
            BoxModelBuilder(cm, faithful="invalid")
        with self.assertRaises(ValueError):
            BoxModelBuilder(cm, d_max=0)
        with self.assertRaises(ValueError):
            BoxModelBuilder(cm, time_limit=0.0)
        with self.assertRaises(ValueError):
            BoxModelBuilder(cm, M=1.0, delta=1.0)

        # BoxModelNotFound when MILP is infeasible for all d <= d_max
        infeasible_res = MagicMock()
        infeasible_res.success = False
        infeasible_res.status = 2
        infeasible_res.x = None
        with patch("mowl.geometric.boxes._scipy_milp", return_value=infeasible_res):
            with self.assertRaises(BoxModelNotFound) as ctx_nf:
                BoxModelBuilder(cm, d_max=3, time_limit=12.5).build()
            msg = str(ctx_nf.exception)
            self.assertIn("3", msg)
            self.assertIn("12.5", msg)
            self.assertIn("no theorem guarantees", msg.lower())

        # BoxModelNotFound on timeout status
        timeout_res = MagicMock()
        timeout_res.success = False
        timeout_res.status = 1
        timeout_res.x = None
        with patch("mowl.geometric.boxes._scipy_milp", return_value=timeout_res):
            with self.assertRaises(BoxModelNotFound) as ctx_to:
                BoxModelBuilder(cm, d_max=4, time_limit=5.0).build()
            self.assertIn("no theorem guarantees", str(ctx_to.exception).lower())

        # MissingDependency if scipy.optimize.milp is unavailable
        with patch("mowl.geometric.boxes._scipy_milp", None):
            with self.assertRaises(MissingDependency) as ctx_md:
                BoxModelBuilder(cm).build()
            self.assertIn("scipy", str(ctx_md.exception).lower())
            self.assertIn("1.9", str(ctx_md.exception))

        # InternalConsistencyError if post-check fails
        with patch.object(ModelChecker, "is_model", return_value=False):
            with self.assertRaises(InternalConsistencyError):
                BoxModelBuilder(cm, d_max=1).build()

    def test_top_forced_boxes_are_representable(self) -> None:
        A, B, C = NS + "A", NS + "B", NS + "C"
        no = NormalisedOntology(
            axioms=[
                NF1(TOP, A),
                NF1(TOP, B),
                NF2(A, B, C),
            ],
            concept_names={A, B, C, TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm = CanonicalModel.from_normalised(no)
        model = BoxModelBuilder(cm).build()
        self.assertEqual(model.dimension, 1)
        self.assertEqual(model.class_region(A), model.universe)
        self.assertEqual(model.class_region(B), model.universe)
        self.assertEqual(model.class_region(C), model.universe)
        self.assertTrue(model.is_model())
        self.assertTrue(model.is_nondegenerate())

        # Second regression: NF1(TOP, A), NF2(A, B, BOTTOM) -> B <= BOTTOM follows
        # (since A = TOP), so B is unsatisfiable and gets no box; builder returns a model.
        no2 = NormalisedOntology(
            axioms=[
                NF1(TOP, A),
                NF2(A, B, BOTTOM),
            ],
            concept_names={A, B, TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm2 = CanonicalModel.from_normalised(no2)
        self.assertFalse(cm2.is_satisfiable(B))
        model2 = BoxModelBuilder(cm2).build()
        self.assertIsInstance(model2, BoxModel)
        self.assertEqual(model2.class_region(A), model2.universe)
        self.assertTrue(model2.class_region(B).is_empty)
        self.assertNotIn(B, model2.boxes())
        self.assertTrue(model2.is_model())
        self.assertTrue(model2.is_nondegenerate())

    def test_import_without_jvm(self) -> None:
        cmd = [
            sys.executable,
            "-c",
            (
                "import mowl.geometric; "
                "from mowl.geometric import BoxModel, BoxModelBuilder, boxes; "
                "print('ok')"
            ),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, f"stderr: {res.stderr}")
        self.assertIn("ok", res.stdout)
