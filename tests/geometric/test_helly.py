"""Tests for Helly-satisfiability check and certificates in mowl.geometric.helly."""
from __future__ import annotations

import os
import random
import subprocess
from unittest import TestCase

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.errors import InconsistentOntology
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
    NormalisedOntology,
)
from tests.datasetFactory import FamilyDataset
from tests.geometric.helpers import random_el_tbox

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


class TestHellyCheck(TestCase):

    def test_t3_certificate(self) -> None:
        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)

        report = helly_check(cm)
        self.assertIsInstance(report, HellyReport)
        self.assertTrue(report.complete)
        self.assertFalse(report.has_roles)
        self.assertEqual(report.k_max, 4)

        self.assertEqual(len(report.certificates_strong), 1)
        self.assertEqual(len(report.certificates_nondegenerate), 1)
        self.assertIs(report.box_representable_nondegenerate, False)
        self.assertIs(report.box_representable_strong, False)

        cert = report.certificates_nondegenerate[0]
        self.assertEqual(cert, report.certificates_strong[0])

        X1, X2, X3 = NS + "X1", NS + "X2", NS + "X3"
        A, B, C = NS + "A", NS + "B", NS + "C"

        self.assertEqual(cert.c1, (X1,))
        self.assertEqual(cert.c2, (X2,))
        self.assertEqual(cert.c3, (X3,))
        self.assertEqual(set(cert.c1) | set(cert.c2) | set(cert.c3), {X1, X2, X3})
        self.assertEqual(cert.pairwise_names, (A, C, B))
        self.assertEqual(set(cert.pairwise_names), {A, B, C})
        _assert_certificate_confirmed(self, cm, cert)

        # T3 with only one-way inclusions X1 ⊓ X2 ⊑ A (not A ⊑ X1, A ⊑ X2):
        # pairwise conjunctions are not O-equivalent to any concept name, so
        # the certificate is strong only (not nondegenerate).
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
        rep_oneway = helly_check(cm_oneway)
        self.assertIn(
            HellyCertificate(
                c1=(X1,), c2=(X2,), c3=(X3,), pairwise_names=(None, None, None)
            ),
            rep_oneway.certificates_strong,
        )
        self.assertEqual(rep_oneway.certificates_nondegenerate, [])
        self.assertIs(rep_oneway.box_representable_strong, False)
        self.assertIs(rep_oneway.box_representable_nondegenerate, True)

    def test_helly_ok(self) -> None:
        ds = PathDataset(_fixture_path("helly_ok.owl"))
        cm = CanonicalModel.from_dataset(ds)

        report = helly_check(cm)
        self.assertEqual(report.certificates_strong, [])
        self.assertEqual(report.certificates_nondegenerate, [])
        self.assertTrue(report.complete)
        self.assertFalse(report.has_roles)
        self.assertIs(report.box_representable_strong, True)
        self.assertIs(report.box_representable_nondegenerate, True)

    def test_size4_minimal_set(self) -> None:
        P, Q, R, S = NS + "P", NS + "Q", NS + "R", NS + "S"
        Y1, Y2 = NS + "Y1", NS + "Y2"
        no = NormalisedOntology(
            axioms=[
                NF2(P, Q, Y1),
                NF2(Y1, R, Y2),
                NF2(Y2, S, BOTTOM),
            ],
            concept_names={P, Q, R, S, Y1, Y2, TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm = CanonicalModel.from_normalised(no)

        # {Y2, S} is jointly unsatisfiable of size 2 -> NOT a certificate
        self.assertFalse(cm.joint_satisfiable([Y2, S]))
        # {Y1, R, S} has size 3, all pairs satisfiable, jointly unsatisfiable -> strong certificate
        self.assertTrue(cm.joint_satisfiable([Y1, R]))
        self.assertTrue(cm.joint_satisfiable([Y1, S]))
        self.assertTrue(cm.joint_satisfiable([R, S]))
        self.assertFalse(cm.joint_satisfiable([Y1, R, S]))

        # {P, Q, R, S} has size 4, all 3-subsets satisfiable, jointly unsatisfiable -> strong certificate
        for sub3 in ([P, Q, R], [P, Q, S], [P, R, S], [Q, R, S]):
            self.assertTrue(cm.joint_satisfiable(sub3))
        self.assertFalse(cm.joint_satisfiable([P, Q, R, S]))

        # Neither Y1 nor Y2 is equivalent to its left-hand conjunction (only P ⊓ Q ⊑ Y1 and Y1 ⊓ R ⊑ Y2)
        self.assertTrue(cm.entails_conjunction([P, Q], Y1))
        self.assertFalse(cm.entails(Y1, P))
        self.assertTrue(cm.entails_conjunction([R, Y1], Y2))
        self.assertFalse(cm.entails(Y2, R))

        # k_max = 4 finds both {R, S, Y1} (size 3) and {P, Q, R, S} (size 4)
        report4 = helly_check(cm, k_max=4)
        expected_cert3 = HellyCertificate(
            c1=(R,), c2=(S,), c3=(Y1,), pairwise_names=(None, None, None)
        )
        expected_cert4 = HellyCertificate(
            c1=(P,), c2=(Q,), c3=(R, S), pairwise_names=(None, None, None)
        )
        self.assertEqual(report4.certificates_strong, [expected_cert3, expected_cert4])
        self.assertEqual(report4.certificates_nondegenerate, [])
        self.assertIs(report4.box_representable_strong, False)
        for cert in report4.certificates_strong:
            self.assertNotIn(Y2, set(cert.c1) | set(cert.c2) | set(cert.c3))
            _assert_certificate_confirmed(self, cm, cert)

        # k_max = 3 finds {R, S, Y1} (size 3) but cuts off before {P, Q, R, S} (size 4)
        report3 = helly_check(cm, k_max=3)
        self.assertFalse(report3.complete)
        self.assertEqual(report3.certificates_strong, [expected_cert3])
        self.assertEqual(report3.certificates_nondegenerate, [])
        self.assertIs(report3.box_representable_nondegenerate, None)
        self.assertIs(report3.box_representable_strong, False)

        # k_max = 2 cuts off before any size >= 3 certificates are checked
        report2 = helly_check(cm, k_max=2)
        self.assertFalse(report2.complete)
        self.assertEqual(report2.certificates_strong, [])
        self.assertEqual(report2.certificates_nondegenerate, [])
        self.assertIs(report2.box_representable_nondegenerate, None)
        self.assertIs(report2.box_representable_strong, None)

        # k_max = 5 exhausts all candidates completely
        report5 = helly_check(cm, k_max=5)
        self.assertTrue(report5.complete)
        self.assertEqual(report5.certificates_strong, [expected_cert3, expected_cert4])
        self.assertEqual(report5.certificates_nondegenerate, [])
        self.assertIs(report5.box_representable_nondegenerate, True)
        self.assertIs(report5.box_representable_strong, False)

        # max_certificates = 1 cuts off after the first strong certificate
        report_mc1 = helly_check(cm, k_max=4, max_certificates=1)
        self.assertFalse(report_mc1.complete)
        self.assertEqual(report_mc1.certificates_strong, [expected_cert3])
        self.assertIs(report_mc1.box_representable_strong, False)
        self.assertIs(report_mc1.box_representable_nondegenerate, None)

    def test_roles_incomplete(self) -> None:
        ds = PathDataset(_fixture_path("roles.owl"))
        cm = CanonicalModel.from_dataset(ds)

        report = helly_check(cm)
        self.assertTrue(report.has_roles)
        self.assertFalse(report.complete)
        self.assertEqual(report.certificates_strong, [])
        self.assertEqual(report.certificates_nondegenerate, [])
        self.assertIs(report.box_representable_strong, None)
        self.assertIs(report.box_representable_nondegenerate, None)

        # Verify each role axiom type (NF3, NF4, NF6, NF7) triggers has_roles=True and complete=False
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
                rep = helly_check(cm_role)
                self.assertTrue(rep.has_roles)
                self.assertFalse(rep.complete)
                self.assertIs(rep.box_representable_strong, None)
                self.assertIs(rep.box_representable_nondegenerate, None)

        # Certificate on an ontology with roles still proves non-box-representability (False, not None)
        ds_t3 = PathDataset(_fixture_path("t3.owl"))
        cm_t3 = CanonicalModel.from_dataset(ds_t3)
        no_t3_roles = NormalisedOntology(
            axioms=list(cm_t3.axioms) + [NF3(NS + "Z", r, NS + "Z")],
            concept_names=set(cm_t3.concept_names),
            role_names={r},
            dropped_axioms=0,
        )
        rep_t3_roles = helly_check(CanonicalModel.from_normalised(no_t3_roles))
        self.assertTrue(rep_t3_roles.has_roles)
        self.assertFalse(rep_t3_roles.complete)
        self.assertEqual(len(rep_t3_roles.certificates_nondegenerate), 1)
        self.assertIs(rep_t3_roles.box_representable_nondegenerate, False)
        self.assertIs(rep_t3_roles.box_representable_strong, False)

    def test_certificates_are_confirmed(self) -> None:
        # Check fixtures
        for fname in ("t3.owl", "helly_ok.owl", "roles.owl", "top_lhs.owl"):
            cm = CanonicalModel.from_dataset(PathDataset(_fixture_path(fname)))
            report = helly_check(cm)
            for cert in report.certificates_strong:
                _assert_certificate_confirmed(self, cm, cert)

        # Check FamilyDataset
        cm_fam = CanonicalModel.from_dataset(FamilyDataset())
        rep_fam = helly_check(cm_fam)
        for cert in rep_fam.certificates_strong:
            _assert_certificate_confirmed(self, cm_fam, cert)

        # Check 30 random propositional TBoxes
        total_certs = 0
        for seed in range(30):
            no = random_el_tbox(random.Random(seed), allow_roles=False)
            try:
                cm = CanonicalModel.from_normalised(no)
            except InconsistentOntology:
                continue
            rep = helly_check(cm, k_max=4)
            for cert in rep.certificates_strong:
                total_certs += 1
                _assert_certificate_confirmed(self, cm, cert)
            for cert in rep.certificates_nondegenerate:
                self.assertIn(cert, rep.certificates_strong)
                self.assertTrue(all(p is not None for p in cert.pairwise_names))

        #Also verify a seeded multi-axiom propositional TBox with NF1/NF2 produces confirmed certificates
        N = [f"{NS}C{i}" for i in range(6)]
        no_custom = NormalisedOntology(
            axioms=[
                NF2(N[0], N[1], N[3]),
                NF2(N[3], N[2], BOTTOM),
                NF1(N[4], N[0]),
            ],
            concept_names=set(N) | {TOP, BOTTOM},
            role_names=set(),
            dropped_axioms=0,
        )
        cm_custom = CanonicalModel.from_normalised(no_custom)
        rep_custom = helly_check(cm_custom)
        self.assertGreaterEqual(len(rep_custom.certificates_strong), 2)
        for cert in rep_custom.certificates_strong:
            _assert_certificate_confirmed(self, cm_custom, cert)

    def test_import_without_jvm(self) -> None:
        cmd = [
            "/home/leechuck/.venvs/mowl-dev/bin/python",
            "-c",
            (
                "import mowl.geometric; "
                "from mowl.geometric import HellyCertificate, HellyReport, helly_check, helly; "
                "print('ok')"
            ),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, f"stderr: {res.stderr}")
        self.assertIn("ok", res.stdout)

