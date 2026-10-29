"""Tests for the public API surface of mowl.geometric."""
from __future__ import annotations

import subprocess
import sys
from unittest import TestCase

import mowl
mowl.init_jvm("2g")

import mowl.geometric as geo


REQUIRED_PUBLIC_NAMES = {
    "CanonicalModel",
    "Completion",
    "CompletionResult",
    "ConvexPolyhedralModel",
    "IntervalUnionModel",
    "BoxModel",
    "BoxModelBuilder",
    "ModelChecker",
    "GeometricInterpretation",
    "DictInterpretation",
    "helly_check",
    "HellyReport",
    "HellyCertificate",
    "normalise",
    "NormalisedOntology",
    "to_owl",
    "Axiom",
    "NF1",
    "NF2",
    "NF3",
    "NF4",
    "NF6",
    "NF7",
    "TOP",
    "BOTTOM",
    "Face",
    "ThickFace",
    "PairHull",
    "IntervalUnion",
    "RectUnion",
    "Box",
    "EMPTY_1D",
    "simplex_points",
    "to_bounding_boxes",
    "class_points",
    "boxes_as_tensors",
    "GeometricError",
    "InconsistentOntology",
    "UnsupportedAxiom",
    "UnsupportedRegion",
    "NotBoxRepresentable",
    "BoxModelNotFound",
    "InternalConsistencyError",
    "MissingDependency",
}


class TestPublicAPI(TestCase):

    def test_public_api_imports(self) -> None:
        all_names = list(geo.__all__)
        self.assertEqual(len(all_names), len(set(all_names)))

        for req in REQUIRED_PUBLIC_NAMES:
            self.assertIn(req, geo.__all__, f"Missing {req!r} from mowl.geometric.__all__")

        ns = {}
        for name in all_names:
            with self.subTest(symbol=name):
                obj = getattr(geo, name)
                self.assertIsNotNone(obj)
                exec(f"from mowl.geometric import {name}", ns)
                self.assertIs(ns[name], obj)

    def test_lazy_completion_import_without_jvm(self) -> None:
        script = "\n".join(
            [
                "import sys",
                "import mowl.geometric as geo",
                "assert 'mowl.geometric.completion' not in sys.modules",
                "for name in geo.__all__:",
                "    if name not in ('completion', 'CanonicalModel', 'Completion', 'CompletionResult'):",
                "        getattr(geo, name)",
                "assert 'mowl.geometric.completion' not in sys.modules",
                "print('ok')",
            ]
        )
        cmd = [sys.executable, "-c", script]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, f"stderr: {res.stderr}")
        self.assertIn("ok", res.stdout)
