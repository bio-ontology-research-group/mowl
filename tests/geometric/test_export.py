"""Tests for mowl.geometric.export conversion functions."""
from __future__ import annotations

import os
import subprocess
import sys
from unittest import TestCase

import numpy as np

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.boxes import BoxModel, BoxModelBuilder
from mowl.geometric.checker import DictInterpretation, ModelChecker
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.export import (
    boxes_as_tensors,
    class_points,
    to_bounding_boxes,
)
from mowl.geometric.normal_forms import BOTTOM, NF2, TOP, NormalisedOntology
from mowl.geometric.polyhedral import ConvexPolyhedralModel
from mowl.geometric.regions import Box
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl.borg/geo#"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


class TestExport(TestCase):

    def test_bounding_boxes_contain_points(self) -> None:
        for fixture in ("t3.owl", "helly_ok.owl", "roles.owl"):
            with self.subTest(fixture=fixture):
                ds = PathDataset(_fixture_path(fixture))
                cm = CanonicalModel.from_dataset(ds)
                poly = ConvexPolyhedralModel.from_canonical_model(cm)

                bboxes = to_bounding_boxes(poly)
                pts_map = class_points(poly)

                for cname in cm.concept_names:
                    self.assertIn(cname, bboxes)
                    self.assertIn(cname, pts_map)
                    lo, hi = bboxes[cname]
                    pts = pts_map[cname]
                    self.assertEqual(lo.shape, (poly.dimension,))
                    self.assertEqual(hi.shape, (poly.dimension,))
                    np.testing.assert_array_equal(pts, poly.class_points(cname))

                    if pts.shape[0] == 0:
                        if poly.dimension > 0:
                            self.assertTrue(np.all(np.isnan(lo)))
                            self.assertTrue(np.all(np.isnan(hi)))
                    else:
                        self.assertTrue(np.all(lo <= hi))
                        for row in pts:
                            self.assertTrue(np.all(lo <= row))
                            self.assertTrue(np.all(row <= hi))

        # Also demonstrate on t3 that bounding boxes are lossy and do NOT form a model:
        # for X1, X2, X3 their simplex faces are disjoint on the triple intersection,
        # whereas axis-parallel bounding boxes of the faces of X1, X2, X3 share a
        # common point (which follows from Helly number 2 of boxes).
        ds_t3 = PathDataset(_fixture_path("t3.owl"))
        cm_t3 = CanonicalModel.from_dataset(ds_t3)
        poly_t3 = ConvexPolyhedralModel.from_canonical_model(cm_t3)
        bboxes_t3 = to_bounding_boxes(poly_t3)
        box_classes = {
            cname: (
                Box.empty(poly_t3.dimension)
                if np.any(np.isnan(lo))
                else Box(lo, hi)
            )
            for cname, (lo, hi) in bboxes_t3.items()
        }
        bbox_interp = DictInterpretation(
            family="box",
            universe=Box(
                np.zeros(poly_t3.dimension), np.ones(poly_t3.dimension)
            ),
            classes=box_classes,
            names=cm_t3.names,
        )
        self.assertFalse(ModelChecker(bbox_interp).is_model(cm_t3.axioms))

    def test_bounding_boxes_with_unsatisfiable_class(self) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        ont = adapter.create_ontology("http://mowl.borg/geo/export_unsat")
        A = NS + "A"
        B = NS + "B"
        cls_a = adapter.create_class(A)
        cls_b = adapter.create_class(B)
        bot = adapter.create_class(BOTTOM)
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, cls_b))
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, bot))

        cm = CanonicalModel.from_ontology(ont)
        poly = ConvexPolyhedralModel.from_canonical_model(cm)
        bboxes = to_bounding_boxes(poly)

        lo_a, hi_a = bboxes[A]
        self.assertEqual(lo_a.shape, (poly.dimension,))
        self.assertEqual(hi_a.shape, (poly.dimension,))
        self.assertTrue(np.all(np.isnan(lo_a)))
        self.assertTrue(np.all(np.isnan(hi_a)))

        lo_bot, hi_bot = bboxes[BOTTOM]
        self.assertTrue(np.all(np.isnan(lo_bot)))
        self.assertTrue(np.all(np.isnan(hi_bot)))

        lo_b, hi_b = bboxes[B]
        self.assertFalse(np.any(np.isnan(lo_b)))
        self.assertFalse(np.any(np.isnan(hi_b)))

    def test_boxes_as_tensors_order(self) -> None:
        ds = PathDataset(_fixture_path("helly_ok.owl"))
        cm = CanonicalModel.from_dataset(ds)
        box_model = BoxModelBuilder(cm).build()
        d = box_model.dimension

        # 1. Fixture Dataset.class_to_id
        class_to_id = dict(ds.class_to_id)
        lo_arr, hi_arr = boxes_as_tensors(box_model, class_to_id)
        self.assertEqual(lo_arr.shape, (len(class_to_id), d))
        self.assertEqual(hi_arr.shape, (len(class_to_id), d))
        self.assertEqual(lo_arr.dtype, np.float64)
        self.assertEqual(hi_arr.dtype, np.float64)

        raw_boxes = box_model.boxes()
        for cname, idx in class_to_id.items():
            if cname in raw_boxes:
                exp_lo, exp_hi = raw_boxes[cname]
                np.testing.assert_allclose(lo_arr[idx], exp_lo)
                np.testing.assert_allclose(hi_arr[idx], exp_hi)
            else:
                self.assertTrue(np.all(np.isnan(lo_arr[idx])))
                self.assertTrue(np.all(np.isnan(hi_arr[idx])))

        # 2. Centre/offset mode (as_center_offset=True)
        centers, offsets = boxes_as_tensors(
            box_model, class_to_id, as_center_offset=True
        )
        self.assertEqual(centers.shape, (len(class_to_id), d))
        self.assertEqual(offsets.shape, (len(class_to_id), d))
        np.testing.assert_allclose(centers, 0.5 * (lo_arr + hi_arr), equal_nan=True)
        np.testing.assert_allclose(offsets, 0.5 * (hi_arr - lo_arr), equal_nan=True)
        self.assertTrue(np.all(offsets[~np.isnan(offsets)] >= 0.0))

        # 3. Mapping containing absent / unsatisfiable names and permuted row order
        extended_map = {
            NS + "AbsentClass": 0,
            BOTTOM: 1,
            NS + "A": 2,
            TOP: 3,
            NS + "B": 4,
        }
        lo_ext, hi_ext = boxes_as_tensors(box_model, extended_map)
        self.assertEqual(lo_ext.shape, (5, d))
        self.assertTrue(np.all(np.isnan(lo_ext[0])))
        self.assertTrue(np.all(np.isnan(hi_ext[0])))
        self.assertTrue(np.all(np.isnan(lo_ext[1])))
        self.assertTrue(np.all(np.isnan(hi_ext[1])))
        np.testing.assert_allclose(lo_ext[2], raw_boxes[NS + "A"][0])
        np.testing.assert_allclose(hi_ext[2], raw_boxes[NS + "A"][1])
        np.testing.assert_allclose(lo_ext[3], np.zeros(d))
        np.testing.assert_allclose(hi_ext[3], np.full(d, box_model.M))
        np.testing.assert_allclose(lo_ext[4], raw_boxes[NS + "B"][0])
        np.testing.assert_allclose(hi_ext[4], raw_boxes[NS + "B"][1])

        c_ext, o_ext = boxes_as_tensors(
            box_model, extended_map, as_center_offset=True
        )
        self.assertTrue(np.all(np.isnan(c_ext[0])))
        self.assertTrue(np.all(np.isnan(o_ext[0])))
        self.assertTrue(np.all(np.isnan(c_ext[1])))
        self.assertTrue(np.all(np.isnan(o_ext[1])))
        np.testing.assert_allclose(
            c_ext[2], 0.5 * (raw_boxes[NS + "A"][0] + raw_boxes[NS + "A"][1])
        )
        np.testing.assert_allclose(
            o_ext[2], 0.5 * (raw_boxes[NS + "A"][1] - raw_boxes[NS + "A"][0])
        )

        # 4. Empty class_to_id
        lo_empty, hi_empty = boxes_as_tensors(box_model, {})
        self.assertEqual(lo_empty.shape, (0, d))
        self.assertEqual(hi_empty.shape, (0, d))

        # 5. Non-contiguous indices raise ValueError
        with self.assertRaises(ValueError):
            boxes_as_tensors(box_model, {NS + "A": 1})

    def test_import_without_jvm(self) -> None:
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import mowl.geometric.export; "
                "from mowl.geometric import to_bounding_boxes, class_points, boxes_as_tensors, export; "
                "assert 'mowl.geometric.completion' not in sys.modules; "
                "print('ok')"
            ),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, f"stderr: {res.stderr}")
        self.assertIn("ok", res.stdout)
