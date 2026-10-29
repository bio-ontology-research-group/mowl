"""Tests for ExactBoxInit in mowl.geometric.init."""
from __future__ import annotations

import os
import random
import subprocess
import sys
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import mowl
mowl.init_jvm("2g")

from org.semanticweb.owlapi.model import IRI

from mowl.datasets import Dataset, PathDataset
from mowl.geometric import ExactBoxInit
from mowl.geometric.boxes import BoxModelBuilder
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.errors import (
    BoxModelNotFound,
    InconsistentOntology,
    InternalConsistencyError,
    UnsupportedRegion,
)
from mowl.geometric.normal_forms import BOTTOM, NF1, NF2, TOP, NormalisedOntology, to_owl
from mowl.models import BoxSquaredEL
from mowl.nn import BoxSquaredELModule, ELBEModule
from mowl.owlapi import OWLAPIAdapter
from tests.geometric.helpers import (
    NS,
    fixture_path,
    helly_ok_cm,
    helly_ok_dataset,
    roles_cm,
    roles_dataset,
    t3_cm,
    t3_dataset,
)

__all__ = [
    "helly_ok_cm",
    "helly_ok_dataset",
    "roles_cm",
    "roles_dataset",
    "t3_cm",
    "t3_dataset",
]

SCRATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_scratch")


def _scratch_path(filename: str) -> str:
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    return os.path.join(SCRATCH_DIR, filename)


def box(init: ExactBoxInit, row: int, comp: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(lo, hi)`` corners for class ``row`` and component ``comp`` in ``init``."""
    c = init.centers[row, comp]
    o = init.offsets[row, comp]
    return c - o, c + o


def test_projected_k1_containment_t3(t3_dataset: PathDataset) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7, seed=0)
    cm = CanonicalModel.from_dataset(t3_dataset)
    c2i = init.class_to_id
    for a in cm.names:
        for b in cm.names:
            if cm.entails(a, b) and cm.is_satisfiable(a):
                lo_a, hi_a = box(init, c2i[a])
                lo_b, hi_b = box(init, c2i[b])
                assert (lo_b - 1e-6 <= lo_a).all() and (hi_a <= hi_b + 1e-6).all()


def test_projected_k1_containment_roles_and_random_200(roles_dataset: PathDataset) -> None:
    # Containment guarantee on roles.owl
    init_r = ExactBoxInit.from_dataset(roles_dataset, embed_dim=6, seed=1)
    cm_r = CanonicalModel.from_dataset(roles_dataset)
    c2i_r = init_r.class_to_id
    for a in cm_r.names:
        for b in cm_r.names:
            if cm_r.entails(a, b) and cm_r.is_satisfiable(a):
                lo_a, hi_a = box(init_r, c2i_r[a])
                lo_b, hi_b = box(init_r, c2i_r[b])
                assert (lo_b - 1e-6 <= lo_a).all() and (hi_a <= hi_b + 1e-6).all()

    # Containment guarantee on a 200-class random TBox (Spec §6.3)
    rng = random.Random(42)
    names = [f"{NS}C{i}" for i in range(200)]
    axioms = []
    for _ in range(300):
        sub, sup = rng.sample(names, 2)
        axioms.append(NF1(sub, sup))
    for _ in range(80):
        s1, s2, sup = rng.sample(names, 3)
        axioms.append(NF2(s1, s2, sup))
    no = NormalisedOntology(
        axioms=axioms,
        concept_names=set(names) | {TOP, BOTTOM},
        role_names=set(),
        dropped_axioms=0,
    )
    cm_200 = CanonicalModel.from_normalised(no)
    c2i_200 = {name: i for i, name in enumerate(sorted(no.concept_names))}
    init_200 = ExactBoxInit.from_canonical_model(cm_200, c2i_200, embed_dim=8, seed=0)
    for a in cm_200.names:
        for b in cm_200.names:
            if cm_200.entails(a, b) and cm_200.is_satisfiable(a):
                lo_a, hi_a = box(init_200, c2i_200[a])
                lo_b, hi_b = box(init_200, c2i_200[b])
                assert (lo_b - 1e-6 <= lo_a).all() and (hi_a <= hi_b + 1e-6).all()


def test_projected_k1_zero_gci0_loss_after_apply(t3_dataset: PathDataset) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7, seed=0)
    model = BoxSquaredEL(
        Dataset(t3_dataset.ontology),
        embed_dim=7,
        margin=0.0,
        model_filepath=_scratch_path("box2el.pt"),
    )
    n_written = init.apply_to_module(model.module)
    assert n_written == len(init.class_to_id)
    gci0_data = model.training_datasets["gci0"].data
    assert len(gci0_data) > 0
    with torch.no_grad():
        losses = model.module(gci0_data, "gci0")
    assert torch.all(losses <= 1e-6)


def test_unsat_subclass_axiom_is_not_guaranteed() -> None:
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology("http://mowl.borg/geo/unsat_subclass")
    cls_a = adapter.create_class(NS + "A")
    cls_b = adapter.create_class(NS + "B")
    cls_u = adapter.create_class(NS + "U")
    bot = adapter.create_class(BOTTOM)

    mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, cls_b))
    mgr.addAxiom(ont, adapter.create_subclass_of(cls_u, bot))
    mgr.addAxiom(ont, adapter.create_subclass_of(cls_u, cls_a))

    ds = Dataset(ont)
    cm = CanonicalModel.from_dataset(ds)
    assert cm.is_satisfiable(NS + "A")
    assert cm.is_satisfiable(NS + "B")
    assert not cm.is_satisfiable(NS + "U")

    init = ExactBoxInit.from_dataset(ds, embed_dim=4, seed=0)
    model = BoxSquaredEL(
        ds,
        embed_dim=4,
        margin=0.0,
        model_filepath=_scratch_path("unsat_subclass.pt"),
    )
    init.apply_to_module(model.module)

    gci0_data = model.training_datasets["gci0"].data
    with torch.no_grad():
        losses = model.module(gci0_data, "gci0")

    id_to_iri = {idx: iri for iri, idx in init.class_to_id.items()}
    sat_rows = 0
    unsat_rows = 0
    for r in range(len(gci0_data)):
        sub_iri = id_to_iri[int(gci0_data[r, 0].item())]
        loss_val = float(losses[r].item())
        if cm.is_satisfiable(sub_iri):
            assert loss_val <= 1e-6
            sat_rows += 1
        else:
            assert loss_val >= 0.0
            unsat_rows += 1
    assert sat_rows >= 1
    assert unsat_rows >= 1


def test_bottom_and_unsat_are_points(t3_dataset: PathDataset) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, seed=0)
    bot_idx = init.class_to_id[BOTTOM]
    np.testing.assert_allclose(init.centers[bot_idx], 0.0)
    np.testing.assert_allclose(init.offsets[bot_idx], 0.0)

    # Also test an explicit unsatisfiable concept name U != BOTTOM
    A, B, U = NS + "A", NS + "B", NS + "Unsat"
    no = NormalisedOntology(
        axioms=[NF1(A, B), NF1(U, BOTTOM)],
        concept_names={A, B, U, TOP, BOTTOM},
        role_names=set(),
        dropped_axioms=0,
    )
    cm = CanonicalModel.from_normalised(no)
    assert not cm.is_satisfiable(U)
    c2i = {name: idx for idx, name in enumerate(sorted(no.concept_names))}
    init_u = ExactBoxInit.from_canonical_model(cm, c2i, embed_dim=4, n_components=2, seed=0)
    assert init_u.report["n_skipped"] == 0
    for name in (BOTTOM, U):
        idx = c2i[name]
        np.testing.assert_allclose(init_u.centers[idx], 0.0)
        np.testing.assert_allclose(init_u.offsets[idx], 0.0)


def test_init_skips_unknown_classes(roles_dataset: PathDataset) -> None:
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology("http://mowl.borg/geo/extra_test")
    extra_iri = NS + "ExtraOnlyInTest"
    cls_a = adapter.create_class(NS + "A")
    cls_extra = adapter.create_class(extra_iri)
    mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, cls_extra))
    extra_path = _scratch_path("extra.owl")
    mgr.saveOntology(ont, IRI.create("file:" + os.path.abspath(extra_path)))

    ds = PathDataset(fixture_path("roles.owl"), testing_path=extra_path)
    init = ExactBoxInit.from_dataset(ds, embed_dim=5, seed=0)
    extra_idx = init.class_to_id[extra_iri]

    assert init.report["n_skipped"] == 1
    assert init.report["n_initialised"] == len(init.class_to_id) - 1
    assert np.all(np.isnan(init.centers[extra_idx]))
    assert np.all(np.isnan(init.offsets[extra_idx]))

    torch.manual_seed(0)
    module = BoxSquaredELModule(
        nb_ont_classes=len(init.class_to_id),
        nb_rels=max(1, len(ds.object_properties)),
        embed_dim=5,
    )
    orig_center = module.class_center.weight[extra_idx].detach().clone()
    orig_offset = module.class_offset.weight[extra_idx].detach().clone()

    written = init.apply_to_module(module)
    assert written == len(init.class_to_id) - 1
    assert not torch.any(torch.isnan(module.class_center.weight))
    assert not torch.any(torch.isnan(module.class_offset.weight))
    assert torch.allclose(module.class_center.weight[extra_idx], orig_center)
    assert torch.allclose(module.class_offset.weight[extra_idx], orig_offset)


def test_anchor_loss_ignores_uninitialised_rows(roles_dataset: PathDataset) -> None:
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology("http://mowl.borg/geo/extra_anchor")
    extra_iri = NS + "ExtraOnlyInTest"
    mgr.addAxiom(
        ont,
        adapter.create_subclass_of(
            adapter.create_class(NS + "A"), adapter.create_class(extra_iri)
        ),
    )
    extra_path = _scratch_path("extra_anchor.owl")
    mgr.saveOntology(ont, IRI.create("file:" + os.path.abspath(extra_path)))

    ds = PathDataset(fixture_path("roles.owl"), testing_path=extra_path)
    init = ExactBoxInit.from_dataset(ds, embed_dim=5, seed=0)
    extra_idx = init.class_to_id[extra_iri]
    known_idx = init.class_to_id[NS + "A"]

    module = BoxSquaredELModule(
        nb_ont_classes=len(init.class_to_id),
        nb_rels=max(1, len(ds.object_properties)),
        embed_dim=5,
    )
    init.apply_to_module(module)

    loss0 = init.anchor_loss(module)
    assert torch.isfinite(loss0)
    assert float(loss0.item()) <= 1e-12

    # Perturb an initialised row so anchor_loss > 0
    with torch.no_grad():
        module.class_center.weight[known_idx] += 1.0
    loss_before = init.anchor_loss(module).detach().clone()
    assert float(loss_before.item()) > 0.1

    # Perturb the uninitialised (NaN) row: anchor_loss must not change
    with torch.no_grad():
        module.class_center.weight[extra_idx] += 50.0
        module.class_offset.weight[extra_idx] += 50.0
    loss_after = init.anchor_loss(module)
    assert torch.allclose(loss_before, loss_after, atol=1e-7)

    # Gradient on the uninitialised row must be zero
    module.zero_grad()
    loss_after.backward()
    assert module.class_center.weight.grad is not None
    assert torch.all(module.class_center.weight.grad[extra_idx] == 0.0)
    assert torch.all(module.class_offset.weight.grad[extra_idx] == 0.0)


def test_kmeans_components_duplicate_when_extension_small(t3_dataset: PathDataset) -> None:
    init1 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=3, n_components=1, seed=0)
    init3 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=3, n_components=3, seed=0)
    cm = CanonicalModel.from_dataset(t3_dataset)
    c2i = init3.class_to_id

    singleton_found = False
    for name in cm.names:
        if len(cm.extension(name)) == 1:
            singleton_found = True
            idx = c2i[name]
            for comp in range(3):
                np.testing.assert_allclose(
                    init3.centers[idx, comp], init1.centers[idx, 0], atol=1e-6
                )
                np.testing.assert_allclose(
                    init3.offsets[idx, comp], init1.offsets[idx, 0], atol=1e-6
                )
    assert singleton_found

    # Also check |E_C| == 2 with n_components == 3: 3rd component duplicates 1st
    A, B = NS + "A", NS + "B"
    no = NormalisedOntology(
        axioms=[NF1(A, B)],
        concept_names={A, B, TOP, BOTTOM},
        role_names=set(),
        dropped_axioms=0,
    )
    cm2 = CanonicalModel.from_normalised(no)
    assert len(cm2.extension(B)) == 2
    c2i2 = {name: i for i, name in enumerate(sorted(no.concept_names))}
    init_b = ExactBoxInit.from_canonical_model(
        cm2, c2i2, embed_dim=3, n_components=3, seed=0
    )
    b_idx = c2i2[B]
    np.testing.assert_allclose(
        init_b.centers[b_idx, 2], init_b.centers[b_idx, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        init_b.offsets[b_idx, 2], init_b.offsets[b_idx, 0], atol=1e-6
    )
    assert init_b.report["n_degenerate_kmeans"] == 0

    # When kmeans2 returns fewer than k' non-empty clusters, report["n_degenerate_kmeans"] counts it
    with patch(
        "mowl.geometric.init.kmeans2",
        side_effect=lambda pts, kp, **kw: (pts[:kp], np.zeros(len(pts), dtype=int)),
    ):
        init_deg = ExactBoxInit.from_canonical_model(
            cm2, c2i2, embed_dim=3, n_components=3, seed=0
        )
    # B has |E_B| == 2 so k' == 2 (TOP uses the full-domain bbox directly), and all points were assigned to cluster 0
    assert init_deg.report["n_degenerate_kmeans"] == 1
    assert not np.any(np.isnan(init_deg.centers))


def test_apply_to_elbe_and_box2el_write_same_values(t3_dataset: PathDataset) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, seed=0)
    n_classes = len(init.class_to_id)
    box2el = BoxSquaredELModule(nb_ont_classes=n_classes, nb_rels=1, embed_dim=5)
    elbe = ELBEModule(nb_ont_classes=n_classes, nb_rels=1, embed_dim=5)

    n1 = init.apply_to_module(box2el)
    n2 = init.apply_to_module(elbe)
    assert n1 == n2 == n_classes

    c, o, mask = init.as_tensors()
    assert torch.allclose(box2el.class_center.weight[mask], elbe.class_embed.weight[mask])
    assert torch.allclose(box2el.class_offset.weight[mask], elbe.class_offset.weight[mask])
    assert torch.allclose(box2el.class_center.weight[mask], c[mask, 0])
    assert torch.allclose(box2el.class_offset.weight[mask], o[mask, 0])

    # Flip sign of offsets in ELBE: anchor_loss uses |offset| so still 0
    with torch.no_grad():
        elbe.class_offset.weight.mul_(-1.0)
    assert float(init.anchor_loss(box2el).item()) <= 1e-12
    assert float(init.anchor_loss(elbe).item()) <= 1e-12


def test_milp_falls_back_on_t3(t3_dataset: PathDataset) -> None:
    init = ExactBoxInit.from_dataset(
        t3_dataset, embed_dim=4, method="milp", milp_time_limit=20
    )
    assert init.method == "projected" and init.report["fallback"].startswith(
        "NotBoxRepresentable"
    )

    # Also verify fallback on BoxModelNotFound
    with patch(
        "mowl.geometric.init.BoxModelBuilder.build",
        side_effect=BoxModelNotFound("solver timed out"),
    ):
        init_nf = ExactBoxInit.from_dataset(
            t3_dataset, embed_dim=4, method="milp", milp_time_limit=1.0
        )
    assert init_nf.method == "projected"
    assert init_nf.report["fallback"].startswith("BoxModelNotFound")


def test_milp_on_helly_ok_preserves_disjointness(helly_ok_dataset: PathDataset) -> None:
    embed_dim = 4
    pad = 0.05
    init = ExactBoxInit.from_dataset(
        helly_ok_dataset,
        embed_dim=embed_dim,
        method="milp",
        pad=pad,
        milp_time_limit=20,
    )
    assert init.method == "milp"
    assert "fallback" not in init.report
    assert "milp_dimension" in init.report
    d_prime = init.report["milp_dimension"]
    assert 1 <= d_prime < embed_dim

    cm = CanonicalModel.from_dataset(helly_ok_dataset)
    box_model = BoxModelBuilder(cm, d_max=embed_dim, time_limit=20).build()
    assert d_prime == box_model.dimension
    model_boxes = box_model.boxes()
    c2i = init.class_to_id
    big_m = float(box_model.M)

    def to_unit(x: np.ndarray) -> np.ndarray:
        # Affine map [0, M] -> [-scale, scale] (scale = 1.0) applied by ExactBoxInit
        return 2.0 * np.asarray(x, dtype=np.float64) / big_m - 1.0

    for name in cm.names:
        if name not in c2i:
            continue
        lo_m, hi_m = model_boxes[name]
        lo_i, hi_i = box(init, c2i[name])
        np.testing.assert_allclose(lo_i[:d_prime], to_unit(lo_m), atol=1e-6)
        np.testing.assert_allclose(hi_i[:d_prime], to_unit(hi_m), atol=1e-6)
        np.testing.assert_allclose(
            init.centers[c2i[name], 0, :d_prime],
            0.5 * (to_unit(lo_m) + to_unit(hi_m)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            init.offsets[c2i[name], 0, :d_prime],
            0.5 * (to_unit(hi_m) - to_unit(lo_m)),
            atol=1e-6,
        )
        assert (lo_i[:d_prime] >= -1.0 - 1e-6).all() and (hi_i[:d_prime] <= 1.0 + 1e-6).all()
        np.testing.assert_allclose(
            lo_i[d_prime:], np.float32(-1.0 - pad), atol=1e-6
        )
        np.testing.assert_allclose(
            hi_i[d_prime:], np.float32(1.0 + pad), atol=1e-6
        )

    # Every NF2 bottom axiom A and B <= bot has disjoint boxes in R^4
    disjoint_checked = 0
    for ax in cm.axioms:
        if isinstance(ax, NF2) and ax.sup == BOTTOM:
            lo_a, hi_a = box(init, c2i[ax.sub1])
            lo_b, hi_b = box(init, c2i[ax.sub2])
            assert np.any(hi_a < lo_b - 1e-6) or np.any(hi_b < lo_a - 1e-6)
            disjoint_checked += 1
    assert disjoint_checked > 0

    # Every entailed subsumption A <= B has box_A <= box_B
    for a in cm.names:
        for b in cm.names:
            if cm.entails(a, b) and cm.is_satisfiable(a):
                lo_a, hi_a = box(init, c2i[a])
                lo_b, hi_b = box(init, c2i[b])
                assert (lo_b - 1e-6 <= lo_a).all() and (hi_a <= hi_b + 1e-6).all()

    # If a satisfiable concept is missing from box_model.boxes(), raise InternalConsistencyError
    incomplete_boxes = dict(model_boxes)
    incomplete_boxes.pop(NS + "A", None)
    with patch.object(box_model, "boxes", return_value=incomplete_boxes), patch(
        "mowl.geometric.init.BoxModelBuilder.build", return_value=box_model
    ):
        with pytest.raises(InternalConsistencyError):
            ExactBoxInit.from_canonical_model(
                cm, c2i, embed_dim=embed_dim, method="milp"
            )


def test_apply_rejects_dimension_mismatch(t3_dataset: PathDataset) -> None:
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, seed=0)
    n_classes = len(init.class_to_id)

    # Wrong embed_dim on BoxSquaredELModule
    wrong_dim = BoxSquaredELModule(nb_ont_classes=n_classes, nb_rels=1, embed_dim=4)
    with pytest.raises(UnsupportedRegion):
        init.apply_to_module(wrong_dim)
    with pytest.raises(UnsupportedRegion):
        init.anchor_loss(wrong_dim)

    # Wrong nb_ont_classes
    wrong_rows = BoxSquaredELModule(nb_ont_classes=n_classes + 1, nb_rels=1, embed_dim=5)
    with pytest.raises(UnsupportedRegion):
        init.apply_to_module(wrong_rows)

    # Multi-component init applied to k=1 module (Box²EL / ELBE)
    init_k2 = ExactBoxInit.from_dataset(t3_dataset, embed_dim=5, n_components=2, seed=0)
    box2el = BoxSquaredELModule(nb_ont_classes=n_classes, nb_rels=1, embed_dim=5)
    elbe = ELBEModule(nb_ont_classes=n_classes, nb_rels=1, embed_dim=5)
    with pytest.raises(UnsupportedRegion):
        init_k2.apply_to_module(box2el)
    with pytest.raises(UnsupportedRegion):
        init_k2.apply_to_module(elbe)

    # Unsupported module without class_center or class_embed
    with pytest.raises(UnsupportedRegion):
        init.apply_to_module(nn.Linear(5, 5))


def test_extensions_match_naive(roles_cm: CanonicalModel) -> None:
    comp = roles_cm.completion
    all_concepts = (set(comp.concept_names) | set(roles_cm.concept_names) | {TOP}) - {
        BOTTOM
    }
    naive = {}
    for cname in all_concepts:
        if comp.is_unsatisfiable(cname):
            naive[cname] = frozenset()
        else:
            naive[cname] = frozenset(
                idx
                for idx, b_name in enumerate(roles_cm.names)
                if cname in comp.sub_super.get(b_name, set())
            )
    naive[BOTTOM] = frozenset()

    assert roles_cm._extensions == naive
    for cname, expected in naive.items():
        assert roles_cm.extension(cname) == expected


def test_import_without_jvm() -> None:
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from mowl.geometric import ExactBoxInit, init; "
            "assert 'mowl.geometric.completion' not in sys.modules; "
            "print('ok')"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"stderr: {res.stderr}"
    assert "ok" in res.stdout
