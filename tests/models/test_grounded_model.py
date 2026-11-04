"""Tests for GroundedEL model, training loop, and evaluation (§3, §4, §5, §7, §8)."""
from __future__ import annotations

import os
import tempfile
from typing import Dict, Set

import torch as th
import pytest

from java.io import File
from org.semanticweb.owlapi.model import IRI

from mowl.datasets import PathDataset
from mowl.models import GroundedEL
from mowl.owlapi import OWLAPIAdapter
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import BOTTOM, TOP
from tests.geometric.helpers import NS, fixture_path, t3_cm, t3_dataset


def _least_model_labels_from_cm(
    model: GroundedEL, cm: CanonicalModel
) -> Dict[int, Set[int]]:
    """Build element-to-classes label dictionary from a CanonicalModel for test fixtures."""
    labels: Dict[int, Set[int]] = {}
    for c_iri, c_idx in model.class_index_dict.items():
        if c_iri in (TOP, BOTTOM) or not cm.is_satisfiable(c_iri):
            labels[c_idx] = set()
        else:
            sups = cm.completion.sub_super.get(c_iri, set())
            labels[c_idx] = {
                model.class_index_dict[sup_iri]
                for sup_iri in sups
                if sup_iri in model.class_index_dict and sup_iri not in (TOP, BOTTOM)
            }
    for ind_iri, ind_idx in model.individual_index_dict.items():
        el_idx = model.nb_classes + int(ind_idx)
        nom_name = cm.individuals.get(ind_iri)
        if nom_name is None or not cm.is_satisfiable(nom_name):
            labels[el_idx] = set()
        else:
            sups = cm.completion.sub_super.get(nom_name, set())
            labels[el_idx] = {
                model.class_index_dict[sup_iri]
                for sup_iri in sups
                if sup_iri in model.class_index_dict and sup_iri not in (TOP, BOTTOM)
            }
    return labels


def test_t3_elements_init_zero_epochs_exact(
    t3_dataset: PathDataset, t3_cm: CanonicalModel
) -> None:
    """init='elements' on T3 with least-model labels gives evaluate_model()['exact'] is True with 0 epochs."""
    probe = GroundedEL(t3_dataset, embed_dim=2, n_components=4, epochs=0, seed=0)
    labels = _least_model_labels_from_cm(probe, t3_cm)

    model = GroundedEL(
        t3_dataset,
        embed_dim=2,
        n_components=4,
        n_role_components=2,
        epochs=0,
        init="elements",
        init_labels=labels,
        seed=0,
    )
    history = model.train()
    assert history == []

    crisp = model.crisp_report(n_eval=2000)
    assert set(crisp.keys()) == {"elem_viol", "region_viol"}
    assert crisp["elem_viol"] == 0.0
    assert crisp["region_viol"] == 0.0

    metrics = model.evaluate_model(use_reasoner=True, n_eval=2000)
    assert metrics["elem_viol"] == 0.0
    assert metrics["region_viol"] == 0.0
    assert metrics["exist"] == 1.0
    assert metrics["dead_ok"] is True
    assert metrics["faithful_named_precision"] == 1.0
    assert metrics["faithful_named_recall"] == 1.0
    assert metrics["exact"] is True

    no_reasoner_metrics = model.evaluate_model(use_reasoner=False, n_eval=2000)
    assert no_reasoner_metrics["exact"] is True
    assert no_reasoner_metrics["faithful_named_precision"] is None
    assert no_reasoner_metrics["faithful_named_recall"] is None


def test_t3_random_init_200_epochs_loss_decreases(
    t3_dataset: PathDataset,
) -> None:
    """Fast 200-epoch training run on T3 from random init decreases total loss."""
    model = GroundedEL(
        t3_dataset,
        embed_dim=2,
        n_components=4,
        n_role_components=2,
        epochs=200,
        learning_rate=0.01,
        reseed_every=50,
        seed=0,
        log_every=0,
    )
    history = model.train()
    assert len(history) == 200
    assert history[-1]["total"] < history[0]["total"]


def test_t3_random_init_three_seeds_at_least_two_exact(
    t3_dataset: PathDataset,
) -> None:
    """T3 from random init (K=4, d=2, <= 3000 epochs, 3 seeds) reaches exact=True on at least 2/3 seeds."""
    exact_count = 0
    for seed in (0, 1, 2):
        model = GroundedEL(
            t3_dataset,
            embed_dim=2,
            n_components=4,
            n_role_components=2,
            epochs=100,
            learning_rate=0.01,
            reseed_every=50,
            seed=seed,
            log_every=0,
        )
        history = model.train()
        assert history[-1]["total"] < history[0]["total"]
        ev = model.evaluate_model(use_reasoner=True, n_eval=2000)
        if ev["exact"]:
            exact_count += 1

    assert exact_count >= 2, f"Expected at least 2/3 exact seeds on T3, got {exact_count}/3"


def test_box2el_counterexample_t1_elements_init_exact() -> None:
    """Box2EL counterexample T1 = {A <= exists r.B, exists r.C <= D, exists r.C <= E, D cap E <= bot} is exact under init='elements'."""
    base_tmp = os.environ.get("TMPDIR") or os.path.expanduser("~/tmp")
    os.makedirs(base_tmp, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=base_tmp, prefix="mowl_grounded_t1_") as workdir:
        owl_path = os.path.join(workdir, "t1_counterexample.owl")

        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        df = adapter.data_factory
        ont = adapter.create_ontology("http://mowl.borg/geo/t1_counterexample")

        cls_objs = {
            name: adapter.create_class(NS + name) for name in ("A", "B", "C", "D", "E")
        }
        r_prop = adapter.create_object_property(NS + "r")
        bot_cls = adapter.create_class(BOTTOM)

        for c_obj in cls_objs.values():
            mgr.addAxiom(ont, df.getOWLDeclarationAxiom(c_obj))
        mgr.addAxiom(ont, df.getOWLDeclarationAxiom(r_prop))

        # A <= exists r.B
        mgr.addAxiom(
            ont,
            adapter.create_subclass_of(
                cls_objs["A"],
                adapter.create_object_some_values_from(r_prop, cls_objs["B"]),
            ),
        )
        # exists r.C <= D
        mgr.addAxiom(
            ont,
            adapter.create_subclass_of(
                adapter.create_object_some_values_from(r_prop, cls_objs["C"]),
                cls_objs["D"],
            ),
        )
        # exists r.C <= E
        mgr.addAxiom(
            ont,
            adapter.create_subclass_of(
                adapter.create_object_some_values_from(r_prop, cls_objs["C"]),
                cls_objs["E"],
            ),
        )
        # D cap E <= bot
        mgr.addAxiom(
            ont,
            adapter.create_subclass_of(
                adapter.create_object_intersection_of(cls_objs["D"], cls_objs["E"]),
                bot_cls,
            ),
        )

        mgr.saveOntology(ont, IRI.create(File(owl_path).toURI()))

        ds_t1 = PathDataset(owl_path)
        cm_t1 = CanonicalModel.from_dataset(ds_t1)
        for name in ("A", "B", "C", "D", "E"):
            assert cm_t1.is_satisfiable(NS + name), f"Expected {name} to be satisfiable in T1"

        probe = GroundedEL(ds_t1, embed_dim=2, n_components=4, epochs=0, seed=0)
        labels = _least_model_labels_from_cm(probe, cm_t1)

        model = GroundedEL(
            ds_t1,
            embed_dim=2,
            n_components=4,
            n_role_components=4,
            epochs=0,
            init="elements",
            init_labels=labels,
            seed=0,
        )
        metrics = model.evaluate_model(use_reasoner=True, n_eval=2000)
        assert metrics["elem_viol"] == 0.0
        assert metrics["region_viol"] == 0.0
        assert metrics["exist"] == 1.0
        assert metrics["dead_ok"] is True
        assert metrics["exact"] is True


def test_monotone_capacity_smoke(t3_dataset: PathDataset) -> None:
    """Loss at K and after expand_components(2K) is identical (R3 monotone capacity)."""
    k_init = 3
    model = GroundedEL(
        t3_dataset,
        embed_dim=2,
        n_components=k_init,
        n_role_components=2,
        epochs=0,
        n_region_samples=0,
        seed=42,
    )
    gen_before = th.Generator(device="cpu").manual_seed(777)
    losses_before = model.compute_batch_losses(generator=gen_before)

    model.module.expand_components(2 * k_init)
    assert model.module.n_components == 2 * k_init

    gen_after = th.Generator(device="cpu").manual_seed(777)
    losses_after = model.compute_batch_losses(generator=gen_after)

    for key in ("fact", "axioms", "region", "minimality", "total"):
        assert th.allclose(
            losses_before[key], losses_after[key], atol=1e-6
        ), f"Mismatch in {key} after expand_components: {losses_before[key]} vs {losses_after[key]}"


def test_grounded_model_argument_validation(t3_dataset: PathDataset) -> None:
    """Constructor validates hyperparameter ranges and init_labels requirement."""
    with pytest.raises(ValueError, match="init"):
        GroundedEL(t3_dataset, init="invalid")
    with pytest.raises(ValueError, match="init_labels"):
        GroundedEL(t3_dataset, init="elements", init_labels=None)
    with pytest.raises(ValueError, match="tau0"):
        GroundedEL(t3_dataset, tau0=0.0)
    with pytest.raises(ValueError, match="eps_min"):
        GroundedEL(t3_dataset, eps_min=-0.1)
