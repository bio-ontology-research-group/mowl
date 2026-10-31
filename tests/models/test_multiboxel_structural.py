"""Model-level tests for MultiBoxEL structural objective."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
from mowl.models import MultiBoxEL
from mowl.nn import MultiBoxELModule
import mowl.nn.el.multiboxel.losses as L
from mowl.nn.el.structural import (
    build_structural_tables,
    nonempty_measure,
    pair_size,
    region_size,
)
from tests.geometric.helpers import (
    NS,
    _random_batch,
    _strict_structural_loss,
    _triple_nonempty,
    fixture_path,
)


def test_structural_train_builds_tables_and_masks(tmp_path: Path) -> None:
    """T3 dataset, 2 epochs: definition_table.n_defined == 3, 3 gci1 rows masked, existence_targets.pairs has 3 rows with cm."""
    ds = PathDataset(fixture_path("t3.owl"))
    cm = CanonicalModel.from_dataset(ds)

    model = MultiBoxEL(
        ds,
        embed_dim=2,
        n_components=2,
        objective="structural",
        max_components=16,
        canonical_model=cm,
        neg_sampling_gcis=[],
        model_filepath=str(tmp_path / "m_struct.pt"),
    )
    assert model.module.definition_table is None
    assert model.module.existence_targets is None

    model.train(epochs=2, validate_every=10)

    table = model.module.definition_table
    targets = model.module.existence_targets
    assert table is not None
    assert targets is not None
    assert table.n_defined == 3

    gci1_rows = model.training_datasets["gci1"][:]
    masked_gci1 = table.masked_gci1(gci1_rows)
    assert int(masked_gci1.sum().item()) == 3

    assert targets.pairs.shape == (3, 2)


def test_axiom_objective_unchanged() -> None:
    """On a fixed random module with objective='axiom', each module.gciX_loss(rows) equals the direct L.gciX_loss(...) call."""
    n_classes, n_rels, n_inds, d, k = 10, 3, 4, 4, 2
    th.manual_seed(12345)
    mod = MultiBoxELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
        gamma=0.05,
        delta=2.0,
        objective="axiom",
    )
    gen = th.Generator().manual_seed(54321)

    gci0_rows = _random_batch("gci0", 2, 16, n_classes, n_rels, n_inds, gen)
    gci0_bot_rows = _random_batch("gci0_bot", 2, 16, n_classes, n_rels, n_inds, gen)
    gci1_rows = _random_batch("gci1", 3, 16, n_classes, n_rels, n_inds, gen)
    gci1_bot_rows = _random_batch("gci1_bot", 3, 16, n_classes, n_rels, n_inds, gen)
    gci2_rows = _random_batch("gci2", 3, 16, n_classes, n_rels, n_inds, gen)
    gci3_rows = _random_batch("gci3", 3, 16, n_classes, n_rels, n_inds, gen)
    gci3_bot_rows = _random_batch("gci3_bot", 3, 16, n_classes, n_rels, n_inds, gen)
    ca_rows = _random_batch("class_assertion", 2, 16, n_classes, n_rels, n_inds, gen)
    opa_rows = _random_batch(
        "object_property_assertion", 3, 16, n_classes, n_rels, n_inds, gen
    )

    for is_train in (True, False):
        mod.train(is_train)

        assert th.equal(
            mod.gci0_loss(gci0_rows),
            L.gci0_loss(
                gci0_rows,
                mod.class_center,
                mod.class_offset,
                mod.gamma,
                n_components=k,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        assert th.equal(
            mod.gci0_bot_loss(gci0_bot_rows),
            L.gci0_bot_loss(
                gci0_bot_rows,
                mod.class_offset,
                n_components=k,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        assert th.equal(
            mod.gci1_loss(gci1_rows),
            L.gci1_loss(
                gci1_rows,
                mod.class_center,
                mod.class_offset,
                mod.gamma,
                n_components=k,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        assert th.equal(
            mod.gci1_bot_loss(gci1_bot_rows),
            L.gci1_bot_loss(
                gci1_bot_rows,
                mod.class_center,
                mod.class_offset,
                mod.gamma,
                n_components=k,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        for neg in (False, True):
            assert th.equal(
                mod.gci2_loss(gci2_rows, neg=neg),
                L.gci2_loss(
                    gci2_rows,
                    mod.class_center,
                    mod.class_offset,
                    mod.head_center,
                    mod.head_offset,
                    mod.tail_center,
                    mod.tail_offset,
                    mod.bump_classes,
                    mod.gamma,
                    mod.delta,
                    n_components=k,
                    neg=neg,
                    aggregation=mod.component_aggregation,
                    tau=mod.tau,
                ),
            )
            assert th.equal(
                mod.object_property_assertion_loss(opa_rows, neg=neg),
                L.object_property_assertion_loss(
                    opa_rows,
                    mod.ind_center,
                    mod.ind_offset,
                    mod.head_center,
                    mod.head_offset,
                    mod.tail_center,
                    mod.tail_offset,
                    mod.bump_individuals,
                    mod.gamma,
                    mod.delta,
                    neg=neg,
                    aggregation=mod.component_aggregation,
                    tau=mod.tau,
                ),
            )
        assert th.equal(
            mod.gci3_loss(gci3_rows),
            L.gci3_loss(
                gci3_rows,
                mod.class_center,
                mod.class_offset,
                mod.head_center,
                mod.head_offset,
                mod.tail_center,
                mod.tail_offset,
                mod.bump_classes,
                mod.gamma,
                n_components=k,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        assert th.equal(
            mod.gci3_bot_loss(gci3_bot_rows),
            L.gci3_bot_loss(
                gci3_bot_rows,
                mod.head_offset,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        assert th.equal(
            mod.class_assertion_loss(ca_rows),
            L.class_assertion_loss(
                ca_rows,
                mod.ind_center,
                mod.ind_offset,
                mod.class_center,
                mod.class_offset,
                mod.gamma,
                n_components=k,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )


def _active_nonempty_boxes(
    model: MultiBoxEL, name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cen, w) of active, non-empty slots for class `name`."""
    idx = th.tensor([model.class_index_dict[NS + name]], dtype=th.long)
    with th.no_grad():
        r = model.module.class_regions(idx)
        ne = nonempty_measure(r.w[0])
        keep = r.active[0] & (ne > 0.0)
        cen = r.cen[0, keep].detach().cpu().numpy()
        w = r.w[0, keep].detach().cpu().numpy()
    return cen, w


def test_multiboxel_structural_survives_reload_and_narrow_tensor_raises(
    tmp_path: Path,
) -> None:
    """Train 3 epochs, save, load_best_model() / from_pretrained(): every GCI loss tensor and n_defined match pre-reload."""
    ds = PathDataset(fixture_path("t3.owl"))
    cm = CanonicalModel.from_dataset(ds)
    ckpt = str(tmp_path / "m_reload.pt")

    th.manual_seed(0)
    np.random.seed(0)
    model = MultiBoxEL(
        ds,
        embed_dim=2,
        n_components=2,
        objective="structural",
        max_components=16,
        canonical_model=cm,
        neg_sampling_gcis=[],
        model_filepath=ckpt,
    )
    model.train(epochs=3, validate_every=10)
    th.save(model.module.state_dict(), ckpt)

    model.module.eval()
    assert model.module.definition_table is not None
    pre_n_defined = model.module.definition_table.n_defined
    assert pre_n_defined == 3
    with th.no_grad():
        pre_gci = {
            name: model.module(gci_ds[:], name).detach().clone()
            for name, gci_ds in model.training_datasets.items()
            if len(gci_ds) > 0
        }
        pre_extra = model.module.structural_extra_loss().detach().clone()

    model.load_best_model()
    assert model.module.definition_table is not None
    assert model.module.existence_targets is not None
    assert model.module.definition_table.n_defined == pre_n_defined
    with th.no_grad():
        for name, expected in pre_gci.items():
            actual = model.module(model.training_datasets[name][:], name)
            assert th.equal(actual, expected), f"Mismatch on {name} after load_best_model"
        assert th.equal(model.module.structural_extra_loss(), pre_extra)

    fresh = MultiBoxEL(
        ds,
        embed_dim=2,
        n_components=2,
        objective="structural",
        max_components=16,
        canonical_model=cm,
        neg_sampling_gcis=[],
        model_filepath=ckpt,
    )
    fresh.from_pretrained(ckpt)
    fresh.module.eval()
    assert fresh.module.definition_table is not None
    assert fresh.module.definition_table.n_defined == pre_n_defined
    with th.no_grad():
        for name, expected in pre_gci.items():
            actual = fresh.module(model.training_datasets[name][:], name)
            assert th.equal(actual, expected), f"Mismatch on {name} after from_pretrained"
            actual_fresh_rows = fresh.module(fresh.training_datasets[name][:], name)
            assert th.equal(actual_fresh_rows.sort().values, expected.sort().values)
        assert th.equal(fresh.module.structural_extra_loss(), pre_extra)

    # Too-narrow normal-form tensors must raise ValueError in build_structural_tables
    class _DummyModel:
        class_index_dict = {"http://mowl.borg/geo#A": 0, "http://mowl.borg/geo#B": 1}
        n_components = 2
        max_components = 4
        structural_definitions = True
        canonical_model = None
        training_datasets = {"gci0": th.tensor([[0]], dtype=th.long)}

    with pytest.raises(ValueError, match="columns"):
        build_structural_tables(_DummyModel())


def test_pizza_role_auxiliary_definitions_detected(tmp_path: Path) -> None:
    """Model-level test: Pizza-style role-auxiliary definitions (and Pizza-EL training ontology) produce n_defined > 0, with gci2/gci3 losses still active."""
    from mowl.owlapi import OWLAPIAdapter
    from org.semanticweb.owlapi.formats import FunctionalSyntaxDocumentFormat
    from org.semanticweb.owlapi.model import IRI
    import java.io

    # 1. Self-contained 6-class Pizza-style ontology:
    # CheeseyPizza == Pizza cap exists hasTopping.CheeseTopping
    # MeatyPizza   == Pizza cap exists hasTopping.MeatTopping
    # SpicyPizza   == Pizza cap exists hasTopping.SpicyTopping
    ad = OWLAPIAdapter()
    ns = "http://mowl.borg/pizza_mini#"
    ont = ad.create_ontology("http://mowl.borg/pizza_mini")
    pizza = ad.create_class(ns + "Pizza")
    cheesey = ad.create_class(ns + "CheeseyPizza")
    meaty = ad.create_class(ns + "MeatyPizza")
    spicy = ad.create_class(ns + "SpicyPizza")
    cheese_top = ad.create_class(ns + "CheeseTopping")
    meat_top = ad.create_class(ns + "MeatTopping")
    spicy_top = ad.create_class(ns + "SpicyTopping")
    has_topping = ad.create_object_property(ns + "hasTopping")

    for defined_cls, filler_cls in (
        (cheesey, cheese_top),
        (meaty, meat_top),
        (spicy, spicy_top),
    ):
        rhs = ad.create_object_intersection_of(
            pizza, ad.create_object_some_values_from(has_topping, filler_cls)
        )
        ad.owl_manager.addAxiom(ont, ad.create_equivalent_classes(defined_cls, rhs))

    mini_owl = tmp_path / "pizza_mini.owl"
    ad.owl_manager.saveOntology(
        ont,
        FunctionalSyntaxDocumentFormat(),
        IRI.create(java.io.File(str(mini_owl)).toURI()),
    )

    ds_mini = PathDataset(str(mini_owl))
    model_mini = MultiBoxEL(
        ds_mini,
        embed_dim=4,
        n_components=4,
        max_components=16,
        objective="structural",
        neg_sampling_gcis=[],
        model_filepath=str(tmp_path / "mini.pt"),
    )
    model_mini.train(epochs=1, validate_every=10)
    assert model_mini.module.definition_table is not None
    assert model_mini.module.definition_table.n_defined == 3
    assert model_mini.module.definition_table.max_level == 1
    # gci2 and gci3 rows are NOT masked (they still train Aux's region)
    gci2_rows = model_mini.training_datasets["gci2"][:]
    gci3_rows = model_mini.training_datasets["gci3"][:]
    assert len(gci2_rows) == 3
    assert len(gci3_rows) == 3

    # 2. Full Pizza-EL training ontology from experiments (loads in < 1s from cache)
    pizza_ont_path = Path(
        "/home/leechuck/Public/software/faithful-dl-embedding/experiments/data/pizza-el/ontology.owl"
    )
    if pizza_ont_path.exists():
        ds_pizza = PathDataset(str(pizza_ont_path))
        model_pizza = MultiBoxEL(
            ds_pizza,
            embed_dim=4,
            n_components=4,
            max_components=16,
            objective="structural",
            neg_sampling_gcis=[],
            model_filepath=str(tmp_path / "pizza.pt"),
        )
        table_pizza, _, _ = build_structural_tables(
            model_pizza, k=4, max_components=16, enabled=True
        )
        assert table_pizza.n_defined > 0
        assert table_pizza.n_defined == 4
        assert table_pizza.max_level == 1


@pytest.mark.slow
def test_t3_structural_random_init_finds_model(tmp_path: Path) -> None:
    """At least 1 of 5 seeds (0..4) reaches strict structural loss <= 1e-6 at margin 0,
    with min(region_size) > 0 on all target classes, min(pair_size) > 0 on all target pairs,
    and empty X1 cap X2 cap X3.
    """
    ds = PathDataset(fixture_path("t3.owl"))
    cm = CanonicalModel.from_dataset(ds)

    successes = 0
    outcomes = []
    for seed in range(5):
        th.manual_seed(seed)
        np.random.seed(seed)
        model = MultiBoxEL(
            ds,
            embed_dim=2,
            n_components=4,
            margin=0.02,
            learning_rate=0.01,
            objective="structural",
            max_components=16,
            size_reg=1e-2,
            reseed_every=50,
            points_seed=seed,
            canonical_model=cm,
            neg_sampling_gcis=[],
            model_filepath=str(tmp_path / f"t3_struct_s{seed}.pt"),
        )
        model.train(epochs=1500, validate_every=2000)

        strict_loss = _strict_structural_loss(model)
        mod = model.module
        targets = mod.existence_targets
        assert targets is not None
        with th.no_grad():
            cls_regs = mod.class_regions(targets.classes)
            min_cls_ne = float(region_size(cls_regs).min().item())
            pa_regs = mod.class_regions(targets.pairs[:, 0])
            pb_regs = mod.class_regions(targets.pairs[:, 1])
            min_pair_ne = float(pair_size(pa_regs, pb_regs).min().item())

        b_x1 = _active_nonempty_boxes(model, "X1")
        b_x2 = _active_nonempty_boxes(model, "X2")
        b_x3 = _active_nonempty_boxes(model, "X3")
        has_nonempty_slots = (
            b_x1[0].shape[0] > 0 and b_x2[0].shape[0] > 0 and b_x3[0].shape[0] > 0
        )
        triple_ne = (
            _triple_nonempty(b_x1, b_x2, b_x3, tol=0.0)
            if has_nonempty_slots
            else True
        )
        ok = (
            strict_loss <= 1e-6
            and min_cls_ne > 0.0
            and min_pair_ne > 0.0
            and has_nonempty_slots
            and not triple_ne
        )
        outcomes.append((seed, strict_loss, min_cls_ne, min_pair_ne, triple_ne, ok))
        if ok:
            successes += 1

    for seed, strict_loss, min_cls_ne, min_pair_ne, triple_ne, ok in outcomes:
        print(
            f"seed={seed}: strict_loss={strict_loss:.3e}, min_cls_ne={min_cls_ne:.4f}, "
            f"min_pair_ne={min_pair_ne:.4f}, triple_ne={triple_ne}, ok={ok}"
        )

    assert successes >= 1, (
        f"Expected at least 1 of 5 seeds to reach strict_loss <= 1e-6 with positive target existence "
        f"and empty X1 cap X2 cap X3; got outcomes={outcomes}"
    )
