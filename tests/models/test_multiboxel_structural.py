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
from mowl.nn.el.structural import nonempty_measure
from tests.geometric.helpers import NS, _random_batch, fixture_path
from tests.models.test_multiboxel import _triple_nonempty


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


def _strict_structural_loss(model: MultiBoxEL) -> float:
    """Evaluate total structural GCI loss + existence term at margin=0, size_reg=0."""
    mod = model.module
    old_margin = mod.margin
    old_size_reg = mod.size_reg
    try:
        mod.margin = 0.0
        mod.size_reg = 0.0
        mod.eval()
        total = 0.0
        with th.no_grad():
            for gci_name, gci_ds in model.training_datasets.items():
                if len(gci_ds) == 0:
                    continue
                scores = mod(gci_ds[:], gci_name)
                total += float(scores.mean().item())
            total += float(mod.structural_extra_loss().item())
        return total
    finally:
        mod.margin = old_margin
        mod.size_reg = old_size_reg


@pytest.mark.slow
def test_t3_structural_random_init_finds_model(tmp_path: Path) -> None:
    """At least 1 of 5 seeds (0..4) reaches strict structural loss <= 1e-6 at margin 0 with empty X1 cap X2 cap X3."""
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
        ok = strict_loss <= 1e-6 and has_nonempty_slots and not triple_ne
        outcomes.append((seed, strict_loss, triple_ne, ok))
        if ok:
            successes += 1

    assert successes >= 1, (
        f"Expected at least 1 of 5 seeds to reach strict_loss <= 1e-6 with empty X1 cap X2 cap X3; "
        f"got outcomes={outcomes}"
    )
