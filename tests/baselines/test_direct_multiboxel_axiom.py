"""The direct MultiBoxEL module with ``objective="axiom"`` is exactly its loss functions
(:mod:`mowl.baselines`)."""
from __future__ import annotations

import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.baselines.nn.el.multiboxel.module import MultiBoxELModule
import mowl.baselines.nn.el.multiboxel.losses as L
from tests.geometric.helpers import _random_batch


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
            mod.gci0_loss(gci0_rows).reshape(-1),
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
            mod.gci0_bot_loss(gci0_bot_rows).reshape(-1),
            L.gci0_bot_loss(
                gci0_bot_rows,
                mod.class_offset,
                n_components=k,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        assert th.equal(
            mod.gci1_loss(gci1_rows).reshape(-1),
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
            mod.gci1_bot_loss(gci1_bot_rows).reshape(-1),
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
                mod.gci2_loss(gci2_rows, neg=neg).reshape(-1),
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
                mod.object_property_assertion_loss(opa_rows, neg=neg).reshape(-1),
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
            mod.gci3_loss(gci3_rows).reshape(-1),
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
            mod.gci3_bot_loss(gci3_bot_rows).reshape(-1),
            L.gci3_bot_loss(
                gci3_bot_rows,
                mod.head_offset,
                aggregation=mod.component_aggregation,
                tau=mod.tau,
            ),
        )
        assert th.equal(
            mod.class_assertion_loss(ca_rows).reshape(-1),
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
