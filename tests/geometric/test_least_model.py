"""Tests for the indexed sparse least-model backend."""

from __future__ import annotations

import resource
import time

import numpy as np
import pytest

from mowl.geometric.least_model import (
    check_model,
    interval_realisation,
    least_model,
    least_model_from_model,
)

pytest_plugins = ("tests.geometric.helpers",)


def test_rules_lp_checker_and_pruning():
    args = dict(
        nb_rels=2,
        gci0=[[0, 1]],
        gci1=[[1, 2, 3]],
        gci2=[[0, 0, 2]],
        gci3=[[0, 2, 3]],
        role_chain=[[0, 0, 1]],
        gci0_bot=[[3, 4]],
    )
    it = least_model(5, **args)
    lp = least_model(5, method="lp", **args)
    assert np.array_equal(it.raw_labels.toarray(), lp.raw_labels.toarray())
    assert np.array_equal(it.raw_edges, lp.raw_edges)
    assert np.array_equal(it.dead, lp.dead)
    assert lp.lp_max_fractionality < 1e-7
    assert all(
        v == 0
        for v in check_model(
            it, **{k: v for k, v in args.items() if k != "nb_rels"}
        ).values()
    )
    assert not it.labels[it.dead].nnz
    assert not np.any(it.dead[it.edges[:, 0]]) and not np.any(it.dead[it.edges[:, 1]])


def test_role_chains_inclusions_and_abox_assertions():
    chain_args = dict(
        nb_rels=3,
        gci2=[[0, 0, 2], [2, 0, 4]],
        role_chain=[[0, 0, 1]],
        role_inclusion=[[1, 2]],
    )
    lm = least_model(5, **chain_args)
    assert (0, 4, 2) in set(map(tuple, lm.edges.tolist()))
    assert all(
        v == 0
        for v in check_model(
            lm, **{k: v for k, v in chain_args.items() if k != "nb_rels"}
        ).values()
    )
    lp = least_model(5, method="lp", **chain_args)
    assert np.array_equal(lm.raw_edges, lp.raw_edges)

    abox_args = dict(
        nb_rels=1,
        nb_individuals=2,
        class_assertion=[[0, 0], [1, 0]],
        object_property_assertion=[[0, 0, 1]],
        gci3=[[0, 0, 1]],
    )
    abox = least_model(2, **abox_args)
    assert abox.dense_labels()[2, 1]
    assert all(
        v == 0
        for v in check_model(
            abox, **{k: v for k, v in abox_args.items() if k not in ("nb_rels", "nb_individuals")}
        ).values()
    )


def test_random_positive_lp_objectives_match_iteration():
    args = dict(
        nb_rels=1, gci0=[[0, 1]], gci1=[[1, 2, 3]], gci2=[[0, 0, 2]], gci3=[[0, 2, 3]]
    )
    baseline = least_model(4, **args)
    nvars = baseline.n_blocks * 4 + baseline.n_blocks**2 + baseline.n_blocks
    for seed in range(3):
        weights = np.random.default_rng(seed).uniform(0.1, 3.0, size=nvars)
        got = least_model(4, method="lp", objective_weights=weights, **args)
        assert np.array_equal(got.raw_labels.toarray(), baseline.raw_labels.toarray())
        assert np.array_equal(got.raw_edges, baseline.raw_edges)
        assert np.array_equal(got.dead, baseline.dead)
        assert got.lp_max_fractionality < 1e-7


def test_quotient_bottom_and_folded_heads():
    separate = least_model(4, bot_id=3, gci0=[[0, 1]], gci0_bot=[[1, 3]])
    folded = least_model(4, bot_id=3, gci0=[[0, 1], [1, 3]])
    assert np.array_equal(separate.dead, folded.dead)
    assert np.array_equal(separate.raw_labels.toarray(), folded.raw_labels.toarray())
    for separate_rules, folded_rules in (
        (dict(gci1_bot=[[0, 1, 3]]), dict(gci1=[[0, 1, 3]])),
        (
            dict(gci2=[[0, 0, 1]], gci3_bot=[[0, 1, 3]]),
            dict(gci2=[[0, 0, 1]], gci3=[[0, 1, 3]]),
        ),
    ):
        a = least_model(4, nb_rels=1, bot_id=3, **separate_rules)
        b = least_model(4, nb_rels=1, bot_id=3, **folded_rules)
        assert np.array_equal(a.dead, b.dead)
        assert np.array_equal(a.raw_labels.toarray(), b.raw_labels.toarray())
        assert np.array_equal(a.raw_edges, b.raw_edges)
    # Merge disjoint class elements: their conjunction is now inhabited.
    q = least_model(3, gci1_bot=[[0, 1, 2]], element_map=[0, 0, 2])
    assert q.dead[0]
    assert q.consistent


def test_extra_facts_and_quotients_are_monotone():
    base = least_model(4, gci0=[[0, 1]])
    with_fact = least_model(4, gci0=[[0, 1]], extra_facts=[[0, 2]])
    assert np.all(~base.labels.toarray() | with_fact.labels.toarray())
    base_q = least_model(4, gci0=[[0, 1]])
    quotient = least_model(4, gci0=[[0, 1]], element_map=[0, 1, 1, 3])
    assert np.all(~base_q.dense_labels() | quotient.dense_labels())
    assert all(v == 0 for v in check_model(quotient, gci0=[[0, 1]]).values())


def test_subsumption_and_intervals():
    lm = least_model(6, gci0=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    result = interval_realisation(lm, order="dfs")
    recovered = np.zeros((lm.n_blocks, lm.nb_classes), dtype=bool)
    for c, intervals in result.intervals.items():
        for start, end in intervals:
            recovered[start:end + 1, c] = True
    assert np.array_equal(recovered, lm.labels[result.order].toarray())
    n_tree = 200
    tree = least_model(
        n_tree,
        gci0=[
            [child, parent]
            for child in range(n_tree - 1)
            for parent in [int((child - 1) // 2)]
            if child > 0
        ],
    )
    assert interval_realisation(tree, order="dfs").runs_per_class.max() == 1
    rng = np.random.default_rng(4)
    pairs = [(int(a), int(b)) for a, b in rng.integers(0, 6, size=(20, 2))]
    random_lm = least_model(6, gci0=pairs)
    matrix = random_lm.subsumption_matrix().toarray()
    for a in range(6):
        for b in range(6):
            assert bool(matrix[a, b]) == random_lm.holds(a, b)


def test_validation_and_sweep_limit():
    with pytest.raises(ValueError):
        least_model(3, gci0=[[0, 3]])
    with pytest.raises(RuntimeError):
        least_model(3, gci0=[[0, 1], [1, 2]], max_sweeps=1)


@pytest.mark.slow
def test_sparse_scale_synthetic():
    rng = np.random.default_rng(29)
    n = 50_000

    def dag_edges(count):
        block = rng.integers(0, n // 100, size=count) * 100
        a = block + rng.integers(0, 80, size=count)
        b = block + rng.integers(80, 100, size=count)
        return np.column_stack((a, b))

    gci0 = dag_edges(150_000)
    gci1 = np.column_stack(
        (
            rng.integers(0, n // 10, size=20_000),
            rng.integers(9 * n // 10, n, size=20_000),
            rng.integers(0, n, size=20_000),
        )
    )
    gci2 = np.column_stack(
        (
            rng.integers(n // 2, n, size=30_000),
            rng.integers(0, 10, size=30_000),
            rng.integers(n // 2, n, size=30_000),
        )
    )
    gci3 = np.column_stack(
        (
            rng.integers(10, 20, size=10_000),
            rng.integers(n // 2, n, size=10_000),
            rng.integers(0, n, size=10_000),
        )
    )
    started = time.monotonic()
    lm = least_model(
        n,
        nb_rels=20,
        gci0=gci0,
        gci1=gci1,
        gci2=gci2,
        gci3=gci3,
        role_chain=[[0, 0, 0]],
    )
    elapsed = time.monotonic() - started
    assert elapsed < 600
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 30 * 1024 * 1024
    assert all(
        v == 0
        for v in check_model(
            lm, gci0=gci0, gci1=gci1, gci2=gci2, gci3=gci3, role_chain=[[0, 0, 0]]
        ).values()
    )
    real = interval_realisation(lm, "dfs")
    assert real.runs_per_class.sum() >= (lm.labels.getnnz(axis=0) > 0).sum()
    sub = lm.subsumption_matrix(np.arange(5_000), empty_subsumed_by_all=False)
    assert sub.diagonal().all()
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 30 * 1024 * 1024


def test_t3_groundedel_integration(t3_dataset, t3_cm):
    from mowl.models import GroundedEL
    from mowl.owlapi.defaults import BOT, TOP

    probe = GroundedEL(t3_dataset, embed_dim=2, n_components=2, epochs=0, seed=0)
    lm = least_model_from_model(probe)
    dense = lm.dense_labels()
    expected = np.zeros_like(dense)
    for iri, idx in probe.class_index_dict.items():
        if iri == TOP and lm.top_id is not None:
            expected[idx, lm.top_id] = True
        if iri not in (TOP, BOT) and t3_cm.is_satisfiable(iri):
            for sup in t3_cm.completion.sub_super.get(iri, set()):
                if sup in probe.class_index_dict and sup not in (TOP, BOT):
                    expected[idx, probe.class_index_dict[sup]] = True
            if lm.top_id is not None:
                expected[idx, lm.top_id] = True
    for iri, ind_idx in probe.individual_index_dict.items():
        nominal = t3_cm.individuals.get(iri)
        row = probe.nb_classes + int(ind_idx)
        if nominal is not None and t3_cm.is_satisfiable(nominal):
            for sup in t3_cm.completion.sub_super.get(nominal, set()):
                if sup in probe.class_index_dict and sup not in (TOP, BOT):
                    expected[row, probe.class_index_dict[sup]] = True
            if lm.top_id is not None:
                expected[row, lm.top_id] = True
    assert np.array_equal(dense, expected)
    exact = GroundedEL(
        t3_dataset,
        embed_dim=2,
        n_components=probe.nb_elements,
        epochs=0,
        seed=0,
        init="elements",
        init_labels=dense,
        init_edges=lm.element_edges(),
    )
    exact.train()
    report = exact.crisp_report(n_eval=200)
    assert report["elem_viol"] == 0
    assert report["region_viol"] == 0
    tables = {k: getattr(v, "data", v) for k, v in probe.training_datasets.items()}
    assert all(v == 0 for v in check_model(lm, **tables).values())


def _random_theory(rng, nc=9, nr=2, ni=3):
    bot = nc - 1
    cls = lambda k: rng.integers(0, nc - 1, size=k)  # noqa: E731
    return dict(
        gci0=np.column_stack((cls(6), cls(6))),
        gci1=np.column_stack((cls(3), cls(3), cls(3))),
        gci2=np.column_stack((cls(4), rng.integers(0, nr, 4), cls(4))),
        gci3=np.column_stack((rng.integers(0, nr, 3), cls(3), cls(3))),
        gci1_bot=np.column_stack((cls(2), cls(2), np.full(2, bot))),
        gci3_bot=np.column_stack((rng.integers(0, nr, 1), cls(1), np.full(1, bot))),
        role_inclusion=np.array([[0, 1]]),
        role_chain=np.array([[1, 1, 1]]),
        class_assertion=np.column_stack((rng.integers(0, ni, 3), cls(3))),
        object_property_assertion=np.column_stack(
            (rng.integers(0, ni, 3), rng.integers(0, nr, 3), rng.integers(0, ni, 3))
        ),
    )


def test_interval_realisation_with_dead_blocks():
    # bottom's own block is always dead; C is unsatisfiable; r-edges exist
    tables = dict(
        gci0=[[1, 0], [2, 0], [3, 1], [3, 2]],
        gci1_bot=[[1, 2, 4]],
        gci2=[[1, 0, 2]],
    )
    lm = least_model(5, nb_rels=1, bot_id=4, **tables)
    assert lm.unsatisfiable_classes().tolist() == [3]
    for order in ("dfs", "identity"):
        real = interval_realisation(lm, order)
        assert sorted(real.order.tolist()) == [0, 1, 2]
        rank = {b: i for i, b in enumerate(real.order.tolist())}
        for c in range(5):
            got = {i for lo, hi in real.intervals[c] for i in range(lo, hi + 1)}
            assert got == {rank[b] for b in lm.labels[:, c].nonzero()[0]}
        assert real.role_products == 1
    with pytest.raises(ValueError):
        interval_realisation(lm, [0, 1, 4])


def test_vectorised_checker_matches_reference_on_valid_and_broken_models():
    from scipy import sparse

    from mowl.geometric.least_model import _check_model_reference

    rng = np.random.default_rng(5)
    n_broken = 0
    for _ in range(50):
        tables = _random_theory(rng)
        lm = least_model(9, nb_rels=2, nb_individuals=3, bot_id=8, **tables)
        assert check_model(lm, **tables) == _check_model_reference(lm, **tables)
        clean = check_model(lm, **tables)
        # the pruned view is a model iff the KB is consistent; otherwise only
        # assertions about dead individuals fail
        bad = {k for k, v in clean.items() if v}
        assert bad <= ({"class_assertion", "object_property_assertion"} if not lm.consistent else set())
        # break the model: drop random labels and edges
        coo = lm.labels.tocoo()
        keep = rng.random(coo.nnz) > 0.3
        lm.labels = sparse.csr_matrix(
            (np.ones(keep.sum(), dtype=bool), (coo.row[keep], coo.col[keep])),
            shape=lm.labels.shape,
        )
        lm.edges = lm.edges[rng.random(len(lm.edges)) > 0.3]
        got = check_model(lm, **tables)
        assert got == _check_model_reference(lm, **tables)
        n_broken += any(v > 0 for v in got.values())
    assert n_broken > 25


def test_subsumption_matrix_chunked_and_empty_columns():
    rng = np.random.default_rng(11)
    for _ in range(10):
        tables = _random_theory(rng)
        lm = least_model(9, nb_rels=2, nb_individuals=3, bot_id=8, **tables)
        dense = lm.labels.toarray()
        brute = np.array(
            [[not (dense[:, i] & ~dense[:, j]).any() for j in range(9)] for i in range(9)]
        )
        assert np.array_equal(lm.subsumption_matrix(chunk=2).toarray(), brute)
        nonempty = dense.any(axis=0)
        lazy = lm.subsumption_matrix(empty_subsumed_by_all=False).toarray()
        assert np.array_equal(lazy, brute & nonempty[:, None])
        assert not dense[:, 8].any()


def test_lp_callable_weights_are_deterministic():
    tables = dict(gci0=[[0, 1]], gci1=[[0, 1, 2]])
    w = lambda rng: rng.uniform(0.5, 2.0, size=3 * 3 + 3)  # noqa: E731
    a = least_model(3, method="lp", objective_weights=w, **tables)
    b = least_model(3, method="lp", objective_weights=w, **tables)
    assert a.lp_objective == b.lp_objective
    assert np.array_equal(a.labels.toarray(), least_model(3, **tables).labels.toarray())


def test_death_propagates_over_256_dead_successors():
    # Regression: counting dead successors in int8 wrapped to 0 at multiples of 256,
    # so a class with exactly 256 unsatisfiable fillers stayed alive.
    n = 256
    args = dict(
        gci2=[[0, 0, 1 + i] for i in range(n)],
        gci0_bot=[[1 + i, n + 1] for i in range(n)],
    )
    lm = least_model(n + 2, nb_rels=1, bot_id=n + 1, **args)
    assert lm.dead[lm.element_map[0]]
    assert all(v == 0 for v in check_model(lm, **args).values())
