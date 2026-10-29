"""Shared test helpers for mowl.geometric test suite."""
from __future__ import annotations

import os
import random
from typing import List, Set, Tuple

import pytest

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
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

NS = "http://mowl.borg/geo#"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture_path(name: str) -> str:
    """Return absolute path to a fixture OWL file in ``tests/geometric/fixtures``."""
    return os.path.join(FIXTURES_DIR, name)


@pytest.fixture
def t3_dataset() -> PathDataset:
    """Return a :class:`PathDataset` for ``t3.owl``."""
    return PathDataset(fixture_path("t3.owl"))


@pytest.fixture
def helly_ok_dataset() -> PathDataset:
    """Return a :class:`PathDataset` for ``helly_ok.owl``."""
    return PathDataset(fixture_path("helly_ok.owl"))


@pytest.fixture
def roles_dataset() -> PathDataset:
    """Return a :class:`PathDataset` for ``roles.owl``."""
    return PathDataset(fixture_path("roles.owl"))


@pytest.fixture
def top_lhs_dataset() -> PathDataset:
    """Return a :class:`PathDataset` for ``top_lhs.owl``."""
    return PathDataset(fixture_path("top_lhs.owl"))


@pytest.fixture
def t3_cm(t3_dataset: PathDataset) -> CanonicalModel:
    """Return the :class:`CanonicalModel` for ``t3.owl``."""
    return CanonicalModel.from_dataset(t3_dataset)


@pytest.fixture
def helly_ok_cm(helly_ok_dataset: PathDataset) -> CanonicalModel:
    """Return the :class:`CanonicalModel` for ``helly_ok.owl``."""
    return CanonicalModel.from_dataset(helly_ok_dataset)


@pytest.fixture
def roles_cm(roles_dataset: PathDataset) -> CanonicalModel:
    """Return the :class:`CanonicalModel` for ``roles.owl``."""
    return CanonicalModel.from_dataset(roles_dataset)


@pytest.fixture
def top_lhs_cm(top_lhs_dataset: PathDataset) -> CanonicalModel:
    """Return the :class:`CanonicalModel` for ``top_lhs.owl``."""
    return CanonicalModel.from_dataset(top_lhs_dataset)


def random_el_tbox(
    rng: random.Random, allow_roles: bool = True
) -> NormalisedOntology:
    """Generate a random normalised EL TBox with <= 10 axioms, 6 concept names, and <= 2 roles.

    - Concept names: ``N0..N5`` in ``http://mowl.borg/geo#``.
    - Role names: ``r0, r1`` when ``allow_roles=True``, else empty.
    - With probability 0.15 per TBox, adds an axiom with ``TOP`` on the left
      (:class:`NF1` or, when ``allow_roles=True``, :class:`NF3`).
    - When ``allow_roles=True``, each :class:`NF4` axiom draws its ``(role, filler)``
      from pairs already used by an :class:`NF3` axiom in the same TBox (or a
      super-role of one via a generated :class:`NF6`), with right-hand side
      ``BOTTOM`` with probability 0.2 and a concept name otherwise.
    """
    names = [f"{NS}N{i}" for i in range(6)]
    roles = [f"{NS}r0", f"{NS}r1"] if allow_roles else []
    n_axioms = rng.randint(8, 10)
    axioms = []
    nf3_pairs: List[Tuple[str, str]] = []
    nf6_edges: List[Tuple[str, str]] = []

    if rng.random() < 0.15:
        if allow_roles and rng.random() < 0.5:
            role = rng.choice(roles)
            filler = rng.choice(names)
            axioms.append(NF3(TOP, role, filler))
            nf3_pairs.append((role, filler))
        else:
            axioms.append(NF1(TOP, rng.choice(names)))

    while len(axioms) < n_axioms:
        if not allow_roles:
            kind = rng.choice(("NF1", "NF2"))
        elif len(nf3_pairs) < 2:
            kind = rng.choices(
                ("NF1", "NF2", "NF3", "NF6"), weights=(1, 1, 7, 1)
            )[0]
        else:
            kind = rng.choices(
                ("NF1", "NF2", "NF3", "NF4", "NF6"), weights=(1, 1, 3, 6, 1)
            )[0]

        if kind == "NF1":
            sub = rng.choice(names)
            sup = (
                BOTTOM
                if rng.random() < 0.05
                else rng.choice([c for c in names if c != sub])
            )
            axioms.append(NF1(sub, sup))
        elif kind == "NF2":
            sub1, sub2 = rng.sample(names, 2)
            sup = (
                BOTTOM
                if rng.random() < 0.05
                else rng.choice([c for c in names if c not in (sub1, sub2)])
            )
            axioms.append(NF2(sub1, sub2, sup))
        elif kind == "NF3":
            sub, filler = rng.sample(names, 2)
            role = rng.choice(roles)
            axioms.append(NF3(sub, role, filler))
            if (role, filler) not in nf3_pairs:
                nf3_pairs.append((role, filler))
        elif kind == "NF4":
            pool = list(nf3_pairs)
            for r_sub, r_sup in nf6_edges:
                for r, f in nf3_pairs:
                    if r == r_sub and (r_sup, f) not in pool:
                        pool.append((r_sup, f))
            role, filler = rng.choice(pool)
            sup = (
                BOTTOM
                if rng.random() < 0.2
                else rng.choice([c for c in names if c != filler])
            )
            axioms.append(NF4(role, filler, sup))
        else:  # NF6
            r_sub, r_sup = rng.sample(roles, 2)
            axioms.append(NF6(r_sub, r_sup))
            if (r_sub, r_sup) not in nf6_edges:
                nf6_edges.append((r_sub, r_sup))

    return NormalisedOntology(
        axioms=axioms,
        concept_names=set(names) | {TOP, BOTTOM},
        role_names=set(roles),
        dropped_axioms=0,
    )


def assert_canonical_model_satisfies_axioms(cm: CanonicalModel) -> None:
    """Assert that ``cm`` set-theoretically satisfies every normalised axiom in ``cm.axioms``.

    Checks:
    - :class:`NF1` (``A <= B``): ``ext(A) <= ext(B)``.
    - :class:`NF2` (``A1 and A2 <= B``): ``ext(A1) & ext(A2) <= ext(B)``.
    - :class:`NF3` (``A <= exists r.B``): for every ``x in ext(A)``, there is
      some ``y in ext(B)`` with ``(x, y) in role(r)``.
    - :class:`NF4` (``exists r.A <= B``): for every ``(x, y) in role(r)``, if
      ``y in ext(A)`` then ``x in ext(B)``.
    - :class:`NF6` (``r <= s``): ``role(r) <= role(s)``.
    - :class:`NF7` (``r1 o r2 <= s``): relational composition
      ``{(x, z) : (x, y) in role(r1) and (y, z) in role(r2)} <= role(s)``.
    """
    for ax in cm.axioms:
        if isinstance(ax, NF1):
            ext_sub = cm.extension(ax.sub)
            ext_sup = cm.extension(ax.sup)
            assert ext_sub.issubset(ext_sup), (
                f"CanonicalModel violated {ax}: {ext_sub} not <= {ext_sup}"
            )
        elif isinstance(ax, NF2):
            ext_1 = cm.extension(ax.sub1)
            ext_2 = cm.extension(ax.sub2)
            ext_sup = cm.extension(ax.sup)
            inter = ext_1 & ext_2
            assert inter.issubset(ext_sup), (
                f"CanonicalModel violated {ax}: {inter} not <= {ext_sup}"
            )
        elif isinstance(ax, NF3):
            ext_sub = cm.extension(ax.sub)
            ext_fill = cm.extension(ax.filler)
            r_pairs = cm.role(ax.role)
            for x in ext_sub:
                has_succ = any((x, y) in r_pairs for y in ext_fill)
                assert has_succ, (
                    f"CanonicalModel violated {ax}: element {x} ({cm.domain[x]}) "
                    f"has no {ax.role}-successor in {ext_fill}"
                )
        elif isinstance(ax, NF4):
            ext_fill = cm.extension(ax.filler)
            ext_sup = cm.extension(ax.sup)
            r_pairs = cm.role(ax.role)
            preimage = {x for (x, y) in r_pairs if y in ext_fill}
            assert preimage.issubset(ext_sup), (
                f"CanonicalModel violated {ax}: preimage {preimage} not <= {ext_sup}"
            )
        elif isinstance(ax, NF6):
            r_sub = cm.role(ax.sub)
            r_sup = cm.role(ax.sup)
            assert r_sub.issubset(r_sup), (
                f"CanonicalModel violated {ax}: {r_sub} not <= {r_sup}"
            )
        elif isinstance(ax, NF7):
            r1 = cm.role(ax.r1)
            r2 = cm.role(ax.r2)
            r_sup = cm.role(ax.sup)
            composed: Set[Tuple[int, int]] = {
                (x, z)
                for (x, y1) in r1
                for (y2, z) in r2
                if y1 == y2
            }
            assert composed.issubset(r_sup), (
                f"CanonicalModel violated {ax}: composition {composed} not <= {r_sup}"
            )
        else:  # pragma: no cover
            raise AssertionError(f"Unexpected axiom type: {ax!r}")
