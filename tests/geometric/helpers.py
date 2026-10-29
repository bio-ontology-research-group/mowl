"""Shared test helpers for mowl.geometric test suite."""
from __future__ import annotations

from typing import Set, Tuple

from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import NF1, NF2, NF3, NF4, NF6, NF7


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
