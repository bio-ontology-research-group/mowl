"""The direct temporal al-cone construction (Leemhuis, FCR 2024, Theorem 2), kept for
comparison (see :mod:`mowl.baselines`).

In :mod:`mowl.baselines.geometric.core.temporal_cones` the construction is the combination of the source
``lasso model`` and the ``half-axes`` reading with one cell per distinct trace.
"""
from typing import Dict, FrozenSet, List, Tuple

from mowl.baselines.geometric.core import temporal_cones
from mowl.baselines.geometric.core.cones import ALCone, halfaxis_vector
from mowl.baselines.geometric.core.temporal_cones import LassoModel

__all__ = ["TemporalALConeInterpretation"]


class TemporalALConeInterpretation(temporal_cones.TemporalALConeInterpretation):
    """A temporal al-cone interpretation with the direct construction."""

    @classmethod
    def from_lasso_model(cls, model: LassoModel) -> "TemporalALConeInterpretation":
        """The construction of Theorem 2 of the paper (see the module docstring): every
        distinct trace becomes a half-axis, the atom :math:`A` at time :math:`i` is the
        al-cone of the half-axes of the traces with :math:`A` at :math:`i`, and an
        individual is the ray of its trace's half-axis."""
        distinct: List[Tuple[FrozenSet[str], ...]] = []
        index: Dict[Tuple[FrozenSet[str], ...], int] = {}
        for tr in model.traces:
            key = tuple(tr)
            if key not in index:
                index[key] = len(distinct)
                distinct.append(key)
        # An odd number of traces leaves one half-axis without a trace; it gets a copy of
        # the trace of half-axis 0, so the interpretation has no element of its own.
        axes = list(distinct) + ([distinct[0]] if len(distinct) % 2 == 1 else [])
        dim = max(1, len(axes) // 2)
        names = sorted(set().union(*(s for key in distinct for s in key)))
        atoms = {name: [ALCone.from_halfaxes(
            dim, [h for h, key in enumerate(axes) if name in key[t]])
            for t in range(model.length)] for name in names}
        inds = {a: halfaxis_vector(dim, index[tuple(model.traces[d])])
                for a, d in model.individuals.items()}
        out = cls(dim, model.loop_start, atoms, inds, length=model.length)
        out.traces = distinct
        return out
