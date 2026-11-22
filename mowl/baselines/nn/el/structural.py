"""The data of the structural objective of MultiBoxEL and PrismEL: definition tables, existence
targets and witness targets, and the regions with signed half-widths that the grounded model
reads.

1. **Regions with absent components (:class:`Regions`):** Each class is materialised as
   :math:`K` component slots with centre :math:`c \\in \\mathbb{R}^d`, *signed* half-width
   :math:`w \\in \\mathbb{R}^d` (negative in a coordinate iff the slot is empty along that
   coordinate), and a boolean ``active`` mask.
2. **Definition table (:class:`DefinitionTable`, :func:`slot_components`), existence targets
   (:class:`ExistenceTargets`), and witness targets (:class:`WitnessTargets`):** Compile
   acyclic conjunction definitions :math:`A \\equiv X \\sqcap Y` within the slot/depth budget,
   map materialised definition slots back to primitive component indices, identify
   satisfiable classes and conjunction pairs requiring positive measure, and build
   positive/negative class-membership targets for per-class witness points.

These are data (which classes are defined, which must be non-empty), not losses. The losses of
the structural reading are computed from the parts
(:class:`~mowl.representation.library.structural_reading.StructuralReading`, its penalties,
and the slot algebra of :mod:`mowl.representation.library.structural_regions`); the direct
implementation keeps its own copy in :mod:`mowl.baselines`.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch as th

if TYPE_CHECKING:
    from mowl.baselines.geometric.core.completion import CanonicalModel

__all__ = [
    "Regions",
    "inside_measure",
    "DefinitionTable",
    "slot_components",
    "WitnessTargets",
    "ExistenceTargets",
    "build_structural_tables",
]


class Regions(NamedTuple):
    """Materialised multi-component box regions with signed half-widths and active mask.

    Attributes
    ----------
    cen : torch.Tensor
        Component centres of shape ``(B, K, d)``.
    w : torch.Tensor
        Signed half-widths of shape ``(B, K, d)``; negative in coordinate ``m``
        if and only if the box is empty along coordinate ``m``.
    active : torch.Tensor
        Boolean mask of shape ``(B, K)`` indicating which slots are active.
    """

    cen: th.Tensor
    w: th.Tensor
    active: th.Tensor


def inside_measure(points: th.Tensor, regions: Regions) -> th.Tensor:
    """The signed depth of ``points`` ``(W, d)`` in ``regions`` ``(W, K, d)``: the depth of
    :func:`~mowl.representation.library.structural_regions.slot_depth` (positive iff the
    point lies in the interior of an active slot; ``-1e6`` without gradient for a row without
    an active slot), for :class:`Regions`.

    Raises :class:`ValueError` if the shapes of ``points`` and ``regions`` do not fit.
    """
    shapes_fit = (points.ndim == 2 and regions.cen.ndim == 3
                  and tuple(regions.w.shape) == tuple(regions.cen.shape)
                  and tuple(regions.active.shape) == tuple(regions.cen.shape[:2])
                  and points.shape[0] == regions.cen.shape[0]
                  and points.shape[1] == regions.cen.shape[2])
    if not shapes_fit:
        raise ValueError(f"Incompatible shapes between points {tuple(points.shape)} and "
                         f"regions (cen={tuple(regions.cen.shape)}, "
                         f"w={tuple(regions.w.shape)}, active={tuple(regions.active.shape)}).")
    if points.shape[0] == 0:
        return points.new_empty(0)
    per_slot = (regions.w - (points[:, None, :] - regions.cen).abs()).amin(dim=-1)
    some = regions.active.any(dim=1)
    best = th.where(regions.active, per_slot, th.full_like(per_slot, -float("inf")))
    best = best.max(dim=1).values
    return th.where(some, best, th.full_like(best, -1e6))


@dataclass
class DefinitionTable:
    """Compiled structural conjunction definitions :math:`A := X \\sqcap Y` extracted from normal forms.

    Guarantees:

    - The accepted definition graph is a DAG: ``level[a] == 1 + max(level[def_x[a]], level[def_y[a]])``
      for every defined class ``a``, and ``level[p] == 0`` for every primitive class ``p``.
    - For every defined class ``a``, ``comps[a] == comps[def_x[a]] * comps[def_y[a]] <= max_components``
      and ``level[a] <= max_level`` (when a bound was passed to :meth:`from_normal_forms`).
    - :meth:`masked_gci0` marks the ``gci0`` rows ``(A, X)`` and ``(A, Y)`` of
      accepted definitions :math:`A \\to (X, Y)` (when :math:`(A, \\text{Aux})` is
      only entailed through a role auxiliary, no ``gci0`` row exists for it and
      therefore only ``(A, X)`` and the ``gci1`` rows are masked; the
      ``gci2``/``gci3`` rows keep training :math:`\\text{Aux}`), and
      :meth:`masked_gci1` marks ``(X, Y, A)`` and ``(Y, X, A)``; ``gci2`` and
      ``gci3`` rows are never masked (they continue to train the role
      auxiliary's region).

    Non-guarantees:

    - Only compiles binary conjunction definitions :math:`A \\equiv X \\sqcap Y`
      where each inclusion :math:`A \\sqsubseteq D` (:math:`D \\in \\{X, Y\\}`) is
      witnessed either directly in ``gci0`` or via a single role-auxiliary step
      :math:`A \\sqsubseteq \\exists r.C \\sqsubseteq \\text{Aux}` in ``gci2`` + ``gci3``;
      definitions whose normalisation introduces nested auxiliaries on both
      sides (e.g. :math:`A \\equiv \\text{Pizza} \\sqcap \\exists\\text{hasTopping}.(\\text{PizzaTopping} \\sqcap \\exists\\text{hasSpiciness}.\\text{Hot})`
      yields :math:`A \\sqsubseteq \\exists r.\\text{Aux}_1` and
      :math:`\\exists r.\\text{Aux}_2 \\sqsubseteq \\text{Aux}_3` with
      :math:`\\text{Aux}_1 \\neq \\text{Aux}_2`) are not detected, and pure role
      equivalences (:math:`A \\equiv \\exists r.B`) remain primitive.

    Attributes
    ----------
    def_mask : torch.Tensor
        Boolean tensor of shape ``(n_classes,)`` indicating accepted defined classes.
    def_x : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with first definer index (``-1`` for primitive).
    def_y : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with second definer index (``-1`` for primitive).
    level : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with definition depth (``0`` for primitive).
    comps : torch.Tensor
        Long tensor of shape ``(n_classes,)`` with required slot count (``k`` for primitive).
    n_defined : int
        Number of accepted defined classes.
    max_level : int
        Maximum definition level among all classes (``0`` if none defined).
    """

    def_mask: th.Tensor
    def_x: th.Tensor
    def_y: th.Tensor
    level: th.Tensor
    comps: th.Tensor
    n_defined: int
    max_level: int
    _masked_gci0_keys: th.Tensor
    _masked_gci1_keys: th.Tensor

    @property
    def n_classes(self) -> int:
        """Total number of classes in the table."""
        return int(self.def_mask.shape[0])

    @classmethod
    def from_normal_forms(
        cls,
        gci0: th.Tensor,
        gci1: th.Tensor,
        n_classes: int,
        k: int,
        max_components: int,
        top_idx: Optional[int],
        bottom_idx: Optional[int],
        enabled: bool = True,
        max_level: Optional[int] = None,
        gci2: Optional[th.Tensor] = None,
        gci3: Optional[th.Tensor] = None,
    ) -> DefinitionTable:
        """Build a :class:`DefinitionTable` from ``gci0`` and ``gci1`` (and optional ``gci2``/``gci3``) normal-form index tensors.

        Guarantees:

        - When ``enabled=False``, every class is primitive (``n_defined == 0``,
          ``max_level == 0``, ``comps == k``) and no rows are masked.
        - Candidate definers for ``A`` are the lexicographically smallest
          ``(min(X, Y), max(X, Y))`` with ``X != Y``, ``A not in {X, Y}``,
          none of ``A, X, Y`` in ``{top_idx, bottom_idx}``, ``(X, Y, A)`` in
          ``gci1``, and each required inclusion ``(A, D)`` for ``D in {X, Y}``
          present either directly in ``gci0`` or through a role auxiliary via a
          ``gci2`` row ``(A, r, C)`` and a ``gci3`` row ``(r, C, D)`` (same ``r``
          and ``C``, i.e. :math:`A \\sqsubseteq \\exists r.C \\sqsubseteq D` is
          entailed).
        - Candidates are accepted iff acyclic, ``comps(X) * comps(Y) <= max_components``,
          and ``level(A) <= max_level`` (when ``max_level is not None``), breaking
          dependency cycles deterministically in increasing class index.

        Non-guarantees:

        - Definitions whose normalisation introduces nested auxiliaries on both
          sides (e.g.
          :math:`A \\equiv \\text{Pizza} \\sqcap \\exists\\text{hasTopping}.(\\text{PizzaTopping} \\sqcap \\exists\\text{hasSpiciness}.\\text{Hot})`
          yields :math:`A \\sqsubseteq \\exists r.\\text{Aux}_1` and
          :math:`\\exists r.\\text{Aux}_2 \\sqsubseteq \\text{Aux}_3` with
          :math:`\\text{Aux}_1 \\neq \\text{Aux}_2`) are not detected; only a
          single role-auxiliary step
          :math:`A \\sqsubseteq \\exists r.C \\sqsubseteq \\text{Aux}` is.
        - When the ready queue empties while undecided candidates remain, the
          cycle break rejects the lowest-index ``UNDECIDED`` candidate; that
          candidate may sit downstream of a cycle rather than on the cycle
          itself (deterministic, possibly lossy).
        - Rejected candidates remain primitive and their ``gci0``/``gci1`` rows
          are not masked; ``gci2`` and ``gci3`` rows are never masked.

        Parameters
        ----------
        gci0 : torch.Tensor
            Long tensor of shape ``(N0, 2)`` with rows ``(C, D)``.
        gci1 : torch.Tensor
            Long tensor of shape ``(N1, 3)`` with rows ``(C, D, E)``.
        n_classes : int
            Total number of classes.
        k : int
            Number of primitive components per class.
        max_components : int
            Maximum allowed component slots :math:`K` for a defined class.
        top_idx : Optional[int]
            Class index of ``owl:Thing`` if present, else ``None``.
        bottom_idx : Optional[int]
            Class index of ``owl:Nothing`` if present, else ``None``.
        enabled : bool, default True
            If ``False``, disables structural definitions (all classes primitive).
        max_level : Optional[int], default None
            Optional maximum definition level cap (``None`` for unbounded).
        gci2 : Optional[torch.Tensor], default None
            Optional long tensor of shape ``(N2, 3)`` with rows ``(C, r, D)`` for
            :math:`C \\sqsubseteq \\exists r.D` (matching :class:`~mowl.datasets.el.GCI2Dataset`).
        gci3 : Optional[torch.Tensor], default None
            Optional long tensor of shape ``(N3, 3)`` with rows ``(r, C, D)`` for
            :math:`\\exists r.C \\sqsubseteq D` (matching :class:`~mowl.datasets.el.GCI3Dataset`).

        Returns
        -------
        DefinitionTable
            Compiled definition table.
        """
        def_mask = th.zeros(n_classes, dtype=th.bool)
        def_x = th.full((n_classes,), -1, dtype=th.long)
        def_y = th.full((n_classes,), -1, dtype=th.long)
        level = th.zeros(n_classes, dtype=th.long)
        comps = th.full((n_classes,), k, dtype=th.long)

        if not enabled or n_classes == 0 or (max_level is not None and max_level < 1):
            return cls(
                def_mask=def_mask,
                def_x=def_x,
                def_y=def_y,
                level=level,
                comps=comps,
                n_defined=0,
                max_level=0,
                _masked_gci0_keys=th.zeros(0, dtype=th.long),
                _masked_gci1_keys=th.zeros(0, dtype=th.long),
            )

        excluded: Set[int] = {
            int(idx) for idx in (top_idx, bottom_idx) if idx is not None
        }

        gci0_set: Set[Tuple[int, int]] = set()
        if gci0 is not None and gci0.numel() > 0:
            for row in gci0.detach().cpu().reshape(-1, 2).tolist():
                gci0_set.add((int(row[0]), int(row[1])))

        if (
            gci2 is not None
            and gci2.numel() > 0
            and gci3 is not None
            and gci3.numel() > 0
        ):
            rc_to_a: Dict[Tuple[int, int], Set[int]] = {}
            for row in gci2.detach().cpu().reshape(-1, 3).tolist():
                a_i, r_i, c_i = int(row[0]), int(row[1]), int(row[2])
                rc_to_a.setdefault((r_i, c_i), set()).add(a_i)
            for row in gci3.detach().cpu().reshape(-1, 3).tolist():
                r_i, c_i, y_i = int(row[0]), int(row[1]), int(row[2])
                for a_i in rc_to_a.get((r_i, c_i), ()):
                    gci0_set.add((a_i, y_i))

        candidates: Dict[int, Tuple[int, int]] = {}
        if gci1 is not None and gci1.numel() > 0:
            for row in gci1.detach().cpu().reshape(-1, 3).tolist():
                x_raw, y_raw, a_raw = int(row[0]), int(row[1]), int(row[2])
                if x_raw == y_raw or a_raw == x_raw or a_raw == y_raw:
                    continue
                if a_raw in excluded or x_raw in excluded or y_raw in excluded:
                    continue
                if not (
                    0 <= a_raw < n_classes
                    and 0 <= x_raw < n_classes
                    and 0 <= y_raw < n_classes
                ):
                    continue
                if (a_raw, x_raw) in gci0_set and (a_raw, y_raw) in gci0_set:
                    pair = (min(x_raw, y_raw), max(x_raw, y_raw))
                    if a_raw not in candidates or pair < candidates[a_raw]:
                        candidates[a_raw] = pair

        accepted: Dict[int, Tuple[int, int]] = {}
        lvl_arr: List[int] = [0] * n_classes
        cmp_arr: List[int] = [k] * n_classes

        dependants: Dict[int, List[int]] = {u: [] for u in candidates}
        in_deg: Dict[int, int] = {}
        ready_heap: List[int] = []
        for u, (x, y) in candidates.items():
            deg = 0
            if x in candidates:
                dependants[x].append(u)
                deg += 1
            if y in candidates:
                dependants[y].append(u)
                deg += 1
            in_deg[u] = deg
            if deg == 0:
                heapq.heappush(ready_heap, u)

        undecided: Set[int] = set(candidates.keys())
        sorted_cands = sorted(candidates.keys())
        cand_ptr = 0

        def _release_dependants(node: int) -> None:
            for dep in dependants[node]:
                if dep in undecided:
                    in_deg[dep] -= 1
                    if in_deg[dep] == 0:
                        heapq.heappush(ready_heap, dep)

        while undecided:
            while ready_heap:
                u = heapq.heappop(ready_heap)
                if u not in undecided:
                    continue
                undecided.remove(u)
                x, y = candidates[u]
                lv = 1 + max(lvl_arr[x], lvl_arr[y])
                cp = cmp_arr[x] * cmp_arr[y]
                if cp <= max_components and (max_level is None or lv <= max_level):
                    accepted[u] = (x, y)
                    lvl_arr[u] = lv
                    cmp_arr[u] = cp
                _release_dependants(u)

            if undecided:
                while sorted_cands[cand_ptr] not in undecided:
                    cand_ptr += 1
                victim = sorted_cands[cand_ptr]
                cand_ptr += 1
                undecided.remove(victim)
                _release_dependants(victim)

        gci0_keys: List[int] = []
        gci1_keys: List[int] = []
        for a, (x, y) in sorted(accepted.items()):
            def_mask[a] = True
            def_x[a] = x
            def_y[a] = y
            level[a] = lvl_arr[a]
            comps[a] = cmp_arr[a]
            gci0_keys.append(a * n_classes + x)
            gci0_keys.append(a * n_classes + y)
            gci1_keys.append((x * n_classes + y) * n_classes + a)
            gci1_keys.append((y * n_classes + x) * n_classes + a)

        n_defined = len(accepted)
        max_lvl = int(level.max().item()) if n_defined > 0 else 0
        return cls(
            def_mask=def_mask,
            def_x=def_x,
            def_y=def_y,
            level=level,
            comps=comps,
            n_defined=n_defined,
            max_level=max_lvl,
            _masked_gci0_keys=th.tensor(gci0_keys, dtype=th.long),
            _masked_gci1_keys=th.tensor(gci1_keys, dtype=th.long),
        )

    def masked_gci0(self, rows: th.Tensor) -> th.Tensor:
        """Return a boolean mask over ``gci0`` rows ``(C, D)`` covered by accepted definitions.

        Guarantees:

        - Returns a 1-D ``torch.bool`` tensor of length ``rows.shape[0]`` on
          ``rows.device``.
        - Marks entry ``b`` as ``True`` iff ``rows[b]`` is ``(A, X)`` or
          ``(A, Y)`` for an accepted definition :math:`A \\to (X, Y)`. When
          :math:`(A, \\text{Aux})` is only entailed through a role auxiliary, no
          ``gci0`` row exists for it and therefore only ``(A, X)`` and the
          ``gci1`` rows are masked; the ``gci2``/``gci3`` rows keep training
          :math:`\\text{Aux}`.

        Non-guarantees:

        - Assumes row entries are valid class indices in ``0 .. n_classes - 1``.

        Parameters
        ----------
        rows : torch.Tensor
            Integer tensor of shape ``(B, 2)``.

        Returns
        -------
        torch.Tensor
            Boolean mask tensor of shape ``(B,)``.
        """
        bsz = rows.shape[0]
        if bsz == 0 or self._masked_gci0_keys.numel() == 0:
            return th.zeros(bsz, dtype=th.bool, device=rows.device)
        n = self.n_classes
        keys = rows[:, 0].to(th.long) * n + rows[:, 1].to(th.long)
        if self._masked_gci0_keys.device != rows.device:
            self._masked_gci0_keys = self._masked_gci0_keys.to(device=rows.device)
        return th.isin(keys, self._masked_gci0_keys)

    def masked_gci1(self, rows: th.Tensor) -> th.Tensor:
        """Return a boolean mask over ``gci1`` rows ``(C, D, E)`` covered by accepted definitions.

        Guarantees:

        - Returns a 1-D ``torch.bool`` tensor of length ``rows.shape[0]`` on
          ``rows.device``.
        - Marks entry ``b`` as ``True`` iff ``rows[b]`` is ``(X, Y, A)`` or
          ``(Y, X, A)`` for an accepted definition :math:`A \\to (X, Y)`.

        Non-guarantees:

        - Assumes row entries are valid class indices in ``0 .. n_classes - 1``.

        Parameters
        ----------
        rows : torch.Tensor
            Integer tensor of shape ``(B, 3)``.

        Returns
        -------
        torch.Tensor
            Boolean mask tensor of shape ``(B,)``.
        """
        bsz = rows.shape[0]
        if bsz == 0 or self._masked_gci1_keys.numel() == 0:
            return th.zeros(bsz, dtype=th.bool, device=rows.device)
        n = self.n_classes
        r0 = rows[:, 0].to(th.long)
        r1 = rows[:, 1].to(th.long)
        r2 = rows[:, 2].to(th.long)
        keys = (r0 * n + r1) * n + r2
        if self._masked_gci1_keys.device != rows.device:
            self._masked_gci1_keys = self._masked_gci1_keys.to(device=rows.device)
        return th.isin(keys, self._masked_gci1_keys)


def slot_components(
    table: DefinitionTable,
    cls: int,
    slot: int,
    k: Optional[int] = None,
) -> Dict[int, int]:
    """Map primitive class index to component index for a materialised slot of ``cls``.

    Follows the exact recursive slot ordering produced by :func:`pair_slots` in
    :meth:`~mowl.nn.MultiBoxELModule.class_regions`:

    * If ``cls`` is primitive (``not table.def_mask[cls]``), active slot ``s``
      (``0 <= s < k``) maps to ``{cls: s}``.
    * If ``cls`` is an accepted defined class :math:`A \\to (X, Y)` with
      :math:`c_Y` active slots on :math:`Y`, active slot ``s`` corresponds to
      the lexicographic pair ``(s // c_Y, s % c_Y)`` of active slots of :math:`X`
      and :math:`Y`, and recurses to the primitive leaves reachable through
      ``def_x`` and ``def_y``.

    Guarantees:

    - Returns a ``Dict[int, int]`` mapping each primitive leaf class index in
      the definition tree of ``cls`` to its component index in ``0 .. k - 1``.
    - For any tree-structured definition (distinct primitive leaves), the box of
      slot ``slot`` in :meth:`~mowl.nn.MultiBoxELModule.class_regions` equals
      the coordinatewise intersection of the mapped primitive component boxes
      ``(cen[p, c], |offset[p, c]|)`` for ``(p, c)`` in the returned dictionary.
    - Raises :class:`ValueError` if ``cls``, ``slot``, or ``k`` is out of range
      or not an integer.

    Non-guarantees:

    - If a defined class has a diamond DAG where the same primitive leaf is
      reachable along both ``def_x`` and ``def_y`` with different component
      indices in ``slot``, the right branch (``def_y``) overwrites the left
      branch (``def_x``) in the returned dictionary.

    Parameters
    ----------
    table : DefinitionTable
        Compiled conjunction definition table (or a module exposing
        ``definition_table`` and ``n_components``).
    cls : int
        Class index in ``0 .. table.n_classes - 1``.
    slot : int
        Active materialised slot index of ``cls``.
    k : int, optional
        Number of primitive components per class :math:`k \\ge 1`. When ``None``,
        inferred from ``table.comps`` on primitive classes (or ``table.n_components``
        if a module is passed).

    Returns
    -------
    Dict[int, int]
        Mapping ``{primitive_class_idx: component_idx}``.
    """
    if not isinstance(table, DefinitionTable):
        if hasattr(table, "definition_table") and hasattr(table, "n_components"):
            mod = table
            if k is None:
                k = int(mod.n_components)
            if mod.definition_table is None:
                if isinstance(cls, bool) or not isinstance(cls, int):
                    raise ValueError(f"cls must be an int, got {cls!r}")
                if isinstance(slot, bool) or not isinstance(slot, int):
                    raise ValueError(f"slot must be an int, got {slot!r}")
                if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                    raise ValueError(f"k must be an int >= 1, got {k!r}")
                n_cls = int(mod.nb_ont_classes)
                if cls < 0 or cls >= n_cls:
                    raise ValueError(f"cls must be in 0 .. {n_cls - 1}, got {cls}")
                if slot < 0 or slot >= k:
                    raise ValueError(
                        f"slot must be in 0 .. {k - 1} for class {cls}, got {slot}"
                    )
                return {int(cls): int(slot)}
            table = mod.definition_table
        else:
            raise TypeError(
                f"table must be a DefinitionTable, got {type(table).__name__}"
            )

    if isinstance(cls, bool) or not isinstance(cls, int):
        raise ValueError(f"cls must be an int, got {cls!r}")
    if isinstance(slot, bool) or not isinstance(slot, int):
        raise ValueError(f"slot must be an int, got {slot!r}")

    if k is None:
        prim_mask = ~table.def_mask
        if not bool(prim_mask.any().item()):
            raise ValueError("k must be provided when table has no primitive classes.")
        k_val = int(table.comps[prim_mask][0].item())
    else:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be an int >= 1, got {k!r}")
        k_val = int(k)

    n_classes = table.n_classes
    if cls < 0 or cls >= n_classes:
        raise ValueError(f"cls must be in 0 .. {n_classes - 1}, got {cls}")

    def _num_slots(u: int) -> int:
        if not bool(table.def_mask[u].item()):
            return k_val
        x_u = int(table.def_x[u].item())
        y_u = int(table.def_y[u].item())
        return _num_slots(x_u) * _num_slots(y_u)

    total_slots = _num_slots(cls)
    if slot < 0 or slot >= total_slots:
        raise ValueError(
            f"slot must be in 0 .. {total_slots - 1} for class {cls}, got {slot}"
        )

    def _resolve(u: int, s: int) -> Dict[int, int]:
        if not bool(table.def_mask[u].item()):
            return {int(u): int(s)}
        x_u = int(table.def_x[u].item())
        y_u = int(table.def_y[u].item())
        cy = _num_slots(y_u)
        sx = s // cy
        sy = s % cy
        out = _resolve(x_u, sx)
        out.update(_resolve(y_u, sy))
        return out

    return _resolve(cls, slot)


@dataclass
class WitnessTargets:
    """Satisfiable witness classes and positive/negative class-membership targets for :func:`witness_loss`.

    Guarantees:

    - ``classes`` is a 1-D ``torch.long`` tensor of shape ``(W,)``, ``pos`` is a
      2-D ``torch.long`` tensor of shape ``(P, 2)``, and ``neg`` is a 2-D
      ``torch.long`` tensor of shape ``(N, 2)`` with rows ``(witness_row, class_idx)``
      where ``witness_row`` is in ``0 .. W - 1`` (aligned with ``classes``) and
      ``class_idx`` is a full class index.
    - ``top_idx`` and ``bottom_idx`` are excluded from ``classes`` and from the
      ``class_idx`` column of both ``pos`` and ``neg``.
    - Rows of ``pos`` and ``neg`` are deduplicated and disjoint (``pos ∩ neg = ∅``),
      and ``(r, classes[r])`` is always included in ``pos`` for every witness row ``r``.

    Non-guarantees:

    - Without a :class:`~mowl.baselines.geometric.core.completion.CanonicalModel`, positive and
      negative targets are approximated syntactically from ``gci0`` and ``gci1_bot``
      rows only (entailed subsumptions and entailed disjointness are not inferred).
    - Auxiliary classes minted by :class:`~mowl.ontology.normalize.ELNormalizer`
      (``http://mowl.borg/aux/...``) are unknown to a
      :class:`~mowl.baselines.geometric.core.completion.CanonicalModel` built with
      ``mowl.geometric``'s own normaliser, so their relations fall back to the
      asserted ``gci0`` and ``gci1_bot`` rows only.

    Attributes
    ----------
    classes : torch.Tensor
        1-D ``torch.long`` tensor of witness class indices, shape ``(W,)``.
    pos : torch.Tensor
        2-D ``torch.long`` tensor of positive pairs ``(witness_row, class_idx)``,
        shape ``(P, 2)``.
    neg : torch.Tensor
        2-D ``torch.long`` tensor of negative pairs ``(witness_row, class_idx)``,
        shape ``(N, 2)``.
    """

    classes: th.Tensor
    pos: th.Tensor
    neg: th.Tensor

    @classmethod
    def from_normal_forms(
        cls,
        classes: Union[th.Tensor, Sequence[int]],
        gci0: Optional[th.Tensor] = None,
        gci1_bot: Optional[th.Tensor] = None,
        top_idx: Optional[int] = None,
        bottom_idx: Optional[int] = None,
        index_to_iri: Optional[Union[Mapping[int, str], Mapping[str, int], Sequence[str]]] = None,
        cm: Optional[CanonicalModel] = None,
        *,
        n_classes: Optional[int] = None,
        class_index_dict: Optional[Mapping[str, int]] = None,
    ) -> WitnessTargets:
        """Build :class:`WitnessTargets` from satisfiable classes, normal forms, and an optional :class:`CanonicalModel`.

        Guarantees:

        - With ``cm`` provided, for each witness row ``r`` with class ``C = classes[r]``
          whose IRI is in ``cm.concept_names``:

          - ``pos`` includes ``(r, D)`` for ``D == C`` and for every target class ``D``
            whose IRI is in ``cm.concept_names`` with ``cm.entails(iri_c, iri_d)``;
          - ``neg`` includes ``(r, D)`` for every satisfiable target class ``D``
            whose IRI is in ``cm.concept_names`` with
            ``not cm.joint_satisfiable([iri_c, iri_d])`` (excluding any ``D`` in ``pos``);
          - whenever ``C`` or ``D`` has an IRI unknown to ``cm``, falls back to the
            asserted ``gci0``/``gci1_bot`` rule for that pair.

          Raises :class:`ValueError` if ``cm`` is provided without ``index_to_iri``
          (or ``class_index_dict``).
        - Without ``cm``, ``pos`` for witness ``C = classes[r]`` is
          :math:`\\{(r, C)\\} \\cup \\{(r, D) : (C, D) \\in \\text{gci0}\\}` and ``neg`` is
          :math:`\\{(r, D) : (C, D) \\text{ or } (D, C) \\in \\text{gci1\\_bot}\\} \\setminus \\text{pos}`.
        - Excludes ``top_idx`` and ``bottom_idx`` everywhere and deduplicates rows.

        Non-guarantees:

        - Without ``cm``, indirect subsumptions (:math:`C \\sqsubseteq X_1 \\sqsubseteq X_2`)
          and indirect disjointness (:math:`A \\sqsubseteq X_2,\\; X_2 \\sqcap X_3 \\sqsubseteq \\bot`)
          are not added unless asserted directly in ``gci0`` or ``gci1_bot``.
        - Auxiliary classes minted by :class:`~mowl.ontology.normalize.ELNormalizer`
          are unknown to ``cm`` and use the asserted ``gci0``/``gci1_bot`` rules only.

        Parameters
        ----------
        classes : Union[torch.Tensor, Sequence[int]]
            1-D sequence or ``torch.long`` tensor of satisfiable class indices.
        gci0 : Optional[torch.Tensor], default None
            Long tensor of shape ``(N0, 2)`` with rows ``(C, D)``.
        gci1_bot : Optional[torch.Tensor], default None
            Long tensor of shape ``(N2, 2)`` with rows ``(C, D)``.
        top_idx : Optional[int], default None
            Class index of ``owl:Thing`` if present, else ``None``.
        bottom_idx : Optional[int], default None
            Class index of ``owl:Nothing`` if present, else ``None``.
        index_to_iri : Optional[Union[Mapping[int, str], Mapping[str, int], Sequence[str]]], default None
            Mapping from class index to IRI string (or ``class_index_dict`` mapping
            IRI to index, or sequence of IRI strings indexed by class index).
            Required when ``cm`` is given.
        cm : Optional[CanonicalModel], default None
            Saturated canonical model for semantic entailment and joint-satisfiability
            queries.
        n_classes : Optional[int], default None
            Optional total number of classes used to bound class indices.
        class_index_dict : Optional[Mapping[str, int]], default None
            Optional mapping from IRI string to class index (alternative to
            ``index_to_iri``).

        Returns
        -------
        WitnessTargets
            Compiled witness target tensors ``(classes, pos, neg)``.
        """
        idx_to_iri: Optional[Dict[int, str]] = None
        if index_to_iri is not None:
            if isinstance(index_to_iri, Mapping):
                sample_key = next(iter(index_to_iri.keys()), None)
                if isinstance(sample_key, str):
                    idx_to_iri = {int(v): str(k) for k, v in index_to_iri.items()}
                else:
                    idx_to_iri = {int(k): str(v) for k, v in index_to_iri.items()}
            else:
                idx_to_iri = {i: str(v) for i, v in enumerate(index_to_iri)}
        elif class_index_dict is not None:
            idx_to_iri = {int(idx): str(iri) for iri, idx in class_index_dict.items()}

        if cm is not None and idx_to_iri is None:
            raise ValueError("index_to_iri must be provided when cm is not None.")

        excluded: Set[int] = {
            int(idx) for idx in (top_idx, bottom_idx) if idx is not None
        }

        def _valid_idx(idx: int) -> bool:
            if idx < 0 or idx in excluded:
                return False
            if n_classes is not None and idx >= n_classes:
                return False
            return True

        cm_concepts: Set[str] = set()
        if cm is not None:
            cn_attr = getattr(cm, "concept_names", set())
            cm_concepts = set(cn_attr() if callable(cn_attr) else cn_attr)

        classes_raw = (
            th.as_tensor(classes, dtype=th.long).detach().cpu().reshape(-1).tolist()
        )
        witness_classes: List[int] = []
        seen_w: Set[int] = set()
        for c_raw in classes_raw:
            c = int(c_raw)
            if not _valid_idx(c) or c in seen_w:
                continue
            if cm is not None:
                assert idx_to_iri is not None
                iri_c = idx_to_iri.get(c)
                if iri_c is not None and iri_c in cm_concepts and not cm.is_satisfiable(iri_c):
                    continue
            seen_w.add(c)
            witness_classes.append(c)

        gci0_supers: Dict[int, Set[int]] = {}
        all_candidates: Set[int] = set(witness_classes)
        if n_classes is not None:
            for d in range(n_classes):
                if _valid_idx(d):
                    all_candidates.add(d)
        if idx_to_iri is not None:
            for d in idx_to_iri.keys():
                if _valid_idx(d):
                    all_candidates.add(d)

        if gci0 is not None and gci0.numel() > 0:
            for row in gci0.detach().cpu().reshape(-1, 2).tolist():
                c, d = int(row[0]), int(row[1])
                if not (_valid_idx(c) and _valid_idx(d)):
                    continue
                gci0_supers.setdefault(c, set()).add(d)
                all_candidates.add(c)
                all_candidates.add(d)

        gci1_bot_partners: Dict[int, Set[int]] = {}
        if gci1_bot is not None and gci1_bot.numel() > 0:
            for row in gci1_bot.detach().cpu().reshape(-1, 2).tolist():
                c, d = int(row[0]), int(row[1])
                if not (_valid_idx(c) and _valid_idx(d)):
                    continue
                gci1_bot_partners.setdefault(c, set()).add(d)
                gci1_bot_partners.setdefault(d, set()).add(c)
                all_candidates.add(c)
                all_candidates.add(d)

        sorted_candidates = sorted(all_candidates)
        cm_sat: Dict[int, bool] = {}
        if cm is not None:
            assert idx_to_iri is not None
            for d in sorted_candidates:
                iri_d = idx_to_iri.get(d)
                if iri_d is not None and iri_d in cm_concepts:
                    cm_sat[d] = bool(cm.is_satisfiable(iri_d))

        pos_rows: List[Tuple[int, int]] = []
        neg_rows: List[Tuple[int, int]] = []

        for w_row, c in enumerate(witness_classes):
            pos_c: Set[int] = {c}
            neg_c: Set[int] = set()

            iri_c = idx_to_iri.get(c) if idx_to_iri is not None else None
            if cm is not None and iri_c is not None and iri_c in cm_concepts:
                assert idx_to_iri is not None
                for d in sorted_candidates:
                    iri_d = idx_to_iri.get(d)
                    if iri_d is not None and iri_d in cm_concepts:
                        if cm.entails(iri_c, iri_d):
                            pos_c.add(d)
                        elif cm_sat.get(d, False) and not cm.joint_satisfiable([iri_c, iri_d]):
                            neg_c.add(d)
                    else:
                        if d in gci0_supers.get(c, set()):
                            pos_c.add(d)
                        elif d in gci1_bot_partners.get(c, set()):
                            neg_c.add(d)
            else:
                pos_c.update(gci0_supers.get(c, set()))
                neg_c.update(gci1_bot_partners.get(c, set()))

            neg_c.difference_update(pos_c)
            for d in sorted(pos_c):
                pos_rows.append((w_row, d))
            for d in sorted(neg_c):
                neg_rows.append((w_row, d))

        classes_tensor = th.tensor(witness_classes, dtype=th.long)
        pos_tensor = (
            th.tensor(pos_rows, dtype=th.long)
            if pos_rows
            else th.zeros((0, 2), dtype=th.long)
        )
        neg_tensor = (
            th.tensor(neg_rows, dtype=th.long)
            if neg_rows
            else th.zeros((0, 2), dtype=th.long)
        )
        return cls(classes=classes_tensor, pos=pos_tensor, neg=neg_tensor)


@dataclass
class ExistenceTargets:
    """Satisfiable classes, conjunction pairs, and optional witness targets.

    Guarantees:

    - ``classes`` is a 1-D ``torch.long`` tensor of shape ``(C,)`` and ``pairs``
      is a 2-D ``torch.long`` tensor of shape ``(P, 2)`` with ``pairs[:, 0] < pairs[:, 1]``.
    - ``top_idx`` and ``bottom_idx`` are excluded from both ``classes`` and ``pairs``.
    - Pairs are unordered, deduplicated, and strictly satisfy ``C != D``.
    - When ``witness`` is present, ``witness.classes`` is aligned with ``classes``
      and excludes ``top_idx`` and ``bottom_idx``.

    Non-guarantees:

    - Without a :class:`~mowl.baselines.geometric.core.completion.CanonicalModel`, satisfiability
      is approximated syntactically from ``gci0_bot`` and ``gci1_bot`` rows only.
    - Auxiliary classes minted by :class:`~mowl.ontology.normalize.ELNormalizer`
      (``http://mowl.borg/aux/...``) are unknown to a
      :class:`~mowl.baselines.geometric.core.completion.CanonicalModel` built with
      ``mowl.geometric``'s own normaliser, so their entailed unsatisfiability
      cannot be detected via ``cm`` and falls back to the asserted ``gci0_bot``
      and ``gci1_bot`` rules only.
    - ``owl:Thing`` (``top_idx``) is excluded from ``classes`` and ``pairs``, so
      axioms with :math:`\\top` on the left-hand side can be satisfied vacuously
      if :math:`\\top` collapses to an empty region.

    Attributes
    ----------
    classes : torch.Tensor
        1-D ``torch.long`` tensor of satisfiable class indices, shape ``(C,)``.
    pairs : torch.Tensor
        2-D ``torch.long`` tensor of satisfiable conjunction pairs, shape ``(P, 2)``.
    witness : Optional[WitnessTargets], default None
        Optional positive/negative membership targets for per-class witness points.
    """

    classes: th.Tensor
    pairs: th.Tensor
    witness: Optional[WitnessTargets] = None

    @classmethod
    def from_normal_forms(
        cls,
        gci0_bot: th.Tensor,
        gci1: th.Tensor,
        gci1_bot: th.Tensor,
        n_classes: int,
        top_idx: Optional[int],
        bottom_idx: Optional[int],
        index_to_iri: Optional[Dict[int, str]] = None,
        cm: Optional[CanonicalModel] = None,
        gci0: Optional[th.Tensor] = None,
        witness: bool = False,
    ) -> ExistenceTargets:
        """Build :class:`ExistenceTargets` from normal forms and an optional :class:`CanonicalModel`.

        Guarantees:

        - With ``cm`` provided, uses ``cm.is_satisfiable(iri)`` for classes whose
          IRI is in ``cm.concept_names`` and ``cm.joint_satisfiable([iri_c, iri_d])``
          for ``gci1`` pairs whose IRIs are both in ``cm.concept_names``, falling
          back to the asserted ``gci0_bot``/``gci1_bot`` rule whenever an IRI is
          unknown to ``cm`` (both for single classes and for pairs).
          Raises :class:`ValueError` if ``cm`` is provided with ``index_to_iri=None``.
        - Without ``cm``, includes every class not appearing in ``gci0_bot`` and
          every distinct unordered pair ``(min(C, D), max(C, D))`` from ``gci1``
          where neither ``C`` nor ``D`` appears in ``gci0_bot`` and ``(C, D)``
          does not appear in either order in ``gci1_bot``.
        - Excludes ``top_idx`` and ``bottom_idx`` everywhere.
        - When ``witness=True``, attaches :class:`WitnessTargets` built via
          :meth:`WitnessTargets.from_normal_forms`.

        Non-guarantees:

        - Does not test pairs that do not occur as the left-hand side of any
          ``gci1`` row.
        - Auxiliary classes minted by :class:`~mowl.ontology.normalize.ELNormalizer`
          (``http://mowl.borg/aux/...``) are unknown to a
          :class:`~mowl.baselines.geometric.core.completion.CanonicalModel` built with
          ``mowl.geometric``'s own normaliser, so their entailed unsatisfiability
          cannot be detected via ``cm`` (asserted ``gci0_bot``/``gci1_bot`` rules only).

        Parameters
        ----------
        gci0_bot : torch.Tensor
            Long tensor of shape ``(N, 2)`` with rows ``(C, bottom_idx)``.
        gci1 : torch.Tensor
            Long tensor of shape ``(N1, 3)`` with rows ``(C, D, E)``.
        gci1_bot : torch.Tensor
            Long tensor of shape ``(N2, 2)`` with rows ``(C, D)``.
        n_classes : int
            Total number of classes.
        top_idx : Optional[int]
            Class index of ``owl:Thing`` if present, else ``None``.
        bottom_idx : Optional[int]
            Class index of ``owl:Nothing`` if present, else ``None``.
        index_to_iri : Optional[Dict[int, str]], default None
            Mapping from class index to IRI string (required when ``cm`` is given).
        cm : Optional[CanonicalModel], default None
            Saturated canonical model for semantic satisfiability queries.
        gci0 : Optional[torch.Tensor], default None
            Optional ``gci0`` rows ``(C, D)`` used when ``witness=True``.
        witness : bool, default False
            Whether to also populate the ``witness`` attribute with
            :class:`WitnessTargets`.

        Returns
        -------
        ExistenceTargets
            Target class and pair index tensors (plus optional ``witness`` targets).
        """
        if cm is not None and index_to_iri is None:
            raise ValueError("index_to_iri must be provided when cm is not None.")

        excluded: Set[int] = {
            int(idx) for idx in (top_idx, bottom_idx) if idx is not None
        }

        unsat_classes_asserted: Set[int] = set()
        if gci0_bot is not None and gci0_bot.numel() > 0:
            for row in gci0_bot.detach().cpu().reshape(-1, 2).tolist():
                unsat_classes_asserted.add(int(row[0]))

        unsat_pairs_asserted: Set[Tuple[int, int]] = set()
        if gci1_bot is not None and gci1_bot.numel() > 0:
            for row in gci1_bot.detach().cpu().reshape(-1, 2).tolist():
                c_raw, d_raw = int(row[0]), int(row[1])
                unsat_pairs_asserted.add((min(c_raw, d_raw), max(c_raw, d_raw)))

        cm_concepts: Set[str] = set()
        if cm is not None:
            cn_attr = getattr(cm, "concept_names", set())
            cm_concepts = set(cn_attr() if callable(cn_attr) else cn_attr)

        target_classes: List[int] = []
        for c in range(n_classes):
            if c in excluded:
                continue
            if cm is not None:
                assert index_to_iri is not None
                iri = index_to_iri.get(c)
                if iri is not None and iri in cm_concepts:
                    if cm.is_satisfiable(iri):
                        target_classes.append(c)
                else:
                    if c not in unsat_classes_asserted:
                        target_classes.append(c)
            else:
                if c not in unsat_classes_asserted:
                    target_classes.append(c)

        candidate_pairs: Set[Tuple[int, int]] = set()
        if gci1 is not None and gci1.numel() > 0:
            for row in gci1.detach().cpu().reshape(-1, 3).tolist():
                c_raw, d_raw = int(row[0]), int(row[1])
                if c_raw == d_raw or c_raw in excluded or d_raw in excluded:
                    continue
                if not (0 <= c_raw < n_classes and 0 <= d_raw < n_classes):
                    continue
                candidate_pairs.add((min(c_raw, d_raw), max(c_raw, d_raw)))

        target_pairs: List[Tuple[int, int]] = []
        for c, d in sorted(candidate_pairs):
            if cm is not None:
                assert index_to_iri is not None
                iri_c = index_to_iri.get(c)
                iri_d = index_to_iri.get(d)
                if (
                    iri_c is not None
                    and iri_d is not None
                    and iri_c in cm_concepts
                    and iri_d in cm_concepts
                ):
                    if cm.joint_satisfiable([iri_c, iri_d]):
                        target_pairs.append((c, d))
                else:
                    if (
                        c not in unsat_classes_asserted
                        and d not in unsat_classes_asserted
                        and (c, d) not in unsat_pairs_asserted
                    ):
                        target_pairs.append((c, d))
            else:
                if (
                    c not in unsat_classes_asserted
                    and d not in unsat_classes_asserted
                    and (c, d) not in unsat_pairs_asserted
                ):
                    target_pairs.append((c, d))

        classes_tensor = th.tensor(target_classes, dtype=th.long)
        if target_pairs:
            pairs_tensor = th.tensor(target_pairs, dtype=th.long)
        else:
            pairs_tensor = th.zeros((0, 2), dtype=th.long)

        witness_targets: Optional[WitnessTargets] = None
        if witness:
            witness_targets = WitnessTargets.from_normal_forms(
                classes=classes_tensor,
                gci0=gci0,
                gci1_bot=gci1_bot,
                top_idx=top_idx,
                bottom_idx=bottom_idx,
                index_to_iri=index_to_iri,
                cm=cm,
                n_classes=n_classes,
            )

        return cls(classes=classes_tensor, pairs=pairs_tensor, witness=witness_targets)


def _extract_normal_form_rows(training_datasets: Mapping[str, Any], name: str, cols: int) -> th.Tensor:
    """Extract a 2-D ``(N, cols)`` ``torch.long`` CPU tensor for normal-form ``name``."""
    ds = training_datasets.get(name)
    if ds is None:
        return th.zeros((0, cols), dtype=th.long)
    try:
        if len(ds) == 0:
            return th.zeros((0, cols), dtype=th.long)
    except TypeError:
        pass
    rows = ds[:]
    if isinstance(rows, (tuple, list)):
        rows = rows[0]
    rows = th.as_tensor(rows, dtype=th.long).detach().cpu()
    if rows.ndim == 1:
        if rows.numel() == 0:
            return th.zeros((0, cols), dtype=th.long)
        if rows.shape[0] < cols:
            raise ValueError(
                f"Expected at least {cols} columns for {name!r}, got shape {tuple(rows.shape)}."
            )
        rows = rows.unsqueeze(0)
    elif rows.ndim == 2:
        if rows.shape[1] < cols:
            raise ValueError(
                f"Expected at least {cols} columns for {name!r}, got shape {tuple(rows.shape)}."
            )
        if rows.shape[0] == 0:
            return th.zeros((0, cols), dtype=th.long)
    else:
        raise ValueError(
            f"Expected 1-D or 2-D tensor for {name!r}, got shape {tuple(rows.shape)}."
        )
    return rows[:, :cols]


def build_structural_tables(
    model_like: Any,
    *,
    k: Optional[int] = None,
    max_components: Optional[int] = None,
    enabled: Optional[bool] = None,
    max_level: Optional[int] = None,
    cm: Optional[CanonicalModel] = None,
    witness: bool = True,
) -> Tuple[DefinitionTable, ExistenceTargets, Optional[int]]:
    """Build :class:`DefinitionTable`, :class:`ExistenceTargets`, and ``top_idx`` from a model.

    Guarantees:

    - Extracts ``gci0`` (2 cols ``(C, D)``), ``gci0_bot`` (2 cols ``(C, bot)``),
      ``gci1`` (3 cols ``(C, D, E)``), ``gci1_bot`` (2 cols ``(C, D)``),
      ``gci2`` (3 cols ``(C, r, D)`` as stored by :class:`~mowl.datasets.el.GCI2Dataset`),
      and ``gci3`` (3 cols ``(r, C, D)`` as stored by :class:`~mowl.datasets.el.GCI3Dataset`)
      from ``model_like.training_datasets``, raising :class:`ValueError` if any
      tensor has fewer columns than required.
    - Returns ``(definition_table, existence_targets, top_idx)``; when
      ``witness=True`` (the default), ``existence_targets.witness`` is populated
      with :class:`WitnessTargets`.

    Non-guarantees:

    - Auxiliary classes minted by :class:`~mowl.ontology.normalize.ELNormalizer`
      are unknown to a :class:`~mowl.baselines.geometric.core.completion.CanonicalModel` built
      with ``mowl.geometric``'s normaliser, so their satisfiability uses the
      asserted ``gci0_bot``/``gci1_bot`` rules only.

    Parameters
    ----------
    model_like : Any
        Model exposing ``training_datasets`` and ``class_index_dict``, plus
        optional attributes ``n_components`` (or ``k``), ``max_components``,
        ``structural_definitions``, ``max_depth``, and ``canonical_model``.
    k : Optional[int], default None
        Number of primitive components per class (defaults to
        ``getattr(model_like, "n_components", getattr(model_like, "k", 1))``).
    max_components : Optional[int], default None
        Maximum component slots for defined classes (defaults to
        ``getattr(model_like, "max_components", 2**30)``).
    enabled : Optional[bool], default None
        Whether structural definitions are enabled (defaults to
        ``getattr(model_like, "structural_definitions", True)``).
    max_level : Optional[int], default None
        Optional maximum definition depth cap (defaults to
        ``getattr(model_like, "max_depth", None)``).
    cm : Optional[CanonicalModel], default None
        Optional pre-built canonical model (defaults to
        ``getattr(model_like, "canonical_model", None)``).
    witness : bool, default True
        Whether to build and attach :class:`WitnessTargets` on
        ``existence_targets.witness``.

    Returns
    -------
    Tuple[DefinitionTable, ExistenceTargets, Optional[int]]
        Compiled ``(definition_table, existence_targets, top_idx)``.
    """
    td = model_like.training_datasets
    gci0_rows = _extract_normal_form_rows(td, "gci0", 2)
    gci0_bot_rows = _extract_normal_form_rows(td, "gci0_bot", 2)
    gci1_rows = _extract_normal_form_rows(td, "gci1", 3)
    gci1_bot_rows = _extract_normal_form_rows(td, "gci1_bot", 2)
    gci2_rows = _extract_normal_form_rows(td, "gci2", 3)
    gci3_rows = _extract_normal_form_rows(td, "gci3", 3)

    class_to_id = dict(model_like.class_index_dict)
    n_classes = len(class_to_id)
    top_idx = class_to_id.get("http://www.w3.org/2002/07/owl#Thing")
    bottom_idx = class_to_id.get("http://www.w3.org/2002/07/owl#Nothing")
    index_to_iri = {int(idx): str(iri) for iri, idx in class_to_id.items()}

    default_k = getattr(model_like, "n_components", getattr(model_like, "k", 1))
    k_val = int(default_k if k is None else k)
    default_max_comp = getattr(model_like, "max_components", 2**30)
    max_comp_val = int(default_max_comp if max_components is None else max_components)
    enabled_val = bool(
        getattr(model_like, "structural_definitions", True)
        if enabled is None
        else enabled
    )
    max_level_val = getattr(model_like, "max_depth", None) if max_level is None else max_level
    cm_val = getattr(model_like, "canonical_model", None) if cm is None else cm

    table = DefinitionTable.from_normal_forms(
        gci0=gci0_rows,
        gci1=gci1_rows,
        n_classes=n_classes,
        k=k_val,
        max_components=max_comp_val,
        top_idx=top_idx,
        bottom_idx=bottom_idx,
        enabled=enabled_val,
        max_level=max_level_val,
        gci2=gci2_rows,
        gci3=gci3_rows,
    )

    targets = ExistenceTargets.from_normal_forms(
        gci0_bot=gci0_bot_rows,
        gci1=gci1_rows,
        gci1_bot=gci1_bot_rows,
        n_classes=n_classes,
        top_idx=top_idx,
        bottom_idx=bottom_idx,
        index_to_iri=index_to_iri,
        cm=cm_val,
        gci0=gci0_rows,
        witness=bool(witness),
    )
    return table, targets, top_idx
