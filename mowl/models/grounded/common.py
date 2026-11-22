"""Helpers shared by the parts of GroundedEL (tables of axioms,
short names, inverse roles)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch as th
from mowl.nn.el.grounded.module import _validate_inverse_of

class _CountInt(int):
    """Integer count with 0-D tensor-compatible ``.detach()``, ``.item()``, and ``.ndim``."""

    ndim: int = 0

    def detach(self) -> "_CountInt":
        return self

    def item(self) -> int:
        return int(self)


def _resolve_inverse_of(
    inverse_of: Optional[
        Union[Sequence[int], th.Tensor, Mapping[Union[str, int], Union[str, int]]]
    ],
    rel_map: Mapping[str, int],
    nb_rels: int,
) -> Optional[List[int]]:
    """Resolve ``inverse_of`` (mapping role-name -> role-name or index sequence) into a validated index list."""
    if inverse_of is None:
        return None
    if isinstance(inverse_of, Mapping):
        inv_list = [-1] * nb_rels

        def _to_rel_idx(key: Union[str, int]) -> int:
            if isinstance(key, str):
                if key not in rel_map:
                    raise ValueError(
                        f"Unknown role name {key!r} in inverse_of; not in object_property_index_dict"
                    )
                return int(rel_map[key])
            if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)
            ):
                raise ValueError(
                    f"inverse_of keys/values must be role names or integer indices, got {key!r}"
                )
            idx = int(key)
            if idx < 0 or idx >= nb_rels:
                raise ValueError(
                    f"inverse_of role index {idx} out of range [0, {nb_rels})"
                )
            return idx

        for k_raw, v_raw in inverse_of.items():
            k_idx = _to_rel_idx(k_raw)
            if v_raw is None or (
                not isinstance(v_raw, (str, bool, np.bool_))
                and isinstance(v_raw, (int, np.integer))
                and int(v_raw) == -1
            ):
                continue
            v_idx = _to_rel_idx(v_raw)
            if k_idx == v_idx:
                raise ValueError(
                    f"inverse_of cannot contain self-inverse role {k_raw!r} (index {k_idx})"
                )
            if inv_list[k_idx] not in (-1, v_idx) or inv_list[v_idx] not in (
                -1,
                k_idx,
            ):
                raise ValueError(
                    f"inverse_of mapping has conflicting partner assignments for role {k_raw!r} <-> {v_raw!r}"
                )
            inv_list[k_idx] = v_idx
            inv_list[v_idx] = k_idx

        return _validate_inverse_of(nb_rels, inv_list).tolist()

    return _validate_inverse_of(nb_rels, inverse_of).tolist()


_TOL_DEPTH_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("0", 0.0),
    ("1e-5", 1e-5),
    ("1e-4", 1e-4),
    ("1e-3", 1e-3),
    ("1e-2", 1e-2),
)


class _ToleranceCountDict(dict):
    """Dictionary of violated-axiom counts keyed by canonical tolerance string or numeric float."""

    @staticmethod
    def _normalize_key(key: object) -> object:
        if isinstance(key, str):
            if key in ("0", "1e-5", "1e-4", "1e-3", "1e-2"):
                return key
            try:
                val = float(key)
            except ValueError:
                return key
        elif not isinstance(key, bool) and isinstance(key, (int, float)):
            val = float(key)
        else:
            return key
        for k_str, tol_v in _TOL_DEPTH_LEVELS:
            if abs(val - tol_v) <= 1e-12:
                return k_str
        return key

    def __getitem__(self, key: object) -> Any:
        return super().__getitem__(self._normalize_key(key))

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._normalize_key(key))

    def get(self, key: object, default: Any = None) -> Any:
        return super().get(self._normalize_key(key), default)


def _short_iri_map(index_dict: Mapping[str, int]) -> Dict[int, str]:
    """Build an index-to-short-name dictionary, falling back to full IRI on collisions."""
    short_by_idx: Dict[int, str] = {}
    full_by_idx: Dict[int, str] = {}
    counts: Dict[str, int] = {}
    for iri, idx in index_dict.items():
        i = int(idx)
        full_by_idx[i] = str(iri)
        s = str(iri)
        if "#" in s and s.rsplit("#", 1)[1]:
            short = s.rsplit("#", 1)[1]
        elif "/" in s and s.rsplit("/", 1)[1]:
            short = s.rsplit("/", 1)[1]
        else:
            short = s
        short_by_idx[i] = short
        counts[short] = counts.get(short, 0) + 1
    return {
        i: (short_by_idx[i] if counts[short_by_idx[i]] == 1 else full_by_idx[i])
        for i in full_by_idx
    }


class _PrebuiltGCIDataset(th.utils.data.Dataset):
    """Lightweight :class:`torch.utils.data.Dataset` wrapper around a prebuilt 2-D ``torch.long`` GCI tensor.

    Matches the ``.data``, ``__getitem__``, and ``__len__`` interface of
    :class:`~mowl.datasets.gci.GCIDataset` and
    :class:`~mowl.datasets.el.el_dataset.RoleAxiomDataset` so
    :meth:`GroundedEL.compute_batch_losses` and :meth:`GroundedEL.crisp_report`
    consume prebuilt normal-form tables identically to OWL-normalised datasets.
    """

    def __init__(self, data: th.Tensor, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self._data = data.to(device=device, dtype=th.long).contiguous()

    @property
    def data(self) -> th.Tensor:
        """Return the underlying 2-D ``torch.long`` axiom index tensor."""
        return self._data

    def __getitem__(self, idx: Any) -> th.Tensor:
        return self._data[idx]

    def __len__(self) -> int:
        return int(self._data.shape[0])


def _validate_index_dict(name: str, mapping: Mapping[str, int], allow_empty: bool = False) -> Dict[str, int]:
    """Validate that ``mapping`` maps string keys bijectively to ``0 .. len(mapping) - 1``."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a Mapping[str, int], got {type(mapping).__name__}")
    n = len(mapping)
    if n == 0 and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    out: Dict[str, int] = {}
    seen: set = set()
    for k, v in mapping.items():
        if not isinstance(k, str):
            raise TypeError(f"{name} keys must be strings, got {type(k).__name__}")
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"{name} values must be integers, got {type(v).__name__}")
        idx = int(v)
        if idx < 0 or idx >= n:
            raise ValueError(f"{name}[{k!r}] = {idx} out of range [0, {n})")
        if idx in seen:
            raise ValueError(f"{name} contains duplicate index {idx}")
        seen.add(idx)
        out[k] = idx
    return out


def _coerce_gci_table(
    name: str,
    raw: Optional[Any],
    arity: int,
    col_bounds: Tuple[int, ...],
    device: str,
    allow_short_bot_col: bool = False,
    bot_id: Optional[int] = None,
) -> _PrebuiltGCIDataset:
    """Validate and wrap a prebuilt normal-form table into :class:`_PrebuiltGCIDataset`."""
    if raw is None:
        empty = th.empty((0, arity), dtype=th.long, device=device)
        return _PrebuiltGCIDataset(empty, device=device)
    if isinstance(raw, _PrebuiltGCIDataset):
        t = raw.data
    elif hasattr(raw, "data") and isinstance(getattr(raw, "data"), th.Tensor):
        t = getattr(raw, "data")
    elif isinstance(raw, th.Tensor):
        if raw.dtype == th.bool or raw.is_floating_point() or raw.is_complex():
            raise ValueError(f"{name} tensor must have integer dtype, got {raw.dtype}")
        t = raw.to(dtype=th.long)
    elif isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw)
        if arr.size == 0:
            empty = th.empty((0, arity), dtype=th.long, device=device)
            return _PrebuiltGCIDataset(empty, device=device)
        if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"{name} must contain integer indices, got dtype {arr.dtype}")
        t = th.as_tensor(arr, dtype=th.long)
    else:
        raise TypeError(
            f"{name} must be a Tensor, sequence, ndarray, or None, got {type(raw).__name__}"
        )

    if t.numel() == 0:
        empty = th.empty((0, arity), dtype=th.long, device=device)
        return _PrebuiltGCIDataset(empty, device=device)

    if t.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor of shape (N, {arity}), got {tuple(t.shape)}")
    if allow_short_bot_col and t.shape[1] == arity - 1:
        fill_col = int(bot_id) if bot_id is not None else 0
        pad = th.full((t.shape[0], 1), fill_col, dtype=th.long, device=t.device)
        t = th.cat([t, pad], dim=1)
    if t.shape[1] != arity:
        raise ValueError(f"{name} must have {arity} columns, got shape {tuple(t.shape)}")

    for c_i, bound in enumerate(col_bounds):
        col = t[:, c_i]
        c_min = int(col.min().item())
        c_max = int(col.max().item())
        if c_min < 0 or c_max >= bound:
            raise ValueError(
                f"{name} column {c_i} indices out of range [0, {bound}): min={c_min}, max={c_max}"
            )
    return _PrebuiltGCIDataset(t, device=device)
