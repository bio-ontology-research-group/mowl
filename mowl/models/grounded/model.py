"""GroundedEL model: element-grounded EL++ embedding model, training loop, and evaluation.

Implements §3, §4, §5, §7, and §8 of the GroundedEL design:

1. **Syntactic domain construction (:math:`\\Delta`):** Builds one element per
   class in :attr:`class_index_dict` (including normaliser auxiliary classes and
   :math:`\\top, \\bot` if present) followed by one element per individual in
   :attr:`individual_index_dict`.
2. **Reasoner-free training loop (:meth:`train`):** Each epoch anneals
   :math:`\\tau` geometrically from ``tau0`` to ``tau1``, samples mini-batches
   per non-empty axiom type together with candidate elements (§3) and region
   points (§4), sums :math:`L_{\\text{fact}} + L_{\\text{axioms}} + \\lambda_{\\text{reg}} L_{\\text{region}} + L_{\\text{minimality}}`,
   updates parameters with Adam, and optionally performs reasoner-free idle
   component re-seeding (§5).
3. **Evaluation (:meth:`evaluate_model`, :meth:`crisp_report`):** Computes crisp
   element and region violation rates, and (when ``use_reasoner=True`` inside
   :meth:`evaluate_model` only) existence, dead-element accuracy, named-class
   faithfulness precision/recall, and exact-model status.

Guarantees:
- **Reasoner-free training path (R1):** Neither :class:`GroundedEL` construction
  nor :meth:`train` nor :meth:`crisp_report` imports or calls any reasoner or
  semantic completion routine. :class:`~mowl.geometric.completion.CanonicalModel`
  is imported lazily exclusively inside :meth:`evaluate_model` when
  ``use_reasoner=True``.
- **Exact-limit initialisation (R2):** With ``init="elements"`` and caller-supplied
  least-model ``init_labels``, :meth:`evaluate_model` reports ``exact=True``
  (``elem_viol == 0``, ``region_viol == 0``, ``exist == 1``, ``dead_ok == True``)
  at zero training epochs.
- **Monotone capacity (R3):** Expanding module components via
  :meth:`~mowl.nn.GroundedELModule.expand_components` preserves all element
  losses and deterministic evaluation scores.

Non-guarantees:
- Gradient-based optimisation from random initialisation is non-convex and is
  not guaranteed to reach an exact zero-violation model on every random seed or
  at insufficient component capacity :math:`K`.
"""
from __future__ import annotations

import logging
import math
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.grounded import (
    GroundedELModule,
    class_assertion,
    crisp_membership,
    fact,
    gci0,
    gci0_bot,
    gci0_bot_region,
    gci0_region,
    gci1,
    gci1_bot,
    gci1_bot_region,
    gci1_region,
    gci2,
    gci2_dead,
    gci2_region,
    gci3,
    gci3_bot,
    gci3_bot_region,
    gci3_region,
    minimality,
    role_assertion,
    role_chain,
    role_chain_region,
    role_inclusion,
    role_inclusion_region,
    sample_class_region,
    sample_elements,
    sample_filler_elements,
    sample_inside_elements,
    sample_pair_region,
    sample_role_region,
)
from mowl.nn.el.grounded.module import (
    _DEAD_LOGIT,
    _EMPTY_DEPTH_FLOOR,
    _INACTIVE_RAW_WIDTH,
    _LIVE_LOGIT,
    _TOP_DEPTH_CEIL,
    _inv_softplus,
    _validate_int_ge,
    _validate_positive_float,
)
from mowl.nn.el.grounded.losses import _validate_margin, _validate_nonneg_weight
from mowl.nn.el.structural import Regions, inside_measure
from mowl.owlapi.defaults import BOT as BOTTOM, TOP

logger = logging.getLogger(__name__)

__all__ = ["GroundedEL"]

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


class GroundedEL(EmbeddingELModel):
    """Element-grounded multi-box and product-role :math:`\\mathcal{EL}^{++}` embedding model.

    Guarantees:
    - **Reasoner-free training (R1):** :meth:`__init__`, :meth:`init_module`,
      :meth:`train`, :meth:`compute_batch_losses`, :meth:`reseed_idle`, and
      :meth:`crisp_report` never import or invoke any reasoner or semantic
      completion algorithm.
    - **Exact-limit realisation (R2):** Constructed with ``init="elements"`` and
      valid least-model ``init_labels``, achieves ``elem_viol == 0.0`` and
      ``region_viol == 0.0`` with zero training epochs.
    - **Monotone capacity (R3):** Calling ``model.module.expand_components(2 * K)``
      leaves element hinge losses bitwise unchanged.

    Non-guarantees:
    - Does not guarantee global convergence to ``exact=True`` from arbitrary
      random initialisations at finite epoch budgets.
    - When ``use_reasoner=False`` in :meth:`evaluate_model`, reasoner-dependent
      metrics (``exist``, ``dead_ok``, ``faithful_named_precision``,
      ``faithful_named_recall``, ``exact``) are computed syntactically or omitted
      in :meth:`crisp_report`.

    Parameters
    ----------
    dataset : mowl.datasets.Dataset
        mOWL dataset containing the ontology.
    embed_dim : int, default 50
        Euclidean embedding dimension :math:`d \\ge 1`.
    n_components : int, default 2
        Number of box components :math:`K \\ge 1` per class.
    n_role_components : int, default 2
        Number of product-box components :math:`K_r \\ge 1` per role.
    epochs : int, default 1000
        Default number of training epochs when ``epochs`` is not passed to
        :meth:`train`.
    learning_rate : float, default 0.01
        Adam learning rate :math:`> 0`.
    batch_size : int, default 4096
        Mini-batch size :math:`\\ge 1` per axiom type.
    tau0 : float, default 1.0
        Initial sigmoid temperature :math:`\\tau_0 > 0`.
    tau1 : float, default 0.01
        Final sigmoid temperature :math:`\\tau_1 > 0`.
    eps_min : float, default 1e-2
        Minimality weight :math:`\\varepsilon \\ge 0` on random element-class
        memberships.
    eps_role : float, default 1e-2
        Minimality weight :math:`\\varepsilon_r \\ge 0` on random role triples.
    eps_dead : float, default 1e-3
        Minimality weight :math:`\\varepsilon_\\delta \\ge 0` on element dead scores
        (kept small but nonzero because :math:`\\delta = 1` everywhere is a trivial
        zero of all hinges).
    lambda_reg : float, default 1.0
        Weight :math:`\\lambda_{\\text{reg}} \\ge 0` on region soundness hinges (§4).
    n_region_samples : int, default 4
        Number of region points sampled per axiom in each step :math:`\\ge 0`.
    n_adv : int, default 2
        Number of signed-gradient ascent steps :math:`\\ge 0` for adversarial
        region sampling.
    region_margin : float, default 0.0
        Containment margin for geometric region depth hinges (§4).
    n_random_elements : int, default 4
        Number of uniform random elements :math:`\\ge 0` sampled per axiom row.
    n_inside_elements : int, default 32
        Maximum number of elements currently crisp-inside the row's LHS class(es)
        evaluated per axiom row (reasoner-free; reads learned memberships only).
    agg : {"max", "lse"}, default "max"
        Component depth aggregation mode.
    reseed_every : int, default 0
        Epoch interval for reasoner-free idle component re-seeding (``0`` disables
        periodic re-seeding).
    device : str, default "cpu"
        PyTorch device string.
    seed : int, default 0
        Random seed for parameter initialisation, mini-batching, and sampling.
    init : {"random", "elements"}, default "random"
        Initialisation mode: ``"random"`` or ``"elements"`` (which applies
        :meth:`~mowl.nn.GroundedELModule.exact_limit_init` using ``init_labels``).
    init_labels : Optional[Any], default None
        Caller-supplied least-model labels required when ``init="elements"``.
        May be a 2-D tensor/array of shape ``(nb_elements, nb_classes)``, a
        ``(labels, edges)`` tuple, or a dict ``{"labels": ..., "edges": ...}``.
    init_edges : Optional[Any], default None
        Optional caller-supplied role edges for ``init="elements"``. When omitted,
        syntactic edges are derived from ``init_labels`` and the normalised role
        axioms (`gci2`, `object_property_assertion`, `role_inclusion`, `role_chain`).
    init_delta : float, default 0.05
        Positive box half-width :math:`\\delta > 0` for ``init="elements"``.
    log_every : int, default 100
        Epoch interval for logging per-term losses in :meth:`train` (``0`` disables).
    agg_tau : float, default 0.1
        Temperature :math:`\\tau_{\\text{agg}} > 0` when ``agg="lse"``.
    dead_init_logit : float, default 0.0
        Initial value for element deadness :math:`\\delta_x \\in [0, 1]`.
    element_features : Optional[torch.Tensor], default None
        Optional feature matrix for data elements (§6 DeepGO-SE hook).
    encoder : Optional[torch.nn.Module], default None
        Optional encoder network for data elements (§6 DeepGO-SE hook).
    model_filepath : Optional[str], default None
        Optional path for saving model checkpoints.
    load_normalized : bool, default False
        Whether the dataset ontology is already normalised.
    load_role_axioms : bool, default True
        Whether to load ``role_inclusion`` and ``role_chain`` datasets.
    n_filler_samples : int, default 64
        Maximum number of candidate filler elements per batch row in ``gci3``
        and ``gci3_bot`` (always including the row's own filler :math:`x_B`).
    region_lhs_grad : bool, default True
        Whether to keep reparametrised region sample points :math:`z` attached to
        the LHS boxes so gradients flow to both LHS and RHS boxes in the region
        hinges (``False`` detaches :math:`z` for ablation).
    dead_from_bot_only : bool, default True
        Whether element deadness :math:`\\delta_x` receives gradient only from
        :math:`\\bot`-derived hinges (`gci0_bot`, `gci1_bot`, `gci3_bot` element
        hinges and `gci2_dead` filler-dead propagation). When ``False``,
        reproduces the ablation behaviour where violated `gci0` hinges also route
        gradient to :math:`\\delta_x`.
    """

    def __init__(
        self,
        dataset=None,
        embed_dim: int = 50,
        n_components: int = 2,
        n_role_components: int = 2,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        batch_size: int = 4096,
        tau0: float = 1.0,
        tau1: float = 0.01,
        eps_min: float = 1e-2,
        eps_role: float = 1e-2,
        eps_dead: float = 1e-3,
        lambda_reg: float = 1.0,
        n_region_samples: int = 4,
        n_adv: int = 2,
        region_margin: float = 0.0,
        n_random_elements: int = 4,
        n_inside_elements: int = 32,
        agg: str = "max",
        reseed_every: int = 0,
        device: str = "cpu",
        seed: int = 0,
        init: str = "random",
        init_labels: Optional[Any] = None,
        init_edges: Optional[Any] = None,
        init_delta: float = 0.05,
        log_every: int = 100,
        agg_tau: float = 0.1,
        dead_init_logit: float = 0.0,
        element_features: Optional[th.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        model_filepath: Optional[str] = None,
        load_normalized: bool = False,
        load_role_axioms: bool = True,
        n_filler_samples: int = 64,
        region_lhs_grad: bool = True,
        dead_from_bot_only: bool = True,
        *,
        class_index_dict: Optional[Mapping[str, int]] = None,
        object_property_index_dict: Optional[Mapping[str, int]] = None,
        individual_index_dict: Optional[Mapping[str, int]] = None,
        named_class_index_dict: Optional[Mapping[str, int]] = None,
        gci0: Optional[Any] = None,
        gci1: Optional[Any] = None,
        gci2: Optional[Any] = None,
        gci3: Optional[Any] = None,
        gci0_bot: Optional[Any] = None,
        gci1_bot: Optional[Any] = None,
        gci3_bot: Optional[Any] = None,
        role_inclusion: Optional[Any] = None,
        role_chain: Optional[Any] = None,
        class_assertion: Optional[Any] = None,
        object_property_assertion: Optional[Any] = None,
    ) -> None:
        _validate_int_ge("embed_dim", embed_dim, 1)
        _validate_int_ge("n_components", n_components, 1)
        _validate_int_ge("n_role_components", n_role_components, 1)
        _validate_int_ge("epochs", epochs, 0)
        _validate_positive_float("learning_rate", learning_rate)
        _validate_int_ge("batch_size", batch_size, 1)
        tau0_f = _validate_positive_float("tau0", tau0)
        tau1_f = _validate_positive_float("tau1", tau1)
        eps_min_f = _validate_nonneg_weight("eps_min", eps_min)
        eps_role_f = _validate_nonneg_weight("eps_role", eps_role)
        eps_dead_f = _validate_nonneg_weight("eps_dead", eps_dead)
        lambda_reg_f = _validate_nonneg_weight("lambda_reg", lambda_reg)
        _validate_int_ge("n_region_samples", n_region_samples, 0)
        _validate_int_ge("n_adv", n_adv, 0)
        region_margin_f = _validate_margin("region_margin", region_margin)
        _validate_int_ge("n_random_elements", n_random_elements, 0)
        _validate_int_ge("n_inside_elements", n_inside_elements, 0)
        _validate_int_ge("n_filler_samples", n_filler_samples, 1)
        if not isinstance(region_lhs_grad, bool):
            raise ValueError(f"region_lhs_grad must be a bool, got {region_lhs_grad!r}")
        if not isinstance(dead_from_bot_only, bool):
            raise ValueError(
                f"dead_from_bot_only must be a bool, got {dead_from_bot_only!r}"
            )
        if not isinstance(agg, str) or agg not in ("max", "lse"):
            raise ValueError(f"agg must be 'max' or 'lse', got {agg!r}")
        _validate_int_ge("reseed_every", reseed_every, 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        if not isinstance(init, str) or init not in ("random", "elements"):
            raise ValueError(f"init must be 'random' or 'elements', got {init!r}")
        if init == "elements" and init_labels is None:
            raise ValueError("init='elements' requires init_labels to be provided")
        init_delta_f = _validate_positive_float("init_delta", init_delta)
        _validate_int_ge("log_every", log_every, 0)
        agg_tau_f = _validate_positive_float("agg_tau", agg_tau)

        if dataset is None and class_index_dict is None:
            raise TypeError(
                "GroundedEL requires either a mowl Dataset or prebuilt tables (class_index_dict=... or GroundedEL.from_tables(...))"
            )

        if dataset is not None:
            super().__init__(
                dataset,
                embed_dim=embed_dim,
                batch_size=batch_size,
                extended=True,
                model_filepath=model_filepath,
                load_normalized=load_normalized,
                device=device,
                learning_rate=learning_rate,
                neg_sampling_gcis=[],
                load_role_axioms=load_role_axioms,
            )
        else:
            # Prebuilt normal-form table construction without an OWL Dataset
            if model_filepath is not None and not isinstance(model_filepath, str):
                raise TypeError(
                    f"Optional parameter model_filepath must be of type str, got {type(model_filepath).__name__}"
                )
            self.dataset = None
            self.embed_dim = int(embed_dim)
            self.batch_size = int(batch_size)
            self._extended = True
            self._model_filepath = model_filepath
            self._load_normalized = bool(load_normalized)
            self.device = device
            self.learning_rate = float(learning_rate)
            self.neg_sampling_gcis = []
            self._load_role_axioms = bool(load_role_axioms)
            self._eval_gci_name: Optional[str] = None

            cls_map = _validate_index_dict(
                "class_index_dict", class_index_dict, allow_empty=False  # type: ignore[arg-type]
            )
            rel_map = _validate_index_dict(
                "object_property_index_dict",
                object_property_index_dict if object_property_index_dict is not None else {},
                allow_empty=True,
            )
            ind_map = _validate_index_dict(
                "individual_index_dict",
                individual_index_dict if individual_index_dict is not None else {},
                allow_empty=True,
            )
            if named_class_index_dict is not None:
                named_cls_map = dict(named_class_index_dict)
            else:
                named_cls_map = {
                    k: v for k, v in cls_map.items() if k not in (TOP, BOTTOM)
                }

            self._class_index_dict = cls_map
            self._class_index_dict_cache = cls_map
            self._object_property_index_dict = rel_map
            self._object_property_index_dict_cache = rel_map
            self._individual_index_dict = ind_map
            self._named_class_index_dict = named_cls_map
            self._auxiliary_classes = []
            self._auxiliary_object_properties = []

            n_c = len(cls_map)
            n_r = len(rel_map)
            n_i = len(ind_map)
            b_id = cls_map.get(BOTTOM, None)

            td_prebuilt: Dict[str, Any] = {
                "gci0": _coerce_gci_table("gci0", gci0, 2, (n_c, n_c), device),
                "gci1": _coerce_gci_table("gci1", gci1, 3, (n_c, n_c, n_c), device),
                "gci2": _coerce_gci_table("gci2", gci2, 3, (n_c, n_r, n_c), device),
                "gci3": _coerce_gci_table("gci3", gci3, 3, (n_r, n_c, n_c), device),
                "gci0_bot": _coerce_gci_table(
                    "gci0_bot",
                    gci0_bot,
                    2,
                    (n_c, n_c),
                    device,
                    allow_short_bot_col=True,
                    bot_id=b_id,
                ),
                "gci1_bot": _coerce_gci_table(
                    "gci1_bot",
                    gci1_bot,
                    3,
                    (n_c, n_c, n_c),
                    device,
                    allow_short_bot_col=True,
                    bot_id=b_id,
                ),
                "gci3_bot": _coerce_gci_table(
                    "gci3_bot",
                    gci3_bot,
                    3,
                    (n_r, n_c, n_c),
                    device,
                    allow_short_bot_col=True,
                    bot_id=b_id,
                ),
            }
            if role_inclusion is not None or self._load_role_axioms:
                td_prebuilt["role_inclusion"] = _coerce_gci_table(
                    "role_inclusion", role_inclusion, 2, (n_r, n_r), device
                )
            if role_chain is not None or self._load_role_axioms:
                td_prebuilt["role_chain"] = _coerce_gci_table(
                    "role_chain", role_chain, 3, (n_r, n_r, n_r), device
                )
            if class_assertion is not None:
                td_prebuilt["class_assertion"] = _coerce_gci_table(
                    "class_assertion", class_assertion, 2, (n_i, n_c), device
                )
            if object_property_assertion is not None:
                td_prebuilt["object_property_assertion"] = _coerce_gci_table(
                    "object_property_assertion",
                    object_property_assertion,
                    3,
                    (n_i, n_r, n_i),
                    device,
                )

            self._training_datasets = td_prebuilt
            self._validation_datasets = {}
            self._testing_datasets = {}
            self._datasets_loaded = True
            self._dataloaders_loaded = False

        self.n_components = int(n_components)
        self.n_role_components = int(n_role_components)
        self.epochs = int(epochs)
        self.tau0 = tau0_f
        self.tau1 = tau1_f
        self.eps_min = eps_min_f
        self.eps_role = eps_role_f
        self.eps_dead = eps_dead_f
        self.lambda_reg = lambda_reg_f
        self.n_region_samples = int(n_region_samples)
        self.n_adv = int(n_adv)
        self.region_margin = region_margin_f
        self.n_random_elements = int(n_random_elements)
        self.n_inside_elements = int(n_inside_elements)
        self.n_filler_samples = int(n_filler_samples)
        self.region_lhs_grad = bool(region_lhs_grad)
        self.dead_from_bot_only = bool(dead_from_bot_only)
        self.agg = agg
        self.agg_tau = agg_tau_f
        self.reseed_every = int(reseed_every)
        self.seed = int(seed)
        self.init = init
        self.init_labels = init_labels
        self.init_edges = init_edges
        self.init_delta = init_delta_f
        self.log_every = int(log_every)
        self.dead_init_logit = float(dead_init_logit)
        self.element_features = element_features
        self.encoder = encoder

        self.nb_classes = len(self.class_index_dict)
        self.nb_inds = len(self.individual_index_dict)
        self.nb_elements = self.nb_classes + self.nb_inds
        self.nb_rels = len(self.object_property_index_dict)
        self.top_id: Optional[int] = self.class_index_dict.get(TOP, None)
        self.bot_id: Optional[int] = self.class_index_dict.get(BOTTOM, None)

        self.loss_history: List[Dict[str, float]] = []
        self.optimizer: Optional[th.optim.Optimizer] = None
        self._filler_by_role: Optional[Dict[int, th.Tensor]] = None
        self._lhs_by_role: Optional[Dict[int, th.Tensor]] = None
        self.generator: th.Generator = th.Generator(device="cpu")
        self.generator.manual_seed(self.seed + 1337)

        self.init_module()

    @classmethod
    def from_tables(
        cls,
        class_index_dict: Mapping[str, int],
        object_property_index_dict: Optional[Mapping[str, int]] = None,
        gci0: Optional[Any] = None,
        gci1: Optional[Any] = None,
        gci2: Optional[Any] = None,
        gci3: Optional[Any] = None,
        gci0_bot: Optional[Any] = None,
        gci1_bot: Optional[Any] = None,
        gci3_bot: Optional[Any] = None,
        role_inclusion: Optional[Any] = None,
        role_chain: Optional[Any] = None,
        class_assertion: Optional[Any] = None,
        object_property_assertion: Optional[Any] = None,
        individual_index_dict: Optional[Mapping[str, int]] = None,
        named_class_index_dict: Optional[Mapping[str, int]] = None,
        **kwargs: Any,
    ) -> "GroundedEL":
        """Construct :class:`GroundedEL` directly from prebuilt normal-form index tables without an OWL file.

        Produces the exact same ``training_datasets`` dictionary structure
        (``"gci0"``, ``"gci1"``, ``"gci2"``, ``"gci3"``, ``"gci0_bot"``,
        ``"gci1_bot"``, ``"gci3_bot"``, ``"role_inclusion"``, ``"role_chain"``)
        consumed by :meth:`compute_batch_losses` and :meth:`crisp_report`.

        Guarantees:
        - Reasoner-free and JVM-free: never starts a JVM or reads an OWL file.
        - Validates bijective ``0 .. N - 1`` index dictionaries and column bounds
          of all supplied GCI tensors.
        - Leaves the existing :class:`~mowl.datasets.Dataset` constructor path
          completely untouched.

        Non-guarantees:
        - Calling :meth:`evaluate_model` with ``use_reasoner=True`` on a
          table-constructed instance raises :class:`ValueError` because no OWL
          ontology is attached; pass ``use_reasoner=False``.

        Parameters
        ----------
        class_index_dict : Mapping[str, int]
            Bijective mapping from class names/IRIs (including ``TOP`` and
            ``BOTTOM`` if used) to integer indices ``0 .. nb_classes - 1``.
        object_property_index_dict : Optional[Mapping[str, int]], default None
            Bijective mapping from relation names/IRIs to ``0 .. nb_rels - 1``.
        gci0 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N0, 2)`` with rows ``[C, D]``
            for :math:`C \\sqsubseteq D`.
        gci1 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N1, 3)`` with rows ``[C, D, E]``
            for :math:`C \\sqcap D \\sqsubseteq E`.
        gci2 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N2, 3)`` with rows ``[C, R, D]``
            for :math:`C \\sqsubseteq \\exists R.D`.
        gci3 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N3, 3)`` with rows ``[R, C, D]``
            for :math:`\\exists R.C \\sqsubseteq D`.
        gci0_bot : Optional[Any], default None
            Optional 2-D integer tensor/array for :math:`C \\sqsubseteq \\bot`
            (shape ``(N, 2)`` or ``(N, 1)``).
        gci1_bot : Optional[Any], default None
            Optional 2-D integer tensor/array for :math:`C \\sqcap D \\sqsubseteq \\bot`
            (shape ``(N, 3)`` or ``(N, 2)``).
        gci3_bot : Optional[Any], default None
            Optional 2-D integer tensor/array for :math:`\\exists R.C \\sqsubseteq \\bot`
            (shape ``(N, 3)`` or ``(N, 2)``).
        role_inclusion : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 2)`` for :math:`R \\sqsubseteq S`.
        role_chain : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 3)`` for :math:`R \\circ S \\sqsubseteq T`.
        class_assertion : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 2)`` for :math:`C(a)`.
        object_property_assertion : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 3)`` for :math:`R(a, b)`.
        individual_index_dict : Optional[Mapping[str, int]], default None
            Optional bijective mapping from individual names/IRIs to ``0 .. nb_inds - 1``.
        named_class_index_dict : Optional[Mapping[str, int]], default None
            Optional subset of ``class_index_dict`` representing named classes.
        **kwargs : Any
            Additional keyword arguments forwarded to :meth:`__init__`.

        Returns
        -------
        GroundedEL
            Initialised model instance backed by the prebuilt tables.
        """
        return cls(
            None,
            class_index_dict=class_index_dict,
            object_property_index_dict=object_property_index_dict,
            individual_index_dict=individual_index_dict,
            named_class_index_dict=named_class_index_dict,
            gci0=gci0,
            gci1=gci1,
            gci2=gci2,
            gci3=gci3,
            gci0_bot=gci0_bot,
            gci1_bot=gci1_bot,
            gci3_bot=gci3_bot,
            role_inclusion=role_inclusion,
            role_chain=role_chain,
            class_assertion=class_assertion,
            object_property_assertion=object_property_assertion,
            **kwargs,
        )

    @property
    def individual_index_dict(self) -> Dict[str, int]:
        """Return mapping from individual IRI to integer index ``0 .. nb_inds - 1``."""
        cached = getattr(self, "_individual_index_dict", None)
        if cached is not None:
            return cached
        if self.dataset is None:
            self._individual_index_dict = {}
            return self._individual_index_dict
        return super().individual_index_dict

    @property
    def named_class_index_dict(self) -> Dict[str, int]:
        """Return mapping from named class IRI to integer index in ``class_index_dict``."""
        cached = getattr(self, "_named_class_index_dict", None)
        if cached is not None:
            return cached
        if self.dataset is None:
            self._named_class_index_dict = {
                k: v
                for k, v in self.class_index_dict.items()
                if k not in (TOP, BOTTOM)
            }
            return self._named_class_index_dict
        return super().named_class_index_dict

    def init_module(self) -> None:
        """Initialise :class:`~mowl.nn.GroundedELModule` and apply ``init="elements"`` when configured.

        Guarantees:
        - Seeds PyTorch RNG with ``self.seed`` prior to module construction.
        - Fixes ``dead_logits[bot_id] = +100.0`` and marks ``bot_id`` components
          inactive when :math:`\\bot` is present in :attr:`class_index_dict`.
        - When ``init == "elements"``, expands ``n_components`` to at least
          ``nb_elements`` and calls :meth:`~mowl.nn.GroundedELModule.exact_limit_init`.

        Non-guarantees:
        - Mutates global ``torch.manual_seed(self.seed)`` during module
          construction for deterministic parameter initialisation.
        """
        th.manual_seed(self.seed)
        eff_k = (
            max(self.n_components, self.nb_elements)
            if self.init == "elements"
            else self.n_components
        )
        self.n_components = eff_k
        self.module = GroundedELModule(
            nb_elements=self.nb_elements,
            nb_classes=self.nb_classes,
            nb_rels=self.nb_rels,
            embed_dim=self.embed_dim,
            n_components=self.n_components,
            n_role_components=self.n_role_components,
            tau=self.tau0,
            agg=self.agg,
            agg_tau=self.agg_tau,
            dead_init_logit=self.dead_init_logit,
            element_features=self.element_features,
            encoder=self.encoder,
            top_id=self.top_id,
            bot_id=self.bot_id,
        ).to(self.device)

        with th.no_grad():
            if self.bot_id is not None:
                self.module.dead_logits[self.bot_id] = _DEAD_LOGIT
                self.module.class_active[self.bot_id, :] = False
                self.module.class_raw_widths[self.bot_id, :] = _INACTIVE_RAW_WIDTH

        if self.init == "elements":
            labels_t, edges_list = self._resolve_init_labels_and_edges()
            self.module.exact_limit_init(
                labels_t,
                edges=edges_list,
                delta=self.init_delta,
                seed=self.seed,
            )
            self.n_components = self.module.n_components
            self.n_role_components = self.module.n_role_components

    def _load_datasets(self) -> None:
        """Load and deterministically sort ``training_datasets`` rows so training is invariant to JVM HashSet order."""
        was_loaded = bool(self._datasets_loaded)
        if not was_loaded:
            random.seed(self.seed)
        super()._load_datasets()
        if not was_loaded and self._training_datasets is not None:
            for ds_obj in self._training_datasets.values():
                t = getattr(ds_obj, "data", None)
                if isinstance(t, th.Tensor) and t.ndim == 2 and t.shape[0] > 1:
                    order = th.arange(t.shape[0], dtype=th.long, device=t.device)
                    for col in range(t.shape[1] - 1, -1, -1):
                        perm = th.argsort(t[order, col], stable=True)
                        order = order[perm]
                    t.copy_(t[order])

    def _resolve_init_labels_and_edges(
        self,
    ) -> Tuple[th.Tensor, List[Tuple[int, int, int]]]:
        """Convert caller-supplied ``init_labels`` (and optional ``init_edges``) into ``(labels_tensor, edges_list)``."""
        raw_labels = self.init_labels
        raw_edges = self.init_edges

        if isinstance(raw_labels, tuple) and len(raw_labels) == 2:
            raw_labels, raw_edges = raw_labels[0], raw_labels[1]
        elif isinstance(raw_labels, Mapping) and "labels" in raw_labels:
            if raw_edges is None and "edges" in raw_labels:
                raw_edges = raw_labels["edges"]
            raw_labels = raw_labels["labels"]

        if isinstance(raw_labels, (th.Tensor, np.ndarray)):
            labels_t = th.as_tensor(raw_labels, dtype=th.bool, device=self.device).clone()
            if labels_t.shape != (self.nb_elements, self.nb_classes):
                raise ValueError(
                    f"init_labels tensor must have shape ({self.nb_elements}, {self.nb_classes}), "
                    f"got {tuple(labels_t.shape)}"
                )
        elif isinstance(raw_labels, Mapping):
            # Support IRI-keyed or index-keyed dictionary mapping class -> elements or element -> classes
            labels_t = th.zeros(
                (self.nb_elements, self.nb_classes), dtype=th.bool, device=self.device
            )
            elem_iri_to_idx: Dict[str, int] = dict(self.class_index_dict)
            for ind_iri, ind_idx in self.individual_index_dict.items():
                elem_iri_to_idx[ind_iri] = self.nb_classes + int(ind_idx)

            for k_key, vals in raw_labels.items():
                if isinstance(k_key, str) and k_key in self.class_index_dict:
                    c_idx = self.class_index_dict[k_key]
                    for v in vals:
                        e_idx = elem_iri_to_idx[v] if isinstance(v, str) else int(v)
                        labels_t[e_idx, c_idx] = True
                else:
                    e_idx = elem_iri_to_idx[k_key] if isinstance(k_key, str) else int(k_key)
                    for v in vals:
                        c_idx = self.class_index_dict[v] if isinstance(v, str) else int(v)
                        labels_t[e_idx, c_idx] = True
        else:
            raise TypeError(
                f"Unsupported init_labels type {type(raw_labels).__name__}; expected Tensor, ndarray, tuple, or Mapping"
            )

        if self.bot_id is not None:
            labels_t[:, self.bot_id] = False
            labels_t[self.bot_id, :] = False
        if self.top_id is not None:
            # Ensure live elements belong to TOP, and x_TOP belongs to TOP when consistent
            live_mask = th.zeros((self.nb_elements,), dtype=th.bool, device=self.device)
            cls_arange = th.arange(self.nb_classes, device=self.device)
            live_mask[: self.nb_classes] = labels_t[cls_arange, cls_arange]
            if self.nb_inds > 0:
                live_mask[self.nb_classes :] = labels_t[self.nb_classes :].any(dim=1)
            if bool(live_mask.any().item()) and not bool(labels_t[self.top_id, self.top_id].item()):
                labels_t[self.top_id, self.top_id] = True
                live_mask[self.top_id] = True
            labels_t[live_mask, self.top_id] = True

        if raw_edges is not None:
            edges_list: List[Tuple[int, int, int]] = []
            if isinstance(raw_edges, Mapping):
                for r_key, pairs in raw_edges.items():
                    r_idx = (
                        self.object_property_index_dict[r_key]
                        if isinstance(r_key, str)
                        else int(r_key)
                    )
                    for u, v in pairs:
                        edges_list.append((int(u), int(v), int(r_idx)))
            elif isinstance(raw_edges, (th.Tensor, np.ndarray)):
                arr = th.as_tensor(raw_edges, dtype=th.long).cpu()
                if arr.numel() > 0:
                    for row in arr.tolist():
                        edges_list.append((int(row[0]), int(row[1]), int(row[2])))
            else:
                for u, v, r in raw_edges:
                    r_idx = (
                        self.object_property_index_dict[r]
                        if isinstance(r, str)
                        else int(r)
                    )
                    edges_list.append((int(u), int(v), int(r_idx)))
            return labels_t, edges_list

        # Derive syntactic edges from labels_t and normalised training datasets (CR3, CR5, CR6, ABox)
        edges_set = set()
        td = self.training_datasets
        cls_arange = th.arange(self.nb_classes, device=self.device)
        live_elem = th.zeros((self.nb_elements,), dtype=th.bool, device=self.device)
        live_elem[: self.nb_classes] = labels_t[cls_arange, cls_arange]
        if self.nb_inds > 0:
            live_elem[self.nb_classes :] = labels_t[self.nb_classes :].any(dim=1)

        if "object_property_assertion" in td and len(td["object_property_assertion"]) > 0:
            for row in td["object_property_assertion"][:].detach().cpu().tolist():
                ind_a, r_id, ind_b = int(row[0]), int(row[1]), int(row[2])
                edges_set.add((self.nb_classes + ind_a, self.nb_classes + ind_b, r_id))

        if "gci2" in td and len(td["gci2"]) > 0:
            for row in td["gci2"][:].detach().cpu().tolist():
                a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                if not bool(live_elem[b_id].item()):
                    continue
                xs = th.nonzero(labels_t[:, a_id] & live_elem, as_tuple=False).view(-1).cpu().tolist()
                for x_id in xs:
                    edges_set.add((int(x_id), b_id, r_id))

        r_inc: List[Tuple[int, int]] = []
        if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
            r_inc = [
                (int(row[0]), int(row[1]))
                for row in td["role_inclusion"][:].detach().cpu().tolist()
            ]
        r_chain: List[Tuple[int, int, int]] = []
        if "role_chain" in td and len(td["role_chain"]) > 0:
            r_chain = [
                (int(row[0]), int(row[1]), int(row[2]))
                for row in td["role_chain"][:].detach().cpu().tolist()
            ]

        if r_inc or r_chain:
            changed = True
            while changed:
                changed = False
                current = list(edges_set)
                for u, v, r_sub in current:
                    for sub_id, sup_id in r_inc:
                        if r_sub == sub_id and (u, v, sup_id) not in edges_set:
                            edges_set.add((u, v, sup_id))
                            changed = True
                if r_chain:
                    by_role: Dict[int, List[Tuple[int, int]]] = {}
                    for u, v, r_id in edges_set:
                        by_role.setdefault(r_id, []).append((u, v))
                    for r1, r2, sup_id in r_chain:
                        for u, mid1 in by_role.get(r1, ()):
                            for mid2, w_el in by_role.get(r2, ()):
                                if mid1 == mid2 and (u, w_el, sup_id) not in edges_set:
                                    edges_set.add((u, w_el, sup_id))
                                    changed = True

        return labels_t, sorted(edges_set)

    def _ensure_role_tables(self) -> None:
        """Build cached per-role filler and LHS element index tables from ``training_datasets``."""
        if self._filler_by_role is not None and self._lhs_by_role is not None:
            return
        td = self.training_datasets
        fillers: Dict[int, Dict[int, None]] = {r: {} for r in range(self.nb_rels)}
        lhs_map: Dict[int, Dict[int, None]] = {r: {} for r in range(self.nb_rels)}

        if "gci2" in td and len(td["gci2"]) > 0:
            for row in td["gci2"][:].detach().cpu().tolist():
                a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                fillers[r_id][b_id] = None
                lhs_map[r_id][a_id] = None

        if "object_property_assertion" in td and len(td["object_property_assertion"]) > 0:
            for row in td["object_property_assertion"][:].detach().cpu().tolist():
                ind_a, r_id, ind_b = int(row[0]), int(row[1]), int(row[2])
                el_a = self.nb_classes + ind_a
                el_b = self.nb_classes + ind_b
                fillers[r_id][el_b] = None
                lhs_map[r_id][el_a] = None

        self._filler_by_role = {
            r: th.tensor(list(vals.keys()), dtype=th.long, device=self.device)
            for r, vals in fillers.items()
        }
        self._lhs_by_role = {
            r: th.tensor(list(vals.keys()), dtype=th.long, device=self.device)
            for r, vals in lhs_map.items()
        }

    def _sample_rows(self, data: th.Tensor, gen: th.Generator) -> th.Tensor:
        """Sample up to ``self.batch_size`` rows from ``data`` using ``gen``."""
        n = int(data.shape[0])
        if n <= self.batch_size:
            return data.to(device=self.device, dtype=th.long)
        idx = th.randint(
            0, n, (self.batch_size,), dtype=th.long, device=gen.device, generator=gen
        ).to(device=data.device)
        return data[idx].to(device=self.device, dtype=th.long)

    def _batch_candidate_elements(
        self,
        batch_elems: th.Tensor,
        gen: th.Generator,
    ) -> th.Tensor:
        """Return candidate batch elements (all elements when ``nb_elements <= 32``, else subsampled)."""
        if self.nb_elements <= 32:
            return th.arange(self.nb_elements, dtype=th.long, device=self.device)
        uniq = th.unique(batch_elems)
        max_b = max(self.n_random_elements, 8)
        if uniq.numel() <= max_b:
            return uniq
        perm = th.randperm(uniq.numel(), device=gen.device, generator=gen)[
            :max_b
        ].to(device=uniq.device)
        return uniq[perm]

    def _inside_candidate_elements(
        self,
        lhs_classes: th.Tensor,
        gen: th.Generator,
        pos_all: th.Tensor,
    ) -> Optional[th.Tensor]:
        """Return up to ``n_inside_elements`` crisp-inside elements when not already covered by full ``_batch_candidate_elements``."""
        if self.n_inside_elements <= 0 or self.nb_elements <= 32:
            return None
        return sample_inside_elements(
            self.module,
            lhs_classes,
            self.n_inside_elements,
            gen,
            pos_all=pos_all,
        )

    def _sample_batch_y_fillers(
        self,
        rows: th.Tensor,
        gen: th.Generator,
    ) -> th.Tensor:
        """Subsample ``all_fillers`` to at most ``self.n_filler_samples`` per batch, always including each row's own filler ``rows[:, 1]``."""
        bsz = int(rows.shape[0])
        assert self._filler_by_role is not None
        role_filler_parts = [rows[:, 1]] + [
            self._filler_by_role[int(r_i)]
            for r_i in th.unique(rows[:, 0]).cpu().tolist()
            if int(r_i) in self._filler_by_role
        ]
        all_fillers = th.unique(th.cat(role_filler_parts))
        if all_fillers.numel() <= self.n_filler_samples:
            return sample_filler_elements(
                all_fillers,
                self.nb_elements,
                self.n_random_elements,
                generator=gen,
                batch_size=bsz,
            )
        n_shared = max(self.n_filler_samples - 1, 0)
        if n_shared > 0:
            perm = th.randperm(
                all_fillers.numel(), device=gen.device, generator=gen
            )[:n_shared].to(device=all_fillers.device)
            sub_fillers = all_fillers[perm]
        else:
            sub_fillers = th.empty((0,), dtype=th.long, device=self.device)
        y_rest = sample_filler_elements(
            sub_fillers,
            self.nb_elements,
            self.n_random_elements,
            generator=gen,
            batch_size=bsz,
        )
        return th.cat([rows[:, 1].unsqueeze(1), y_rest], dim=1)

    def _grid_role_membership(
        self,
        pos_all: th.Tensor,
        x_mat: th.Tensor,
        y_mat: th.Tensor,
        r_ids: th.Tensor,
        need_crisp: bool = False,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """Compute ``role_membership`` (and optional crisp role depth) on a ``(B, sx, sy)`` grid by reducing over ``d`` on ``(B, sx)`` and ``(B, sy)`` first."""
        mod = self.module
        bsz, sx = x_mat.shape
        sy = int(y_mat.shape[1])
        h_reg, t_reg = mod.role_regions(r_ids)
        px_2d = pos_all[x_mat]  # (B, sx, d)
        py_2d = pos_all[y_mat]  # (B, sy, d)
        h_cen = h_reg.cen.to(dtype=px_2d.dtype)
        h_w = h_reg.w.to(dtype=px_2d.dtype)
        t_cen = t_reg.cen.to(dtype=py_2d.dtype)
        t_w = t_reg.w.to(dtype=py_2d.dtype)

        h_dep = th.amin(
            h_w.unsqueeze(1) - th.abs(px_2d.unsqueeze(2) - h_cen.unsqueeze(1)), dim=-1
        )  # (B, sx, K_r)
        t_dep = th.amin(
            t_w.unsqueeze(1) - th.abs(py_2d.unsqueeze(2) - t_cen.unsqueeze(1)), dim=-1
        )  # (B, sy, K_r)
        slot_depth = th.minimum(h_dep.unsqueeze(2), t_dep.unsqueeze(1)).reshape(
            bsz * sx * sy, -1
        )  # (B * sx * sy, K_r)
        act_flat = (
            h_reg.active.view(bsz, 1, 1, -1)
            .expand(bsz, sx, sy, -1)
            .reshape(bsz * sx * sy, -1)
        )
        dep_soft = mod._aggregate_slot_depth(slot_depth, act_flat, crisp=False)
        rho_xy = th.sigmoid(dep_soft / mod.tau.to(dtype=dep_soft.dtype))
        if not need_crisp:
            return rho_xy, None
        dep_crisp = mod._aggregate_slot_depth(slot_depth, act_flat, crisp=True)
        return rho_xy, dep_crisp

    def _depth_and_membership_pos_grad_only(
        self,
        points: th.Tensor,
        class_ids: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute ``(depth, membership)`` with class box parameters detached so gradients flow only to ``points``."""
        mod = self.module
        reg = mod.class_regions(class_ids)
        slot_dep = th.amin(
            reg.w.detach() - th.abs(points.unsqueeze(1) - reg.cen.detach()),
            dim=-1,
        )
        dep = mod._aggregate_slot_depth(slot_dep, reg.active, crisp=False)
        m = th.sigmoid(dep / mod.tau.to(dtype=dep.dtype))
        if self.top_id is not None:
            m = th.where(class_ids == self.top_id, th.ones_like(m), m)
        if self.bot_id is not None:
            m = th.where(class_ids == self.bot_id, th.zeros_like(m), m)
        return dep, m

    @property
    def tau_s(self) -> th.Tensor:
        """Return the learnable inductive score temperature :math:`\\tau_s = \\max(\\operatorname{softplus}(\\text{raw}), 10^{-3})`."""
        return self.module.tau_s

    def reset_generator(self, seed: Optional[int] = None) -> th.Generator:
        """Reset and return ``self.generator`` for external training loops.

        Guarantees:
        - Seeds ``self.generator`` with ``seed`` when provided, or ``self.seed + 1337``
          (matching :meth:`train`) when ``seed is None``.

        Non-guarantees:
        - Mutates ``self.generator`` in-place.
        """
        eff_seed = (self.seed + 1337) if seed is None else int(seed)
        self.generator.manual_seed(eff_seed)
        return self.generator

    def scheduled_tau(self, step: int, total_steps: int) -> float:
        """Return geometrically annealed :math:`\\tau` at ``(step, total_steps)`` from ``tau0`` to ``tau1``.

        Guarantees:
        - Pure function of ``(step, total_steps, self.tau0, self.tau1)``.
        - Returns ``self.tau0`` when ``total_steps <= 1`` or ``step <= 0``.
        - Returns ``self.tau1`` at ``step == total_steps - 1`` when ``total_steps > 1``.

        Non-guarantees:
        - Does not mutate ``self.module.tau`` (use :meth:`step_tau` to update in-place).
        """
        _validate_int_ge("step", step, 0)
        _validate_int_ge("total_steps", total_steps, 1)
        if total_steps <= 1:
            return float(self.tau0)
        frac = min(max(step / (total_steps - 1), 0.0), 1.0)
        return float(self.tau0 * ((self.tau1 / self.tau0) ** frac))

    def step_tau(self, step: int, total_steps: int) -> float:
        """Update ``self.module.tau`` according to the geometric annealing schedule and return the new ``tau``."""
        tau_val = self.scheduled_tau(step, total_steps)
        self.module.set_tau(tau_val)
        return tau_val

    def scheduled_lr(
        self,
        step: int,
        total_steps: int,
        lr: Optional[float] = None,
        lr_final: Optional[float] = None,
        lr_schedule: str = "cosine",
    ) -> float:
        """Compute scheduled learning rate at ``(step, total_steps)`` for external training loops."""
        _validate_int_ge("step", step, 0)
        _validate_int_ge("total_steps", total_steps, 1)
        base_lr = self.learning_rate if lr is None else _validate_positive_float("lr", lr)
        eff_final = (
            None
            if lr_final is None
            else _validate_positive_float("lr_final", lr_final)
        )
        if not isinstance(lr_schedule, str) or lr_schedule not in ("cosine", "linear"):
            raise ValueError(
                f"lr_schedule must be 'cosine' or 'linear', got {lr_schedule!r}"
            )
        return self._scheduled_lr(step, total_steps, base_lr, eff_final, lr_schedule)

    def clamp_dead_logits(self) -> None:
        """Clamp ``self.module.dead_logits`` to ``[0, 1]`` and keep ``dead_logits[bot_id] == +100.0``."""
        with th.no_grad():
            self.module.dead_logits.clamp_(0.0, 1.0)
            if self.bot_id is not None:
                self.module.dead_logits[self.bot_id] = _DEAD_LOGIT

    def _encode_features(self, features: th.Tensor) -> th.Tensor:
        """Validate ``features`` and map through ``self.module.encoder`` to ``(B, embed_dim)`` positions."""
        mod = self.module
        enc = mod.encoder if mod.encoder is not None else self.encoder
        if enc is None:
            raise ValueError(
                "GroundedEL.encoder is None; pass encoder=... at construction to use compute_data_loss or score"
            )
        if mod.encoder is None:
            mod.encoder = enc.to(self.device)
        if not isinstance(features, th.Tensor):
            raise TypeError(
                f"features must be a torch.Tensor, got {type(features).__name__}"
            )
        if features.ndim == 1:
            feat_2d = features.unsqueeze(0)
        elif features.ndim == 2:
            feat_2d = features
        else:
            raise ValueError(
                f"features must be 2-D (B, feat_dim) (or 1-D (feat_dim,)), got shape {tuple(features.shape)}"
            )
        feat_dev = feat_2d.to(device=self.device, dtype=mod.class_centers.dtype)
        pos = enc(feat_dev)
        if pos.ndim != 2 or pos.shape[0] != feat_dev.shape[0] or pos.shape[1] != self.embed_dim:
            raise ValueError(
                f"encoder(features) must return shape ({feat_dev.shape[0]}, {self.embed_dim}), got {tuple(pos.shape)}"
            )
        return pos

    def _chunked_term_scores(
        self,
        pos: th.Tensor,
        term_ids: th.Tensor,
        max_elements: int = 8_388_608,
    ) -> th.Tensor:
        """Compute ``sigmoid(depth(pos, term_ids, crisp=True) / tau_s)`` in term chunks bounded by ``max_elements``."""
        mod = self.module
        bsz = int(pos.shape[0])
        n_terms = int(term_ids.numel())
        tau_s = mod.tau_s.to(dtype=pos.dtype)
        if bsz == 0 or n_terms == 0:
            return th.empty((bsz, n_terms), dtype=pos.dtype, device=pos.device)

        elems_per_term = max(1, bsz * max(1, mod.n_components) * max(1, self.embed_dim))
        term_chunk = max(1, int(max_elements) // elems_per_term)

        chunks: List[th.Tensor] = []
        for t_start in range(0, n_terms, term_chunk):
            t_end = min(t_start + term_chunk, n_terms)
            c_sub = term_ids[t_start:t_end]
            c_len = int(c_sub.numel())
            pos_exp = (
                pos.unsqueeze(1)
                .expand(bsz, c_len, self.embed_dim)
                .reshape(bsz * c_len, self.embed_dim)
            )
            c_exp = c_sub.unsqueeze(0).expand(bsz, c_len).reshape(bsz * c_len)
            dep_sub = mod.depth(pos_exp, c_exp, crisp=True).view(bsz, c_len)
            s_sub = th.sigmoid(dep_sub / tau_s)
            chunks.append(s_sub)

        if len(chunks) == 1:
            return chunks[0]
        return th.cat(chunks, dim=1)

    def compute_data_loss(
        self,
        features: th.Tensor,
        label_matrix: th.Tensor,
        term_class_ids: Union[Sequence[int], th.Tensor],
        max_elements: int = 8_388_608,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute inductive binary cross-entropy loss and scores for encoded data elements (§6 / E3).

        Computes ``positions = self.encoder(features)`` (where ``self.encoder`` is a
        trainable submodule registered on ``self.module`` and part of the model's
        parameters/optimizer), crisp box depths ``depth(positions, term_class_ids, crisp=True)``,
        and scores ``s = sigmoid(depth(positions, term_class_ids, crisp=True) / tau_s)``
        with learnable scalar temperature ``tau_s = max(softplus(raw_tau_s), 1e-3)``
        (initialised to ``0.1``). Returns ``(BCE(s, label_matrix), s)`` with mean
        reduction.

        Data elements never get a dead score (:math:`\\delta \\equiv 0`) or a
        minimality penalty on their positions: there are no data elements in
        :math:`\\Delta` here; proteins/data instances exist only through
        ``self.encoder`` at forward/scoring time.

        Guarantees:
        - Reasoner-free (R1): never imports or calls any reasoner or completion algorithm.
        - Chunks computation over ``term_class_ids`` whenever
          ``B * n_terms * K * d > max_elements`` so peak intermediate memory stays
          bounded even on large ontologies (e.g. GO BP with ``128 * 21356 * K * d``).
        - Propagates gradients to ``self.encoder``, ``self.module.raw_tau_s``, and
          the queried class boxes in ``self.module``.
        - Never creates or reads ``dead_logits`` or minimality terms for the encoded
          data positions.

        Non-guarantees:
        - Under ``crisp=True`` (hard ``max`` over active component slots), box
          gradients for each ``(protein, term)`` pair route only to the single
          active component slot attaining the maximum depth.

        Parameters
        ----------
        features : torch.Tensor
            Input feature matrix of shape ``(B, feat_dim)``.
        label_matrix : torch.Tensor
            Binary target matrix of shape ``(B, n_terms)`` with entries in ``[0, 1]``.
        term_class_ids : Union[Sequence[int], torch.Tensor]
            1-D sequence or tensor of ``n_terms`` class indices in
            ``[0, nb_classes)`` (e.g. the ``Aux_G`` class indices).
        max_elements : int, default 8_388_608
            Maximum ``B * term_chunk * K * d`` elements per chunk before splitting
            across ``term_class_ids``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(loss, scores)`` where ``loss`` is the mean binary cross-entropy
            scalar tensor and ``scores`` has shape ``(B, n_terms)``.
        """
        _validate_int_ge("max_elements", max_elements, 1)
        pos = self._encode_features(features)
        bsz = int(pos.shape[0])
        term_ids = self.module._to_index_tensor(
            term_class_ids, self.nb_classes, "term_class_ids"
        )
        if term_ids.ndim != 1 or term_ids.numel() == 0:
            raise ValueError(
                f"term_class_ids must be a non-empty 1-D sequence/tensor, got shape {tuple(term_ids.shape)}"
            )
        n_terms = int(term_ids.numel())
        if not isinstance(label_matrix, th.Tensor):
            raise TypeError(
                f"label_matrix must be a torch.Tensor, got {type(label_matrix).__name__}"
            )
        if label_matrix.shape != (bsz, n_terms):
            raise ValueError(
                f"label_matrix must have shape ({bsz}, {n_terms}), got {tuple(label_matrix.shape)}"
            )
        targets = label_matrix.to(device=self.device, dtype=pos.dtype)
        scores = self._chunked_term_scores(pos, term_ids, max_elements=max_elements)
        bce = F.binary_cross_entropy(scores, targets, reduction="mean")
        return bce, scores

    @th.no_grad()
    def score(
        self,
        features: th.Tensor,
        term_class_ids: Union[Sequence[int], th.Tensor],
        max_elements: int = 8_388_608,
    ) -> th.Tensor:
        """Compute inductive prediction scores ``sigmoid(depth(encoder(features), term_class_ids, crisp=True) / tau_s)`` without gradients.

        Data elements never get a dead score (:math:`\\delta \\equiv 0`) or a
        minimality penalty on their positions: there are no data elements in
        :math:`\\Delta` here; proteins/data instances exist only through
        ``self.encoder``.

        Guarantees:
        - Runs under :func:`torch.no_grad` and returns a tensor of shape
          ``(B, n_terms)`` (or ``(1, n_terms)`` when ``features`` is 1-D) with
          values in ``(0, 1)``.
        - Chunks over ``term_class_ids`` when ``B * n_terms * K * d > max_elements``.

        Non-guarantees:
        - Does not toggle ``self.module.eval()``; callers using dropout/batchnorm in
          ``self.encoder`` should set ``model.module.eval()`` during evaluation.

        Parameters
        ----------
        features : torch.Tensor
            Input feature matrix of shape ``(B, feat_dim)`` (or 1-D ``(feat_dim,)``).
        term_class_ids : Union[Sequence[int], torch.Tensor]
            1-D sequence or tensor of ``n_terms`` class indices in ``[0, nb_classes)``.
        max_elements : int, default 8_388_608
            Maximum ``B * term_chunk * K * d`` elements per chunk.

        Returns
        -------
        torch.Tensor
            Predicted score matrix of shape ``(B, n_terms)``.
        """
        _validate_int_ge("max_elements", max_elements, 1)
        pos = self._encode_features(features)
        term_ids = self.module._to_index_tensor(
            term_class_ids, self.nb_classes, "term_class_ids"
        )
        if term_ids.ndim != 1 or term_ids.numel() == 0:
            raise ValueError(
                f"term_class_ids must be a non-empty 1-D sequence/tensor, got shape {tuple(term_ids.shape)}"
            )
        return self._chunked_term_scores(pos, term_ids, max_elements=max_elements)

    def compute_batch_losses(
        self,
        generator: Optional[th.Generator] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Dict[str, th.Tensor]:
        """Compute one step of per-term GroundedEL TBox/ABox losses (§3 and §4).

        Can be used directly as the TBox step of an external training loop (such as
        joint inductive protein + ontology training in DeepGO-SE / E3):

        - **Generator state:** Pass ``generator=model.generator`` (or omit
          ``generator`` to use ``self.generator`` automatically, which is seeded
          with ``self.seed + 1337`` at construction and can be reset via
          :meth:`reset_generator`). Consecutive calls advance ``self.generator`` so
          each step samples a fresh TBox mini-batch and random/adversarial points.
        - **Temperature schedule state:** Either pass ``step`` and ``total_steps``
          directly to :meth:`compute_batch_losses` (which calls
          ``self.step_tau(step, total_steps)`` before evaluating memberships) or
          call :meth:`step_tau` / ``self.module.set_tau(tau)`` once per epoch/step.
        - **Post-step deadness projection:** After ``optimizer.step()``, call
          :meth:`clamp_dead_logits` to project ``self.module.dead_logits`` onto
          ``[0, 1]`` and keep ``dead_logits[bot_id] = +100.0``.

        Guarantees:
        - Returns a dictionary containing ``"fact"``, ``"axioms"``, ``"region"``,
          ``"minimality"``, and ``"total"`` scalar tensors on ``self.device``.
        - Uses ``generator`` when provided, or ``self.generator`` when ``None``,
          for all random element and region draws.
        - When ``step`` and ``total_steps`` are provided, updates ``self.module.tau``
          via :meth:`step_tau` prior to loss evaluation.

        Non-guarantees:
        - When ``n_region_samples == 0`` or ``lambda_reg == 0``, ``"region"`` is
          exact ``0.0``.

        Parameters
        ----------
        generator : Optional[torch.Generator], default None
            Explicit PyTorch RNG on CPU. Defaults to ``self.generator`` when ``None``.
        step : Optional[int], default None
            Optional 0-based schedule step index ``0 <= step < total_steps`` for
            geometric :math:`\\tau` annealing from ``tau0`` to ``tau1``.
        total_steps : Optional[int], default None
            Optional total number of schedule steps :math:`\\ge 1` for :math:`\\tau`
            annealing. Must be provided if and only if ``step`` is provided.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of scalar loss tensors.
        """
        if (step is None) != (total_steps is None):
            raise ValueError(
                "step and total_steps must both be provided or both be None"
            )
        if step is not None and total_steps is not None:
            self.step_tau(step, total_steps)
        if generator is None:
            generator = self.generator
        self._ensure_role_tables()

        mod = self.module
        device = self.device
        dtype = mod.class_centers.dtype
        td = self.training_datasets

        pos_all = mod.positions()  # (nb_elements, d)
        dead_raw = mod.dead()
        if self.bot_id is not None:
            dead_all = dead_raw.clone()
            dead_all[self.bot_id] = 1.0
        else:
            dead_all = dead_raw
        # Crisp silencer gate for box/position hinges: live elements
        # pull boxes to full containment (m = 1); dead elements are silenced.
        dead_thresh = 0.75 if self.dead_from_bot_only else 0.5
        dead_gate = (dead_all.detach() > dead_thresh).to(dtype=dtype)

        # 1. Fact (existence) loss on class elements
        live_cls_list = [
            c for c in range(self.nb_classes) if c != self.bot_id
        ]
        if live_cls_list:
            cls_ids = th.tensor(live_cls_list, dtype=th.long, device=device)
            if cls_ids.numel() > self.batch_size:
                cls_ids = self._sample_rows(cls_ids.unsqueeze(1), generator).squeeze(1)
            m_self = mod.membership(pos_all[cls_ids], cls_ids)
            loss_fact = fact(m_self, dead=dead_gate[cls_ids]).mean()
        else:
            loss_fact = th.zeros((), dtype=dtype, device=device)

        axiom_terms: List[th.Tensor] = []
        region_terms: List[th.Tensor] = []
        do_region = self.lambda_reg > 0.0 and self.n_region_samples > 0

        # 2. GCI0: A <= B
        if "gci0" in td and len(td["gci0"]) > 0:
            rows = self._sample_rows(td["gci0"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows.reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, 0], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 0],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = mod.membership(px, a_flat)
            mb = mod.membership(px, b_flat)
            if self.dead_from_bot_only:
                is_self = x_flat == a_flat
                in_b_crisp = mod.depth(px, b_flat, crisp=True) > 0.0
                ma_box = th.where(is_self, th.ones_like(ma), ma)
                mb_box = th.where(in_b_crisp, th.ones_like(mb), mb)
                h_box = gci0(ma_box, mb_box, dead=dead_gate[x_flat])
                a_col = rows[:, 0]
                b_col = rows[:, 1]
                d_pull = (1.0 - dead_gate[a_col]) * gci0_region(
                    mod.depth(pos_all[a_col], b_col), margin=self.region_margin
                )
                axiom_terms.append(h_box.mean() + d_pull.mean())
            else:
                h_box = gci0(ma, mb, dead=dead_gate[x_flat])
                is_self = x_flat == a_flat
                is_self_dead = is_self & (dead_gate[x_flat] > 0.5)
                crisp_viol_0 = (
                    is_self
                    & ((mod.depth(px, a_flat, crisp=True) > 0.0) | is_self_dead)
                    & (mod.depth(px, b_flat, crisp=True) <= 0.0)
                )
                ma_d = th.where(is_self_dead, th.ones_like(ma), ma.detach())
                dead_eff_0 = th.where(
                    crisp_viol_0, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = gci0(ma_d, mb.detach(), dead=dead_eff_0)
                axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: gci0_region(
                            mod.depth(z, rows[:, 1].repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod,
                    rows[:, 0],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts_raw = reg.points[reg.valid]
                    pts = pts_raw if self.region_lhs_grad else pts_raw.detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    live_a = 1.0 - dead_gate[av]
                    dead_a = dead_gate[av]
                    reg_h = live_a * gci0_region(
                        mod.depth(pts, bv), margin=self.region_margin
                    ) + dead_a * gci0_bot_region(
                        mod.depth(pts_raw.detach(), av), margin=self.region_margin
                    )
                    region_terms.append(reg_h.mean())

        # 3. GCI0_bot: A <= bot
        if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
            rows = self._sample_rows(td["gci0_bot"][:], generator)
            a_col = rows[:, 0]
            bsz = int(a_col.shape[0])
            cand = self._batch_candidate_elements(a_col, generator)
            in_el = self._inside_candidate_elements(a_col, generator, pos_all)
            x_mat = sample_elements(
                a_col,
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = a_col.unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = mod.membership(px, a_flat)
            if self.dead_from_bot_only:
                crisp_0b = (
                    (x_flat == a_flat)
                    | (mod.depth(px, a_flat, crisp=True) > 0.0)
                    | (dead_all[x_flat].detach() > 0.5)
                )
                dead_box_0b = th.where(
                    x_flat == a_flat, dead_gate[x_flat], th.ones_like(dead_gate[x_flat])
                )
                h_box = gci0_bot(ma, dead=dead_box_0b)
                ma_d = th.where(crisp_0b, th.ones_like(ma), ma.detach())
                dead_eff_0b = th.where(
                    crisp_0b, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = gci0_bot(ma_d, dead=dead_eff_0b)
                axiom_terms.append((h_box + h_dead).mean())
            else:
                h_box = gci0_bot(ma, dead=dead_gate[x_flat])
                ma_d = th.where(x_flat == a_flat, th.ones_like(ma), ma.detach())
                h_dead = gci0_bot(ma_d, dead=dead_all[x_flat])
                axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                a_reg = a_col.repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: gci0_bot_region(
                            mod.depth(z, a_col.repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod,
                    a_col,
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    region_terms.append(
                        gci0_bot_region(
                            mod.depth(pts, av), margin=self.region_margin
                        ).mean()
                    )

        # 4. GCI1: A cap B <= C
        if "gci1" in td and len(td["gci1"]) > 0:
            rows = self._sample_rows(td["gci1"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows.reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, :2], generator, pos_all)
            x_mat = sample_elements(
                rows[:, :2],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            c_flat = rows[:, 2].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = mod.membership(px, a_flat)
            mb = mod.membership(px, b_flat)
            mc = mod.membership(px, c_flat)
            axiom_terms.append(
                gci1(ma, mb, mc, dead=dead_gate[x_flat]).mean()
            )
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                c_reg = rows[:, 2].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: gci1_region(
                            mod.depth(z, rows[:, 2].repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_pair_region(
                    mod,
                    rows[:, 0],
                    rows[:, 1],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts_raw = reg.points[reg.valid]
                    pts = pts_raw if self.region_lhs_grad else pts_raw.detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    cv = c_reg[reg.valid]
                    live_ab = (1.0 - dead_gate[av]) * (1.0 - dead_gate[bv])
                    region_terms.append(
                        (
                            live_ab
                            * gci1_region(
                                mod.depth(pts, cv), margin=self.region_margin
                            )
                        ).mean()
                    )

        # 5. GCI1_bot: A cap B <= bot
        if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
            rows = self._sample_rows(td["gci1_bot"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, :2].reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, :2], generator, pos_all)
            x_mat = sample_elements(
                rows[:, :2],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = mod.membership(px, a_flat)
            mb = mod.membership(px, b_flat)
            both_crisp = (mod.depth(px, a_flat, crisp=True) > 0.0) & (
                mod.depth(px, b_flat, crisp=True) > 0.0
            )
            if self.dead_from_bot_only:
                is_lhs_self = (x_flat == a_flat) | (x_flat == b_flat)
                dep_a_pos, ma_pos = self._depth_and_membership_pos_grad_only(px, a_flat)
                dep_b_pos, mb_pos = self._depth_and_membership_pos_grad_only(px, b_flat)
                ma_box = th.where(is_lhs_self, ma, ma_pos)
                mb_box = th.where(is_lhs_self, mb, mb_pos)
                d_push = (
                    0.5
                    * (1.0 - dead_gate[x_flat])
                    * F.relu(th.minimum(dep_a_pos, dep_b_pos))
                )
                h_box = gci1_bot(ma_box, mb_box, dead=dead_gate[x_flat]) + d_push
                a_col = rows[:, 0]
                b_col = rows[:, 1]
                d_sep = 0.5 * (
                    (1.0 - dead_gate[a_col]) * F.relu(mod.depth(pos_all[a_col], b_col))
                    + (1.0 - dead_gate[b_col]) * F.relu(mod.depth(pos_all[b_col], a_col))
                )
                dead_eff_1b = th.where(
                    both_crisp, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = gci1_bot(
                    th.ones_like(ma), th.ones_like(mb), dead=dead_eff_1b
                )
                axiom_terms.append((h_box + h_dead).mean() + d_sep.mean())
            else:
                h_box = gci1_bot(ma, mb, dead=dead_gate[x_flat])
                dead_eff_1b = th.where(
                    both_crisp, dead_all[x_flat], th.ones_like(dead_all[x_flat])
                )
                h_dead = gci1_bot(ma.detach(), mb.detach(), dead=dead_eff_1b)
                axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: gci1_bot_region(
                            mod.depth(z, rows[:, 0].repeat(max(z.shape[0] // bsz, 1))),
                            mod.depth(z, rows[:, 1].repeat(max(z.shape[0] // bsz, 1))),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_pair_region(
                    mod,
                    rows[:, 0],
                    rows[:, 1],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    region_terms.append(
                        gci1_bot_region(
                            mod.depth(pts, av),
                            mod.depth(pts, bv),
                            margin=self.region_margin,
                        ).mean()
                    )

        # 6. GCI2: A <= exists r.B
        if "gci2" in td and len(td["gci2"]) > 0:
            rows = self._sample_rows(td["gci2"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, [0, 2]].reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, 0], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 0],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            r_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 2].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            pb = pos_all[b_flat]
            ma = mod.membership(px, a_flat)
            rho_xb = mod.role_membership(px, pb, r_flat)
            df_b = dead_all[b_flat].detach()
            if self.dead_from_bot_only:
                crisp_2d = (
                    (x_flat == a_flat)
                    | (mod.depth(px, a_flat, crisp=True) > 0.0)
                    | (dead_all[x_flat].detach() > 0.5)
                )
                ma_d = th.where(crisp_2d, th.ones_like(ma), ma.detach())
            else:
                is_self_dead = (x_flat == a_flat) & (dead_gate[x_flat] > 0.5)
                ma_d = th.where(is_self_dead, th.ones_like(ma), ma.detach())
            edge_h = gci2(ma, rho_xb, dead=dead_gate[x_flat], dead_filler=None)
            dead_prop_h = gci2_dead(
                ma, dead_filler=df_b, dead=dead_gate[x_flat]
            ) + th.where(
                df_b > 0.5,
                gci2_dead(ma_d, dead_filler=df_b, dead=dead_all[x_flat]),
                th.zeros_like(ma),
            )
            axiom_terms.append((edge_h + dead_prop_h).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                r_reg = rows[:, 1].repeat(self.n_region_samples)
                b_reg = rows[:, 2].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: gci2_region(
                            mod.role_depth(
                                z,
                                pos_all[
                                    rows[:, 2].repeat(max(z.shape[0] // bsz, 1))
                                ].detach(),
                                rows[:, 1].repeat(max(z.shape[0] // bsz, 1)),
                            ),
                            margin=self.region_margin,
                            agg=self.agg,
                            agg_tau=self.agg_tau,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod,
                    rows[:, 0],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn,
                )
                if bool(reg.valid.any().item()):
                    pts_raw = reg.points[reg.valid]
                    pts = pts_raw if self.region_lhs_grad else pts_raw.detach()
                    av = a_reg[reg.valid]
                    rv = r_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    live_a = 1.0 - dead_gate[av]
                    dead_a = dead_gate[av]
                    reg_h = live_a * gci2_region(
                        mod.role_depth(pts, pos_all[bv], rv),
                        margin=self.region_margin,
                        agg=self.agg,
                        agg_tau=self.agg_tau,
                        depth_a=mod.depth(pts_raw.detach(), av),
                        dead_filler=dead_all[bv].detach(),
                    ) + dead_a * gci0_bot_region(
                        mod.depth(pts_raw.detach(), av), margin=self.region_margin
                    )
                    region_terms.append(reg_h.mean())

        # 7. GCI3: exists r.B <= C
        if "gci3" in td and len(td["gci3"]) > 0:
            rows = self._sample_rows(td["gci3"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, 1:].reshape(-1), generator)
            in_el = self._inside_candidate_elements(rows[:, 2], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 2],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            y_mat = self._sample_batch_y_fillers(rows, generator)
            sx = int(x_mat.shape[1])
            sy = int(y_mat.shape[1])
            rho_xy, _ = self._grid_role_membership(
                pos_all, x_mat, y_mat, rows[:, 0], need_crisp=False
            )
            b_2d = rows[:, 1].unsqueeze(1).expand(bsz, sy).reshape(-1)
            c_2d = rows[:, 2].unsqueeze(1).expand(bsz, sx).reshape(-1)
            mb_grid = (
                mod.membership(pos_all[y_mat.reshape(-1)], b_2d)
                .view(bsz, 1, sy)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            mc_grid = (
                mod.membership(pos_all[x_mat.reshape(-1)], c_2d)
                .view(bsz, sx, 1)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            dead_x_grid = (
                dead_gate[x_mat].unsqueeze(2).expand(bsz, sx, sy).reshape(-1)
            )
            axiom_terms.append(
                gci3(rho_xy, mb_grid, mc_grid, dead=dead_x_grid).mean()
            )
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                c_reg = rows[:, 2].repeat(self.n_region_samples)
                h_fn_3 = (
                    (
                        lambda xy: gci3_region(
                            mod.role_depth(
                                xy[:, : self.embed_dim],
                                xy[:, self.embed_dim :],
                                rows[:, 0].repeat(max(xy.shape[0] // bsz, 1)),
                            ),
                            mod.depth(
                                xy[:, self.embed_dim :],
                                rows[:, 1].repeat(max(xy.shape[0] // bsz, 1)),
                            ),
                            mod.depth(
                                xy[:, : self.embed_dim],
                                rows[:, 2].repeat(max(xy.shape[0] // bsz, 1)),
                            ),
                            margin=self.region_margin,
                            smooth=True,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                h_samp, t_samp = sample_role_region(
                    mod,
                    rows[:, 0],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn_3,
                )
                if bool(h_samp.valid.any().item()):
                    zx_raw = h_samp.points[h_samp.valid]
                    zy_raw = t_samp.points[t_samp.valid]
                    zx = zx_raw if self.region_lhs_grad else zx_raw.detach()
                    zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                    rv = r_reg[h_samp.valid]
                    bv = b_reg[h_samp.valid]
                    cv = c_reg[h_samp.valid]
                    region_terms.append(
                        gci3_region(
                            mod.role_depth(zx_raw.detach(), zy_raw.detach(), rv),
                            mod.depth(zy, bv),
                            mod.depth(zx, cv),
                            margin=self.region_margin,
                        ).mean()
                    )
                    # Also evaluate at z = zx with y = filler element x_B (§4)
                    py_fill = pos_all[bv]
                    region_terms.append(
                        gci3_region(
                            mod.role_depth(zx_raw.detach(), py_fill, rv),
                            mod.depth(py_fill, bv),
                            mod.depth(zx, cv),
                            margin=self.region_margin,
                        ).mean()
                    )

        # 8. GCI3_bot: exists r.B <= bot
        if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
            rows = self._sample_rows(td["gci3_bot"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, 1], generator)
            in_el = self._inside_candidate_elements(rows[:, 1], generator, pos_all)
            x_mat = sample_elements(
                rows[:, 1],
                self.nb_elements,
                self.n_random_elements,
                generator,
                batch_elements=cand,
                inside_elements=in_el,
            )
            y_mat = self._sample_batch_y_fillers(rows, generator)
            sx = int(x_mat.shape[1])
            sy = int(y_mat.shape[1])
            rho_xy, r_dep_crisp = self._grid_role_membership(
                pos_all, x_mat, y_mat, rows[:, 0], need_crisp=True
            )
            assert r_dep_crisp is not None
            b_2d = rows[:, 1].unsqueeze(1).expand(bsz, sy).reshape(-1)
            py_flat = pos_all[y_mat.reshape(-1)]
            mb_grid = (
                mod.membership(py_flat, b_2d)
                .view(bsz, 1, sy)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            b_crisp_grid = (
                (mod.depth(py_flat, b_2d, crisp=True) > 0.0)
                .view(bsz, 1, sy)
                .expand(bsz, sx, sy)
                .reshape(-1)
            )
            both_crisp = (r_dep_crisp > 0.0) & b_crisp_grid
            x_grid = x_mat.unsqueeze(2).expand(bsz, sx, sy).reshape(-1)
            h_box = gci3_bot(rho_xy, mb_grid, dead=dead_gate[x_grid])
            if self.dead_from_bot_only:
                dead_eff_3b = th.where(
                    both_crisp, dead_all[x_grid], th.ones_like(dead_all[x_grid])
                )
                h_dead = gci3_bot(
                    th.ones_like(rho_xy), th.ones_like(mb_grid), dead=dead_eff_3b
                )
            else:
                dead_eff_3b = th.where(
                    both_crisp, dead_all[x_grid], th.ones_like(dead_all[x_grid])
                )
                h_dead = gci3_bot(
                    rho_xy.detach(), mb_grid.detach(), dead=dead_eff_3b
                )
            axiom_terms.append((h_box + h_dead).mean())
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn_3b = (
                    (
                        lambda xy: gci3_bot_region(
                            mod.role_depth(
                                xy[:, : self.embed_dim],
                                xy[:, self.embed_dim :],
                                rows[:, 0].repeat(max(xy.shape[0] // bsz, 1)),
                            ),
                            mod.depth(
                                xy[:, self.embed_dim :],
                                rows[:, 1].repeat(max(xy.shape[0] // bsz, 1)),
                            ),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                h_samp, t_samp = sample_role_region(
                    mod,
                    rows[:, 0],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn_3b,
                )
                if bool(h_samp.valid.any().item()):
                    zx_raw = h_samp.points[h_samp.valid]
                    zy_raw = t_samp.points[t_samp.valid]
                    zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                    rv = r_reg[h_samp.valid]
                    bv = b_reg[h_samp.valid]
                    region_terms.append(
                        gci3_bot_region(
                            mod.role_depth(zx_raw.detach(), zy_raw.detach(), rv),
                            mod.depth(zy, bv),
                            margin=self.region_margin,
                        ).mean()
                    )

        # 9. Role inclusion: r <= s
        if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
            rows = self._sample_rows(td["role_inclusion"][:], generator)
            bsz = int(rows.shape[0])
            all_el = self._batch_candidate_elements(
                th.arange(min(self.nb_elements, 16), device=device), generator
            )
            sx = int(all_el.numel())
            x_grid = all_el.view(1, sx, 1).expand(bsz, sx, sx).reshape(-1)
            y_grid = all_el.view(1, 1, sx).expand(bsz, sx, sx).reshape(-1)
            r_grid = rows[:, 0].view(bsz, 1, 1).expand(bsz, sx, sx).reshape(-1)
            s_grid = rows[:, 1].view(bsz, 1, 1).expand(bsz, sx, sx).reshape(-1)
            px = pos_all[x_grid]
            py = pos_all[y_grid]
            axiom_terms.append(
                role_inclusion(
                    mod.role_membership(px, py, r_grid),
                    mod.role_membership(px, py, s_grid),
                ).mean()
            )
            if do_region:
                s_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn_ri = (
                    (
                        lambda xy: role_inclusion_region(
                            mod.role_depth(
                                xy[:, : self.embed_dim],
                                xy[:, self.embed_dim :],
                                rows[:, 1].repeat(max(xy.shape[0] // bsz, 1)),
                            ),
                            margin=self.region_margin,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                h_samp, t_samp = sample_role_region(
                    mod,
                    rows[:, 0],
                    generator=generator,
                    n_samples=self.n_region_samples,
                    n_adv=self.n_adv,
                    hinge_fn=h_fn_ri,
                )
                if bool(h_samp.valid.any().item()):
                    zx_raw = h_samp.points[h_samp.valid]
                    zy_raw = t_samp.points[t_samp.valid]
                    zx = zx_raw if self.region_lhs_grad else zx_raw.detach()
                    zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                    sv = s_reg[h_samp.valid]
                    region_terms.append(
                        role_inclusion_region(
                            mod.role_depth(zx, zy, sv),
                            margin=self.region_margin,
                        ).mean()
                    )

        # 10. Role chain: r o s <= t
        if "role_chain" in td and len(td["role_chain"]) > 0:
            rows = self._sample_rows(td["role_chain"][:], generator)
            bsz = int(rows.shape[0])
            all_el = self._batch_candidate_elements(
                th.arange(min(self.nb_elements, 12), device=device), generator
            )
            if all_el.numel() > 10:
                all_el = all_el[:10]
            sx = int(all_el.numel())
            x_g = all_el.view(1, sx, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            y_g = all_el.view(1, 1, sx, 1).expand(bsz, sx, sx, sx).reshape(-1)
            z_g = all_el.view(1, 1, 1, sx).expand(bsz, sx, sx, sx).reshape(-1)
            r_g = rows[:, 0].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            s_g = rows[:, 1].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            t_g = rows[:, 2].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            px = pos_all[x_g]
            py = pos_all[y_g]
            pz = pos_all[z_g]
            axiom_terms.append(
                role_chain(
                    mod.role_membership(px, py, r_g),
                    mod.role_membership(py, pz, s_g),
                    mod.role_membership(px, pz, t_g),
                ).mean()
            )
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                s_reg = rows[:, 1].repeat(self.n_region_samples)
                t_reg = rows[:, 2].repeat(self.n_region_samples)
                hx, hy = sample_role_region(
                    mod, rows[:, 0], generator=generator, n_samples=self.n_region_samples, n_adv=0
                )
                _, tz = sample_role_region(
                    mod, rows[:, 1], generator=generator, n_samples=self.n_region_samples, n_adv=0
                )
                v_chain = hx.valid & tz.valid
                if bool(v_chain.any().item()):
                    zx_raw = hx.points[v_chain]
                    zy_raw = hy.points[v_chain]
                    zz_raw = tz.points[v_chain]
                    zx = zx_raw if self.region_lhs_grad else zx_raw.detach()
                    zy = zy_raw if self.region_lhs_grad else zy_raw.detach()
                    zz = zz_raw if self.region_lhs_grad else zz_raw.detach()
                    rv = r_reg[v_chain]
                    sv = s_reg[v_chain]
                    tv = t_reg[v_chain]
                    region_terms.append(
                        role_chain_region(
                            mod.role_depth(zx_raw.detach(), zy_raw.detach(), rv),
                            mod.role_depth(zy, zz_raw.detach(), sv),
                            mod.role_depth(zx, zz, tv),
                            margin=self.region_margin,
                        ).mean()
                    )

        # 11. Class assertion: A(a)
        if "class_assertion" in td and len(td["class_assertion"]) > 0:
            rows = self._sample_rows(td["class_assertion"][:], generator)
            el_a = self.nb_classes + rows[:, 0]
            c_id = rows[:, 1]
            axiom_terms.append(
                class_assertion(mod.membership(pos_all[el_a], c_id)).mean()
            )

        # 12. Object property assertion: r(a, b)
        if (
            "object_property_assertion" in td
            and len(td["object_property_assertion"]) > 0
        ):
            rows = self._sample_rows(td["object_property_assertion"][:], generator)
            el_a = self.nb_classes + rows[:, 0]
            r_id = rows[:, 1]
            el_b = self.nb_classes + rows[:, 2]
            axiom_terms.append(
                role_assertion(
                    mod.role_membership(pos_all[el_a], pos_all[el_b], r_id)
                ).mean()
            )

        loss_axioms = (
            th.stack(axiom_terms).sum()
            if axiom_terms
            else th.zeros((), dtype=dtype, device=device)
        )
        loss_region = (
            th.stack(region_terms).sum()
            if region_terms
            else th.zeros((), dtype=dtype, device=device)
        )

        # 13. Minimality (principled negatives)
        ord_cls_list = [
            c
            for c in range(self.nb_classes)
            if c != self.top_id and c != self.bot_id
        ]
        if ord_cls_list and self.eps_min > 0.0:
            ord_cls = th.tensor(ord_cls_list, dtype=th.long, device=device)
            if self.nb_elements * ord_cls.numel() <= 1024:
                all_el = th.arange(self.nb_elements, dtype=th.long, device=device)
                gx = all_el.unsqueeze(1).expand(self.nb_elements, ord_cls.numel()).reshape(-1)
                ga = ord_cls.unsqueeze(0).expand(self.nb_elements, ord_cls.numel()).reshape(-1)
            else:
                n_min = min(self.batch_size, max(256, self.nb_elements * self.n_random_elements))
                gx = th.randint(
                    0, self.nb_elements, (n_min,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ga_idx = th.randint(
                    0, ord_cls.numel(), (n_min,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ga = ord_cls[ga_idx]
            m_rand = mod.membership(pos_all[gx], ga)
        else:
            m_rand = None

        if self.nb_rels > 0 and self.eps_role > 0.0:
            if (self.nb_elements ** 2) * self.nb_rels <= 1024:
                all_el = th.arange(self.nb_elements, dtype=th.long, device=device)
                all_r = th.arange(self.nb_rels, dtype=th.long, device=device)
                rx = all_el.view(self.nb_elements, 1, 1).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
                ry = all_el.view(1, self.nb_elements, 1).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
                rr = all_r.view(1, 1, self.nb_rels).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
            else:
                n_rmin = min(self.batch_size, max(256, self.nb_elements * self.n_random_elements))
                rx = th.randint(
                    0, self.nb_elements, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ry = th.randint(
                    0, self.nb_elements, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                rr = th.randint(
                    0, self.nb_rels, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
            rho_rand = mod.role_membership(pos_all[rx], pos_all[ry], rr)
        else:
            rho_rand = None

        dead_min_ids = [
            e for e in range(self.nb_elements) if e != self.bot_id
        ]
        dead_for_min = (
            dead_all[th.tensor(dead_min_ids, dtype=th.long, device=device)]
            if dead_min_ids
            else None
        )
        loss_min = minimality(
            m_rand=m_rand,
            rho_rand=rho_rand,
            dead=dead_for_min,
            eps=self.eps_min,
            eps_r=self.eps_role,
            eps_dead=self.eps_dead,
        )

        total = (
            loss_fact
            + loss_axioms
            + self.lambda_reg * loss_region
            + loss_min
        )
        return {
            "fact": loss_fact,
            "axioms": loss_axioms,
            "region": loss_region,
            "minimality": loss_min,
            "total": total,
        }

    @staticmethod
    def _class_slot_live_counts_chunked(
        cen_c: th.Tensor,
        w_c: th.Tensor,
        act_c: th.Tensor,
        live_pos: th.Tensor,
        chunk_size: int = 8192,
    ) -> th.Tensor:
        """Compute number of live elements crisp-inside each of ``K`` components of a single class in chunks of 8192 elements."""
        k_slots = int(cen_c.shape[0])
        counts = th.zeros((k_slots,), dtype=th.long, device=cen_c.device)
        n_live = int(live_pos.shape[0])
        if n_live == 0 or not bool(act_c.any().item()):
            return counts
        for start in range(0, n_live, chunk_size):
            end = min(start + chunk_size, n_live)
            p_chunk = live_pos[start:end]  # (C_len, d), C_len <= 8192
            c_dep = th.amin(
                w_c.unsqueeze(0) - th.abs(p_chunk.unsqueeze(1) - cen_c.unsqueeze(0)),
                dim=-1,
            )  # (C_len, K)
            inside = (c_dep > 0.0) & act_c.unsqueeze(0)
            counts += inside.sum(dim=0)
        return counts

    def _compute_slot_live_counts(
        self,
        regions: Regions,
        live_pos: th.Tensor,
        query_slot_mask: th.Tensor,
        chunk_size: int = 8192,
    ) -> th.Tensor:
        """Compute exact count of live elements crisp-inside each queried ``(c_id, k)`` slot using 8192-element chunks."""
        device = regions.cen.device
        k_slots = int(regions.cen.shape[1])
        counts = th.zeros((self.nb_classes, k_slots), dtype=th.long, device=device)
        n_live = int(live_pos.shape[0])
        if n_live == 0 or not bool(query_slot_mask.any().item()):
            return counts

        if self.nb_classes <= 2048 or n_live <= 2048:
            query_classes = (
                th.nonzero(query_slot_mask.any(dim=1), as_tuple=False).view(-1).tolist()
            )
            for c_id in query_classes:
                counts[c_id] = self._class_slot_live_counts_chunked(
                    regions.cen[c_id],
                    regions.w[c_id],
                    regions.active[c_id] & query_slot_mask[c_id],
                    live_pos,
                    chunk_size=chunk_size,
                )
            return counts

        # Lossless 4^D_BINS spatial-bin prefilter + exact per-box depth in <= 8192-element chunks
        d_bins = min(self.embed_dim, 5)
        n_bins = 4**d_bins
        p_min = live_pos[:, :d_bins].amin(dim=0)
        p_max = live_pos[:, :d_bins].amax(dim=0)
        scale = 3.999 / (p_max - p_min + 1e-6)
        elem_bins = th.clamp(
            ((live_pos[:, :d_bins] - p_min) * scale).to(th.long), 0, 3
        )
        mult = (
            4 ** th.arange(d_bins, dtype=th.long, device=device)
        ).unsqueeze(0)
        elem_hash = (elem_bins * mult).sum(dim=1)
        order = th.argsort(elem_hash)
        sorted_pos = live_pos[order]
        counts_bin = th.bincount(elem_hash, minlength=n_bins)
        offsets = th.zeros((n_bins + 1,), dtype=th.long, device=device)
        offsets[1:] = th.cumsum(counts_bin, dim=0)

        q_flat_ids = th.nonzero(query_slot_mask.reshape(-1), as_tuple=False).view(-1)
        cen_q = regions.cen.reshape(-1, self.embed_dim)[q_flat_ids]
        w_q = regions.w.reshape(-1, self.embed_dim)[q_flat_ids]
        lo_b = th.clamp(
            ((cen_q[:, :d_bins] - w_q[:, :d_bins] - 1e-6 - p_min) * scale).to(th.long),
            0,
            3,
        )
        hi_b = th.clamp(
            ((cen_q[:, :d_bins] + w_q[:, :d_bins] + 1e-6 - p_min) * scale).to(th.long),
            0,
            3,
        )
        q_counts = th.zeros((q_flat_ids.numel(),), dtype=th.long, device=device)

        for h in range(n_bins):
            s_i = int(offsets[h].item())
            e_i = int(offsets[h + 1].item())
            if s_i == e_i:
                continue
            rem = h
            mask = th.ones((q_flat_ids.numel(),), dtype=th.bool, device=device)
            for m_i in range(d_bins):
                b_m = rem & 3
                rem >>= 2
                mask &= (lo_b[:, m_i] <= b_m) & (hi_b[:, m_i] >= b_m)
            b_ids = th.nonzero(mask, as_tuple=False).view(-1)
            if b_ids.numel() == 0:
                continue
            for e_s in range(s_i, e_i, chunk_size):
                e_e = min(e_s + chunk_size, e_i)
                p_chunk = sorted_pos[e_s:e_e]  # (<= 8192, d)
                for b_s in range(0, b_ids.numel(), 256):
                    b_e = min(b_s + 256, b_ids.numel())
                    b_sub = b_ids[b_s:b_e]
                    c_sub = cen_q[b_sub]
                    w_sub = w_q[b_sub]
                    dep = th.amin(
                        w_sub.unsqueeze(1)
                        - th.abs(p_chunk.unsqueeze(0) - c_sub.unsqueeze(1)),
                        dim=-1,
                    )
                    q_counts[b_sub] += (dep > 0.0).sum(dim=1)

        counts.view(-1)[q_flat_ids] = q_counts
        return counts

    def reseed_idle(
        self,
        generator: Optional[th.Generator] = None,
        reseed_size: float = 0.08,
        prune_unused: bool = True,
    ) -> int:
        """Relocate idle/least-used class components to uncovered own elements and prune empty slots (§5).

        Performs ONLY:
        1. For each live class :math:`A` whose own element :math:`x_A` is not
           crisp-inside :math:`A`: places one component box of half-width
           ``reseed_size`` at :math:`p_{x_A}`, using an inactive slot if any,
           else the active slot that crisp-contains the fewest live elements
           (breaking ties by smallest volume); same for ABox class assertions
           :math:`C(a)` (element of individual :math:`a`).
        2. When ``prune_unused=True``: deactivates active components that
           crisp-contain no live element (inspecting element positions only).

        Guarantees:
        - Reasoner-free (R1): never reads ``gci0``, ``gci1``, ``gci2``, ``gci3``,
          ``gci0_bot``, ``gci1_bot``, ``gci3_bot``, ``role_inclusion``,
          ``role_chain``, or ``object_property_assertion`` tables.
        - Never modifies ``elem_pos.weight`` or ``dead_logits``, and never
          rebuilds whole classes.
        - Computes per-class live element depths in chunks of at most 8192
          elements (:math:`O(|\\Delta| \\cdot K \\cdot d)` memory per chunk; no
          :math:`|\\Delta|^2` tensor anywhere).
        - Returns the total number of modified component slots.

        Non-guarantees:
        - Does not reset Adam optimiser moments directly; :meth:`train` calls
          ``optimizer.state.clear()`` after invoking :meth:`reseed_idle`.

        Parameters
        ----------
        generator : Optional[torch.Generator], default None
            Optional PyTorch RNG (retained for signature compatibility).
        reseed_size : float, default 0.08
            Positive half-width assigned to newly re-seeded component boxes.
        prune_unused : bool, default True
            Whether to deactivate active component slots that crisp-contain zero
            live elements.

        Returns
        -------
        int
            Number of modified component slots.
        """
        reseed_size_f = _validate_positive_float("reseed_size", reseed_size)
        raw_w_target = _inv_softplus(reseed_size_f)
        mod = self.module
        device = self.device
        k_slots = mod.n_components
        moves = 0

        with th.no_grad():
            pos_all = mod.positions()  # (nb_elements, d)
            dead_c = mod.dead() > 0.5
            if self.bot_id is not None:
                dead_c[self.bot_id] = True
            live_mask = ~dead_c
            live_pos = pos_all[live_mask]  # (L, d)

            regions = mod.class_regions()  # (nb_classes, K, d)
            cen = regions.cen
            w = regions.w
            act = regions.active  # (nb_classes, K)

            # Check whether each class's own element x_A is crisp-inside each slot of A
            cls_pos = pos_all[: self.nb_classes]  # (nb_classes, d)
            own_dep = th.amin(
                w - th.abs(cls_pos.unsqueeze(1) - cen), dim=-1
            )  # (nb_classes, K)
            own_in_slot = (
                (own_dep > 0.0) & act & live_mask[: self.nb_classes].unsqueeze(1)
            )  # (nb_classes, K)
            own_inside_cls = own_in_slot.any(dim=1)  # (nb_classes,)

            live_cls_mask = live_mask[: self.nb_classes].clone()
            if self.top_id is not None:
                live_cls_mask[self.top_id] = False
            if self.bot_id is not None:
                live_cls_mask[self.bot_id] = False

            need_own_mask = live_cls_mask & (~own_inside_cls)

            # Determine which active slots need exact live-element counts:
            # - classes in need_own_mask that have all K slots active (to choose fewest-element slot)
            # - classes in ABox class_assertion that may need re-seeding
            # - active slots not already containing x_A when prune_unused=True
            need_tiebreak_cls = need_own_mask & act.all(dim=1) & (k_slots > 1)
            query_slot_mask = need_tiebreak_cls.unsqueeze(1) & act

            ca_el_a = th.empty((0,), dtype=th.long, device=device)
            ca_c_id = th.empty((0,), dtype=th.long, device=device)
            if self.nb_inds > 0:
                td = self.training_datasets
                if "class_assertion" in td and len(td["class_assertion"]) > 0:
                    ca_rows = td["class_assertion"][:].to(device=device, dtype=th.long)
                    ca_el_a_all = self.nb_classes + ca_rows[:, 0]
                    ca_c_id_all = ca_rows[:, 1]
                    ca_valid = live_mask[ca_el_a_all]
                    if self.top_id is not None:
                        ca_valid = ca_valid & (ca_c_id_all != self.top_id)
                    if self.bot_id is not None:
                        ca_valid = ca_valid & (ca_c_id_all != self.bot_id)
                    ca_el_a = ca_el_a_all[ca_valid]
                    ca_c_id = ca_c_id_all[ca_valid]
                    if ca_c_id.numel() > 0 and k_slots > 1:
                        ca_dep_pre = mod.depth(pos_all[ca_el_a], ca_c_id, crisp=True)
                        ca_cand_c = ca_c_id[(ca_dep_pre <= 0.0) | need_own_mask[ca_c_id]]
                        if ca_cand_c.numel() > 0:
                            ca_cls_mask = th.zeros(
                                (self.nb_classes,), dtype=th.bool, device=device
                            )
                            ca_cls_mask[ca_cand_c] = True
                            query_slot_mask = query_slot_mask | (
                                (ca_cls_mask & act.all(dim=1)).unsqueeze(1) & act
                            )

            if prune_unused:
                prune_candidates = act & (~own_in_slot)
                if self.top_id is not None:
                    prune_candidates[self.top_id, :] = False
                if self.bot_id is not None:
                    prune_candidates[self.bot_id, :] = False
                query_slot_mask = query_slot_mask | prune_candidates

            slot_counts = self._compute_slot_live_counts(
                regions, live_pos, query_slot_mask, chunk_size=8192
            )
            # Ensure slots known to contain x_A have count >= 1
            slot_counts = th.where(
                own_in_slot & (slot_counts == 0),
                th.ones_like(slot_counts),
                slot_counts,
            )

            # Step (a.1): live classes A whose own element x_A is not crisp-inside A
            need_cls_ids = th.nonzero(need_own_mask, as_tuple=False).view(-1)
            if need_cls_ids.numel() > 0:
                w_need = w[need_cls_ids]  # (M, K, d)
                act_need = act[need_cls_ids]  # (M, K)
                cnt_need = slot_counts[need_cls_ids]  # (M, K)
                log_vol_need = th.where(
                    act_need,
                    th.log(th.clamp(w_need, min=1e-12)).sum(dim=-1),
                    th.full(act_need.shape, float("-inf"), dtype=w.dtype, device=device),
                )

                best_k = th.zeros((need_cls_ids.numel(),), dtype=th.long, device=device)
                best_act = act_need[:, 0]
                best_cnt = cnt_need[:, 0]
                best_lvol = log_vol_need[:, 0]
                for k in range(1, k_slots):
                    cur_act = act_need[:, k]
                    cur_cnt = cnt_need[:, k]
                    cur_lvol = log_vol_need[:, k]
                    better = (~cur_act & best_act) | (
                        (cur_act == best_act)
                        & (
                            (cur_cnt < best_cnt)
                            | ((cur_cnt == best_cnt) & (cur_lvol < best_lvol))
                        )
                    )
                    best_k = th.where(better, th.full_like(best_k, k), best_k)
                    best_act = th.where(better, cur_act, best_act)
                    best_cnt = th.where(better, cur_cnt, best_cnt)
                    best_lvol = th.where(better, cur_lvol, best_lvol)

                mod.class_centers[need_cls_ids, best_k] = pos_all[need_cls_ids]
                mod.class_raw_widths[need_cls_ids, best_k] = raw_w_target
                mod.class_active[need_cls_ids, best_k] = True
                # Newly placed box at live x_A crisp-contains at least x_A
                slot_counts[need_cls_ids, best_k] = th.clamp(
                    slot_counts[need_cls_ids, best_k], min=1
                )
                moves += int(need_cls_ids.numel())

            # Step (a.2): ABox class assertions C(a) (element of individual a)
            if ca_el_a.numel() > 0:
                ca_dep = mod.depth(pos_all[ca_el_a], ca_c_id, crisp=True)
                unsat_idx = th.nonzero(ca_dep <= 0.0, as_tuple=False).view(-1)
                if unsat_idx.numel() > 0:
                    unsat_el_list = ca_el_a[unsat_idx].detach().cpu().tolist()
                    unsat_c_list = ca_c_id[unsat_idx].detach().cpu().tolist()
                    for el_a, c_id in zip(unsat_el_list, unsat_c_list):
                        cen_c = mod.class_centers[c_id]  # (K, d)
                        raw_w_c = mod.class_raw_widths[c_id]  # (K, d)
                        act_c = mod.class_active[c_id]  # (K,)
                        w_pos_c = F.softplus(raw_w_c)
                        w_c = th.where(
                            act_c.unsqueeze(-1), w_pos_c, th.zeros_like(w_pos_c)
                        )
                        eff_act_c = act_c & (th.amin(w_c, dim=-1) > 0.0)
                        dep_a = th.amin(
                            w_c - th.abs(pos_all[el_a].unsqueeze(0) - cen_c), dim=-1
                        )
                        if bool(((dep_a > 0.0) & eff_act_c).any().item()):
                            continue
                        if bool((~eff_act_c).any().item()):
                            k_star = int(
                                th.nonzero(~eff_act_c, as_tuple=False)[0].item()
                            )
                        elif k_slots == 1:
                            k_star = 0
                        else:
                            cnt_c = slot_counts[c_id]
                            lvol_c = (
                                th.log(th.clamp(w_c, min=1e-12))
                                .sum(dim=-1)
                                .cpu()
                                .tolist()
                            )
                            cnt_list = cnt_c.cpu().tolist()
                            k_star = min(
                                range(k_slots),
                                key=lambda k: (int(cnt_list[k]), float(lvol_c[k])),
                            )
                        mod.class_centers[c_id, k_star].copy_(pos_all[el_a])
                        mod.class_raw_widths[c_id, k_star].fill_(raw_w_target)
                        mod.class_active[c_id, k_star] = True
                        slot_counts[c_id, k_star] = max(
                            int(slot_counts[c_id, k_star].item()), 1
                        )
                        moves += 1

            # Step (b): deactivate active components that crisp-contain no live element
            if prune_unused:
                cur_reg = mod.class_regions()
                unused_active = cur_reg.active & (slot_counts == 0)
                if self.top_id is not None:
                    unused_active[self.top_id, :] = False
                if self.bot_id is not None:
                    unused_active[self.bot_id, :] = False
                n_pruned = int(unused_active.sum().item())
                if n_pruned > 0:
                    mod.class_active[unused_active] = False
                    mod.class_raw_widths[unused_active] = _INACTIVE_RAW_WIDTH
                    moves += n_pruned

        mod.last_reseed_count = int(moves)
        return moves

    @property
    def eval_gci_name(self) -> Optional[str]:
        """Return the GCI / assertion name used for validation and evaluation."""
        return self._eval_gci_name

    @eval_gci_name.setter
    def eval_gci_name(self, value: str) -> None:
        valid_gci_names = ["gci0", "gci1", "gci2", "gci3", "class_assertion"]
        if self._extended:
            valid_gci_names.extend(["gci0_bot", "gci1_bot", "gci3_bot"])
        if value not in valid_gci_names:
            raise ValueError(
                f"eval_gci_name must be one of {valid_gci_names}, got {value!r}"
            )
        self._eval_gci_name = value

    @staticmethod
    def _scheduled_lr(
        epoch: int,
        epochs: int,
        lr: float,
        lr_final: Optional[float],
        lr_schedule: str = "cosine",
    ) -> float:
        """Compute per-epoch learning rate as a pure function of ``(epoch, epochs, lr, lr_final, lr_schedule)``.

        Guarantees:
        - Pure function of its arguments; independent of optimiser state or
          idle-component re-seeding.
        - Returns ``lr`` when ``lr_final is None``, when ``epoch == 0``, or when
          ``epochs <= 1``.
        - Returns ``lr_final`` (to floating-point precision) at
          ``epoch == epochs - 1`` when ``epochs > 1`` and ``lr_final is not None``.

        Non-guarantees:
        - Assumes caller has already validated ``lr_final`` and ``lr_schedule``.
        """
        if lr_final is None or epochs <= 1:
            return float(lr)
        t = epoch / (epochs - 1)
        if lr_schedule == "cosine":
            w = 0.5 * (1.0 + math.cos(math.pi * t))
            return float(w * lr + (1.0 - w) * lr_final)
        return float((1.0 - t) * lr + t * lr_final)

    def train(
        self,
        epochs: Optional[int] = None,
        validate_every: int = 1,
        epoch_callback: Optional[Callable[[int, "GroundedEL"], None]] = None,
        log_every: Optional[int] = None,
        lr_final: Optional[float] = None,
        lr_schedule: str = "cosine",
    ) -> List[Dict[str, float]]:
        """Run the GroundedEL training loop (§3, §4, §5).

        Each epoch:
        1. Anneals :math:`\\tau` geometrically from ``tau0`` to ``tau1`` and, when
           ``lr_final is not None``, updates the Adam learning rate across
           ``optimizer.param_groups`` from ``learning_rate`` to ``lr_final``
           according to ``lr_schedule`` (``"cosine"`` or ``"linear"``).
        2. Computes ``L = L_fact + L_axioms + lambda_reg * L_region + L_minimality``
           via :meth:`compute_batch_losses` and performs one Adam step.
        3. Optionally calls :meth:`reseed_idle` when ``reseed_every > 0`` and
           ``(epoch + 1) % reseed_every == 0`` (never at the last epoch unless
           ``(epoch + 1) % reseed_every == 0``), followed by
           ``optimizer.state.clear()`` to reset Adam moments.
        4. Logs per-term losses every ``log_every`` epochs.
        5. Invokes ``epoch_callback(epoch, self)`` when provided, and performs
           validation / checkpoint saving every ``validate_every`` epochs when
           configured.

        Guarantees:
        - Reasoner-free (R1): never imports or calls any reasoner or semantic
          completion algorithm.
        - Deterministic for a fixed ``self.seed``.
        - When ``lr_final is None``, keeps ``learning_rate`` constant across all
          epochs. When ``lr_final is not None``, computes the epoch learning rate
          as a pure function of ``(epoch, epochs, learning_rate, lr_final)`` so
          :meth:`reseed_idle` never resets the learning-rate schedule.
        - Calls :meth:`reseed_idle` if and only if ``reseed_every > 0`` and
          ``(epoch + 1) % reseed_every == 0``.
        - Records per-epoch loss dictionaries in :attr:`loss_history` and returns
          them.

        Non-guarantees:
        - When ``epochs == 0``, performs zero gradient steps and returns ``[]``.

        Parameters
        ----------
        epochs : Optional[int], default None
            Number of training epochs :math:`\\ge 0`; defaults to ``self.epochs``
            when ``None``.
        validate_every : int, default 1
            Validation / checkpoint interval in epochs (:math:`\\ge 1`). When
            ``validate_every > epochs``, end-of-loop checkpoint overwriting is
            suppressed if ``epoch_callback`` has already written a checkpoint to
            :attr:`model_filepath`.
        epoch_callback : Optional[Callable[[int, GroundedEL], None]], default None
            Optional callback ``epoch_callback(epoch, model)`` invoked at the end
            of each epoch.
        log_every : Optional[int], default None
            Logging interval in epochs; defaults to ``self.log_every`` when ``None``.
        lr_final : Optional[float], default None
            Optional final learning rate :math:`> 0` at the last epoch. ``None``
            keeps a constant learning rate equal to ``self.learning_rate``.
        lr_schedule : {"cosine", "linear"}, default "cosine"
            Learning-rate decay schedule used when ``lr_final is not None``.

        Returns
        -------
        List[Dict[str, float]]
            Per-epoch recorded loss dictionaries.
        """
        eff_epochs = self.epochs if epochs is None else _validate_int_ge("epochs", epochs, 0)
        eff_val_every = _validate_int_ge("validate_every", validate_every, 1)
        eff_log_every = (
            self.log_every
            if log_every is None
            else _validate_int_ge("log_every", log_every, 0)
        )
        eff_lr_final: Optional[float] = (
            None
            if lr_final is None
            else _validate_positive_float("lr_final", lr_final)
        )
        if not isinstance(lr_schedule, str) or lr_schedule not in ("cosine", "linear"):
            raise ValueError(
                f"lr_schedule must be 'cosine' or 'linear', got {lr_schedule!r}"
            )
        self.loss_history = []
        if eff_epochs == 0:
            return self.loss_history

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        self.optimizer = optimizer
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 1337)
        best_val_loss = float("inf")
        saved_best_val = False

        for epoch in range(eff_epochs):
            self.module.train()
            if eff_lr_final is not None:
                cur_lr = self._scheduled_lr(
                    epoch,
                    eff_epochs,
                    self.learning_rate,
                    eff_lr_final,
                    lr_schedule,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
            if eff_epochs > 1:
                tau = self.tau0 * (
                    (self.tau1 / self.tau0) ** (epoch / (eff_epochs - 1))
                )
            else:
                tau = self.tau0
            self.module.set_tau(float(tau))

            losses = self.compute_batch_losses(generator=gen)
            total_loss = losses["total"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with th.no_grad():
                self.module.dead_logits.clamp_(0.0, 1.0)
                if self.bot_id is not None:
                    self.module.dead_logits[self.bot_id] = _DEAD_LOGIT

            if self.reseed_every > 0 and (epoch + 1) % self.reseed_every == 0:
                self.reseed_idle(generator=gen, prune_unused=True)
                optimizer.state.clear()

            rec = {k: float(v.detach().item()) for k, v in losses.items()}
            rec["tau"] = float(tau)
            self.loss_history.append(rec)

            if eff_log_every > 0 and (
                (epoch + 1) % eff_log_every == 0 or epoch == 0 or epoch + 1 == eff_epochs
            ):
                logger.info(
                    "Epoch %d/%d (tau=%.4f): total=%.4e fact=%.4e axioms=%.4e region=%.4e min=%.4e",
                    epoch + 1,
                    eff_epochs,
                    tau,
                    rec["total"],
                    rec["fact"],
                    rec["axioms"],
                    rec["region"],
                    rec["minimality"],
                )

            if epoch_callback is not None:
                epoch_callback(epoch, self)

            if (epoch + 1) % eff_val_every == 0:
                if (
                    self.dataset is not None
                    and self.dataset.validation is not None
                    and self._eval_gci_name is not None
                    and self._eval_gci_name in self.validation_datasets
                    and len(self.validation_datasets[self._eval_gci_name]) > 0
                ):
                    with th.no_grad():
                        self.module.eval()
                        gci_data = self.validation_datasets[self._eval_gci_name][:]
                        vloss = float(
                            th.mean(self.module(gci_data, self._eval_gci_name))
                            .detach()
                            .item()
                        )
                        if vloss < best_val_loss:
                            best_val_loss = vloss
                            if self.model_filepath is not None:
                                th.save(self.module.state_dict(), self.model_filepath)
                                saved_best_val = True

        if self.model_filepath is not None and not saved_best_val:
            if eff_val_every <= eff_epochs or not os.path.exists(self.model_filepath):
                th.save(self.module.state_dict(), self.model_filepath)

        return self.loss_history

    def _box_corner_samples(
        self,
        cen: th.Tensor,
        w: th.Tensor,
        active: th.Tensor,
        gen: th.Generator,
        max_corners_per_box: int = 32,
    ) -> th.Tensor:
        """Return interior-nudged box corner points ``c ± (1 - 1e-4) * w`` for active non-empty boxes."""
        valid_k = active & (th.amin(w, dim=-1) > 0.0)
        if not bool(valid_k.any().item()):
            return th.empty((0, self.embed_dim), dtype=cen.dtype, device=cen.device)
        c_act = cen[valid_k]  # (M, d)
        w_act = w[valid_k] * (1.0 - 1e-4)  # (M, d)
        m_cnt, dim = c_act.shape
        if dim <= 5 and (1 << dim) <= max_corners_per_box:
            signs = th.tensor(
                [
                    [(1.0 if (mask >> bit) & 1 else -1.0) for bit in range(dim)]
                    for mask in range(1 << dim)
                ],
                dtype=cen.dtype,
                device=cen.device,
            )  # (2^d, d)
            pts = c_act.unsqueeze(1) + w_act.unsqueeze(1) * signs.unsqueeze(0)
            return pts.reshape(-1, dim)
        u_bits = (
            th.randint(
                0,
                2,
                (m_cnt, max_corners_per_box, dim),
                device=gen.device,
                generator=gen,
            ).to(device=cen.device, dtype=cen.dtype)
            * 2.0
            - 1.0
        )
        pts = c_act.unsqueeze(1) + w_act.unsqueeze(1) * u_bits
        return pts.reshape(-1, dim)

    def _eval_elem_depths_chunked(
        self,
        eval_pos: th.Tensor,
        mentioned_classes: th.Tensor,
        chunk_size: int = 512,
    ) -> th.Tensor:
        """Compute signed crisp depth ``(L, M)`` only for ``eval_pos`` and ``mentioned_classes`` in chunks."""
        mod = self.module
        device = self.device
        n_eval_el = int(eval_pos.shape[0])
        n_ment = int(mentioned_classes.numel())
        out = th.empty((n_eval_el, n_ment), dtype=th.float32, device=device)
        if n_eval_el == 0 or n_ment == 0:
            return out

        for c_s in range(0, n_ment, chunk_size):
            c_e = min(c_s + chunk_size, n_ment)
            c_sub = mentioned_classes[c_s:c_e]  # (C_sub,)
            reg_sub = mod.class_regions(c_sub)  # (C_sub, K, d)
            c_dep = th.amin(
                reg_sub.w.unsqueeze(0)
                - th.abs(eval_pos.unsqueeze(1).unsqueeze(2) - reg_sub.cen.unsqueeze(0)),
                dim=-1,
            )  # (L, C_sub, K)
            has_act = reg_sub.active.any(dim=-1, keepdim=True)  # (C_sub, 1)
            fill_val = th.where(
                has_act.unsqueeze(0),
                th.full_like(c_dep, float("-inf")),
                th.full_like(c_dep, _EMPTY_DEPTH_FLOOR),
            )
            c_dep = th.where(reg_sub.active.unsqueeze(0), c_dep, fill_val)
            max_dep = th.amax(c_dep, dim=-1)  # (L, C_sub)
            if self.top_id is not None:
                max_dep = th.where(
                    (c_sub == self.top_id).unsqueeze(0),
                    th.full_like(max_dep, _TOP_DEPTH_CEIL),
                    max_dep,
                )
            if self.bot_id is not None:
                max_dep = th.where(
                    (c_sub == self.bot_id).unsqueeze(0),
                    th.full_like(max_dep, _EMPTY_DEPTH_FLOOR),
                    max_dep,
                )
            out[:, c_s:c_e] = max_dep
        return out

    def _eval_elem_memberships_chunked(
        self,
        eval_pos: th.Tensor,
        mentioned_classes: th.Tensor,
        chunk_size: int = 512,
    ) -> th.Tensor:
        """Compute crisp membership ``(L, M)`` only for ``eval_pos`` and ``mentioned_classes`` in chunks."""
        return crisp_membership(
            self._eval_elem_depths_chunked(
                eval_pos, mentioned_classes, chunk_size=chunk_size
            )
        )

    def crisp_report(
        self,
        n_eval: int = 10000,
        tol: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute reasoner-free crisp violation metrics, per-axiom worst depth, and depth-bucket violation counts (§7).

        A sampled point :math:`z` is a counterexample for a region axiom at
        tolerance ``tol >= 0`` only if its violating depth exceeds ``tol``:
        ``depth_RHS(z) < -tol`` (``<= 0.0`` at ``tol == 0.0``) for containment
        forms (``gci0``, ``gci1``, ``gci2``, ``gci3``, ``role_inclusion``,
        ``role_chain``) and ``min(depth_A, depth_B) > tol`` (analogously for
        ``gci0_bot`` and ``gci3_bot``) for disjointness/bottom forms.

        Guarantees:
        - Reasoner-free (R1): never imports or calls any reasoner or
          :class:`~mowl.geometric.completion.CanonicalModel`.
        - Never materialises a :math:`(|\\Delta| \\times |\\mathcal{C}|)` tensor;
          computes memberships only for the sampled evaluation elements and the
          classes mentioned by the evaluated axioms, in chunks.
        - ``tol=0.0`` reproduces exact strict-boundary ``elem_viol`` and
          ``region_viol`` numbers bit-for-bit.
        - Returns ``{"elem_viol": float, "region_viol": float, "worst_depth": dict, "n_axioms_violated_by_depth": dict, "n_empty_classes": int, "n_dead_elements": int}``
          where ``n_axioms_violated_by_depth`` reports violated axiom counts at
          ``tol`` in ``{0, 1e-5, 1e-4, 1e-3, 1e-2}`` and is non-increasing in ``tol``.

        Non-guarantees:
        - Does not compute reasoner-dependent metrics (``exist``, ``dead_ok``,
          ``n_false_dead``, ``faithful_named_*``, ``exact``).

        Parameters
        ----------
        n_eval : int, default 10000
            Total evaluation sample budget across axioms for ``region_viol``.
        tol : float, default 0.0
            Non-negative boundary tolerance :math:`\\ge 0` for region counterexamples.

        Returns
        -------
        Dict[str, Any]
            ``{"elem_viol": ..., "region_viol": ..., "worst_depth": ..., "n_axioms_violated_by_depth": ..., "n_empty_classes": ..., "n_dead_elements": ...}``.
        """
        _validate_int_ge("n_eval", n_eval, 1)
        tol_f = _validate_nonneg_weight("tol", tol)
        mod = self.module
        device = self.device
        td = self.training_datasets
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 9999)

        cls_names = _short_iri_map(self.class_index_dict)
        rel_names = _short_iri_map(self.object_property_index_dict)
        worst_depth: Dict[str, Dict[str, float]] = {}
        tol_counts: Dict[str, int] = {k_str: 0 for k_str, _ in _TOL_DEPTH_LEVELS}

        def _record_containment_axiom(
            ax_type: str,
            ax_str: str,
            has_lhs: bool,
            min_rhs_dep: float,
            viol_zero: bool,
        ) -> bool:
            bucket = worst_depth.setdefault(ax_type, {})
            w_val = float(min_rhs_dep) if has_lhs else 0.0
            if ax_str in bucket:
                bucket[ax_str] = min(bucket[ax_str], w_val)
            else:
                bucket[ax_str] = w_val
            for k_str, t_val in _TOL_DEPTH_LEVELS:
                if t_val == 0.0:
                    is_v = viol_zero
                else:
                    is_v = has_lhs and (w_val < -t_val)
                if is_v:
                    tol_counts[k_str] += 1
            if tol_f == 0.0:
                return viol_zero
            return has_lhs and (w_val < -tol_f)

        def _record_bot_axiom(
            ax_type: str,
            ax_str: str,
            max_lhs_dep: float,
            viol_zero: bool,
        ) -> bool:
            bucket = worst_depth.setdefault(ax_type, {})
            w_val = float(max_lhs_dep)
            if ax_str in bucket:
                bucket[ax_str] = max(bucket[ax_str], w_val)
            else:
                bucket[ax_str] = w_val
            for k_str, t_val in _TOL_DEPTH_LEVELS:
                if t_val == 0.0:
                    is_v = viol_zero
                else:
                    is_v = w_val > t_val
                if is_v:
                    tol_counts[k_str] += 1
            if tol_f == 0.0:
                return viol_zero
            return w_val > tol_f

        with th.no_grad():
            pos_all = mod.positions()  # (nb_elements, d)
            dead_c = mod.dead() > 0.5
            if self.bot_id is not None:
                dead_c[self.bot_id] = True
            dead_f = dead_c.to(dtype=th.float32)
            live_mask = ~dead_c
            live_ids = th.nonzero(live_mask, as_tuple=False).view(-1)

            if live_ids.numel() > 256:
                perm = th.randperm(live_ids.numel(), device="cpu", generator=gen)[:256].to(
                    device=device
                )
                eval_elem_ids = live_ids[perm]
            else:
                eval_elem_ids = live_ids
            eval_pos = pos_all[eval_elem_ids]
            eval_dead_f = dead_f[eval_elem_ids]

            # Collect unique classes mentioned by the evaluated axioms
            cls_parts: List[th.Tensor] = []
            for k_ax, cols in (
                ("gci0", (0, 1)),
                ("gci0_bot", (0,)),
                ("gci1", (0, 1, 2)),
                ("gci1_bot", (0, 1)),
                ("gci2", (0, 2)),
                ("gci3", (1, 2)),
                ("gci3_bot", (1,)),
            ):
                if k_ax in td and len(td[k_ax]) > 0:
                    t_ax = td[k_ax][:].to(device=device, dtype=th.long)
                    for col_i in cols:
                        cls_parts.append(t_ax[:, col_i])
            if cls_parts:
                mentioned_classes = th.unique(th.cat(cls_parts))
            else:
                mentioned_classes = th.empty((0,), dtype=th.long, device=device)

            cls_to_col = th.full(
                (self.nb_classes,), -1, dtype=th.long, device=device
            )
            if mentioned_classes.numel() > 0:
                cls_to_col[mentioned_classes] = th.arange(
                    mentioned_classes.numel(), dtype=th.long, device=device
                )

            # Compute signed depths and crisp memberships ONLY for eval_elem_ids and mentioned_classes in chunks
            elem_dep_live = self._eval_elem_depths_chunked(
                eval_pos, mentioned_classes, chunk_size=512
            )  # (L, M)
            elem_m_live = crisp_membership(elem_dep_live)  # (L, M)

            elem_viol_count = 0
            elem_total_count = 0

            # Count total TBox/RBox axioms to divide n_eval budget cleanly
            total_region_axioms = sum(
                len(td[k])
                for k in (
                    "gci0",
                    "gci0_bot",
                    "gci1",
                    "gci1_bot",
                    "gci2",
                    "gci3",
                    "gci3_bot",
                    "role_inclusion",
                    "role_chain",
                )
                if k in td and len(td[k]) > 0
            )
            reg_viol_axioms = 0
            reg_total_axioms = 0

            if total_region_axioms <= 512:
                n_per_ax = max(16, min(2000, n_eval // max(total_region_axioms, 1)))

                # 1. GCI0: A <= B
                if "gci0" in td and len(td["gci0"]) > 0:
                    for row in td["gci0"][:].detach().cpu().tolist():
                        a_id, b_id = int(row[0]), int(row[1])
                        col_a = int(cls_to_col[a_id].item())
                        col_b = int(cls_to_col[b_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci0(
                                elem_m_live[:, col_a],
                                elem_m_live[:, col_b],
                                dead=eval_dead_f,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_class_region(
                            mod, a_id, generator=gen, n_samples=n_per_ax
                        )
                        reg_a = mod.class_regions(a_id)
                        corners = self._box_corner_samples(
                            reg_a.cen, reg_a.w, reg_a.active, gen
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        if corners.numel() > 0:
                            pts_list.append(corners)
                        pts = th.cat(pts_list, dim=0)
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            db = mod.depth(pts, b_id, crisp=True)
                            mask = da > 0.0
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(db[mask].min().item())
                                viol_0 = bool((db[mask] <= 0.0).any().item())
                        ax_s = f"{cls_names.get(a_id, str(a_id))} <= {cls_names.get(b_id, str(b_id))}"
                        if _record_containment_axiom("gci0", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 2. GCI0_bot: A <= bot
                if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
                    for row in td["gci0_bot"][:].detach().cpu().tolist():
                        a_id = int(row[0])
                        col_a = int(cls_to_col[a_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci0_bot(elem_m_live[:, col_a], dead=eval_dead_f)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_class_region(
                            mod, a_id, generator=gen, n_samples=n_per_ax
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        pts = th.cat(pts_list, dim=0)
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            max_lhs = float(da.max().item())
                            viol_0 = bool(s_reg.valid.any().item()) or bool(
                                (da > 0.0).any().item()
                            )
                        else:
                            max_lhs = float(_EMPTY_DEPTH_FLOOR)
                            viol_0 = bool(s_reg.valid.any().item())
                        ax_s = f"{cls_names.get(a_id, str(a_id))} <= bot"
                        if _record_bot_axiom("gci0_bot", ax_s, max_lhs, viol_0):
                            reg_viol_axioms += 1

                # 3. GCI1: A cap B <= C
                if "gci1" in td and len(td["gci1"]) > 0:
                    for row in td["gci1"][:].detach().cpu().tolist():
                        a_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                        col_a = int(cls_to_col[a_id].item())
                        col_b = int(cls_to_col[b_id].item())
                        col_c = int(cls_to_col[c_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci1(
                                elem_m_live[:, col_a],
                                elem_m_live[:, col_b],
                                elem_m_live[:, col_c],
                                dead=eval_dead_f,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_pair_region(
                            mod, a_id, b_id, generator=gen, n_samples=n_per_ax
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        pts = th.cat(pts_list, dim=0)
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            db = mod.depth(pts, b_id, crisp=True)
                            dc = mod.depth(pts, c_id, crisp=True)
                            mask = (da > 0.0) & (db > 0.0)
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(dc[mask].min().item())
                                viol_0 = bool((dc[mask] <= 0.0).any().item())
                        ax_s = (
                            f"{cls_names.get(a_id, str(a_id))} cap "
                            f"{cls_names.get(b_id, str(b_id))} <= "
                            f"{cls_names.get(c_id, str(c_id))}"
                        )
                        if _record_containment_axiom("gci1", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 4. GCI1_bot: A cap B <= bot
                if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
                    for row in td["gci1_bot"][:].detach().cpu().tolist():
                        a_id, b_id = int(row[0]), int(row[1])
                        col_a = int(cls_to_col[a_id].item())
                        col_b = int(cls_to_col[b_id].item())
                        if eval_elem_ids.numel() > 0:
                            h_e = gci1_bot(
                                elem_m_live[:, col_a],
                                elem_m_live[:, col_b],
                                dead=eval_dead_f,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_inter = sample_pair_region(
                            mod, a_id, b_id, generator=gen, n_samples=n_per_ax
                        )
                        if bool(s_inter.valid.any().item()):
                            pts_list = [eval_pos, s_inter.points[s_inter.valid]]
                        else:
                            s_a = sample_class_region(
                                mod, a_id, generator=gen, n_samples=n_per_ax
                            )
                            pts_list = [eval_pos]
                            if bool(s_a.valid.any().item()):
                                pts_list.append(s_a.points[s_a.valid])
                        pts = th.cat(pts_list, dim=0)
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            db = mod.depth(pts, b_id, crisp=True)
                            ab_dep = th.minimum(da, db)
                            max_lhs = float(ab_dep.max().item())
                            viol_0 = bool(s_inter.valid.any().item()) or bool(
                                (ab_dep > 0.0).any().item()
                            )
                        else:
                            max_lhs = float(_EMPTY_DEPTH_FLOOR)
                            viol_0 = bool(s_inter.valid.any().item())
                        ax_s = (
                            f"{cls_names.get(a_id, str(a_id))} cap "
                            f"{cls_names.get(b_id, str(b_id))} <= bot"
                        )
                        if _record_bot_axiom("gci1_bot", ax_s, max_lhs, viol_0):
                            reg_viol_axioms += 1

                # 5. GCI2: A <= exists r.B
                if "gci2" in td and len(td["gci2"]) > 0:
                    for row in td["gci2"][:].detach().cpu().tolist():
                        a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                        col_a = int(cls_to_col[a_id].item())
                        pb = pos_all[b_id]
                        if eval_elem_ids.numel() > 0:
                            rho_e = crisp_membership(
                                mod.role_depth(eval_pos, pb, r_id, crisp=True)
                            )
                            db_e = dead_f[b_id].expand_as(eval_dead_f)
                            h_e = gci2(
                                elem_m_live[:, col_a],
                                rho_e,
                                dead=eval_dead_f,
                                dead_filler=db_e,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        s_reg = sample_class_region(
                            mod, a_id, generator=gen, n_samples=n_per_ax
                        )
                        reg_a = mod.class_regions(a_id)
                        corners = self._box_corner_samples(
                            reg_a.cen, reg_a.w, reg_a.active, gen
                        )
                        pts_list = [eval_pos]
                        if bool(s_reg.valid.any().item()):
                            pts_list.append(s_reg.points[s_reg.valid])
                        if corners.numel() > 0:
                            pts_list.append(corners)
                        pts = th.cat(pts_list, dim=0)
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if pts.shape[0] > 0:
                            da = mod.depth(pts, a_id, crisp=True)
                            r_dep = mod.role_depth(pts, pb, r_id, crisp=True)
                            mask = da > 0.0
                            has_lhs = bool(mask.any().item())
                            filler_dead = bool(dead_c[b_id].item())
                            if has_lhs:
                                rhs_dep = -da[mask] if filler_dead else r_dep[mask]
                                min_rhs = float(rhs_dep.min().item())
                                viol_0 = filler_dead or bool(
                                    (rhs_dep <= 0.0).any().item()
                                )
                        ax_s = (
                            f"{cls_names.get(a_id, str(a_id))} <= exists "
                            f"{rel_names.get(r_id, str(r_id))}."
                            f"{cls_names.get(b_id, str(b_id))}"
                        )
                        if _record_containment_axiom("gci2", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 6. GCI3: exists r.B <= C
                if "gci3" in td and len(td["gci3"]) > 0:
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for row in td["gci3"][:].detach().cpu().tolist():
                        r_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                        col_b = int(cls_to_col[b_id].item())
                        col_c = int(cls_to_col[c_id].item())
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            for y_idx in range(n_y):
                                y_id = int(y_eval_ids[y_idx].item())
                                rho_xy = crisp_membership(
                                    mod.role_depth(
                                        eval_pos, pos_all[y_id], r_id, crisp=True
                                    )
                                )
                                mb_y = elem_m_live[y_idx, col_b].expand_as(rho_xy)
                                h_e = gci3(
                                    rho_xy,
                                    mb_y,
                                    elem_m_live[:, col_c],
                                    dead=eval_dead_f,
                                )
                                elem_viol_count += int((h_e > 0.0).sum().item())
                                elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        h_s, t_s = sample_role_region(
                            mod, r_id, generator=gen, n_samples=n_per_ax
                        )
                        has_lhs = False
                        min_rhs = 0.0
                        viol_0 = False
                        if bool(h_s.valid.any().item()):
                            zx = h_s.points[h_s.valid]
                            zy = t_s.points[t_s.valid]
                            r_dep = mod.role_depth(zx, zy, r_id, crisp=True)
                            db_z = mod.depth(zy, b_id, crisp=True)
                            dc_z = mod.depth(zx, c_id, crisp=True)
                            mask = (r_dep > 0.0) & (db_z > 0.0)
                            has_lhs = bool(mask.any().item())
                            if has_lhs:
                                min_rhs = float(dc_z[mask].min().item())
                                viol_0 = bool((dc_z[mask] <= 0.0).any().item())
                        ax_s = (
                            f"exists {rel_names.get(r_id, str(r_id))}."
                            f"{cls_names.get(b_id, str(b_id))} <= "
                            f"{cls_names.get(c_id, str(c_id))}"
                        )
                        if _record_containment_axiom("gci3", ax_s, has_lhs, min_rhs, viol_0):
                            reg_viol_axioms += 1

                # 7. GCI3_bot: exists r.B <= bot
                if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for row in td["gci3_bot"][:].detach().cpu().tolist():
                        r_id, b_id = int(row[0]), int(row[1])
                        col_b = int(cls_to_col[b_id].item())
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            for y_idx in range(n_y):
                                y_id = int(y_eval_ids[y_idx].item())
                                rho_xy = crisp_membership(
                                    mod.role_depth(
                                        eval_pos, pos_all[y_id], r_id, crisp=True
                                    )
                                )
                                mb_y = elem_m_live[y_idx, col_b].expand_as(rho_xy)
                                h_e = gci3_bot(rho_xy, mb_y, dead=eval_dead_f)
                                elem_viol_count += int((h_e > 0.0).sum().item())
                                elem_total_count += int(h_e.numel())
                        reg_total_axioms += 1
                        h_s, t_s = sample_role_region(
                            mod, r_id, generator=gen, n_samples=n_per_ax
                        )
                        if bool(h_s.valid.any().item()):
                            zx = h_s.points[h_s.valid]
                            zy = t_s.points[t_s.valid]
                            r_dep = mod.role_depth(zx, zy, r_id, crisp=True)
                            db_z = mod.depth(zy, b_id, crisp=True)
                            rb_dep = th.minimum(r_dep, db_z)
                            max_lhs = float(rb_dep.max().item())
                            viol_0 = bool((rb_dep > 0.0).any().item())
                        else:
                            max_lhs = float(_EMPTY_DEPTH_FLOOR)
                            viol_0 = False
                        ax_s = (
                            f"exists {rel_names.get(r_id, str(r_id))}."
                            f"{cls_names.get(b_id, str(b_id))} <= bot"
                        )
                        if _record_bot_axiom("gci3_bot", ax_s, max_lhs, viol_0):
                            reg_viol_axioms += 1
            else:
                # Chunked vectorised evaluation for large ontologies (e.g. GO)
                n_per_ax = max(1, min(16, n_eval // max(total_region_axioms, 1)))
                ax_chunk = 512

                if "gci0" in td and len(td["gci0"]) > 0:
                    data_g0 = td["gci0"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g0.shape[0], ax_chunk):
                        rows = data_g0[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c = rows[:, 0]
                        b_c = rows[:, 1]
                        ca = cls_to_col[a_c]
                        cb = cls_to_col[b_c]
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]  # (L, B)
                            mb_e = elem_m_live[:, cb]  # (L, B)
                            h_e = gci0(
                                ma_e,
                                mb_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            da_e = elem_dep_live[:, ca]
                            db_e = elem_dep_live[:, cb]
                            mask_e = da_e > 0.0
                            min_dep = th.where(
                                mask_e, db_e, th.full_like(db_e, float("inf"))
                            ).amin(dim=0)
                            has_lhs = mask_e.any(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        s_reg = sample_class_region(mod, a_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            db_r = mod.depth(s_reg.points, b_rep, crisp=True)
                            mask_r = (da_r > 0.0) & s_reg.valid
                            dep_r = th.where(
                                mask_r, db_r, th.full_like(db_r, float("inf"))
                            ).view(bsz, n_per_ax)
                            min_dep = th.minimum(min_dep, dep_r.amin(dim=1))
                            has_lhs = has_lhs | mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & (min_dep <= 0.0)
                        rows_list = rows.detach().cpu().tolist()
                        has_list = has_lhs.detach().cpu().tolist()
                        min_list = min_dep.detach().cpu().tolist()
                        v0_list = viol_0_b.detach().cpu().tolist()
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows_list, has_list, min_list, v0_list
                        ):
                            a_i, b_i = int(r_row[0]), int(r_row[1])
                            ax_s = f"{cls_names.get(a_i, str(a_i))} <= {cls_names.get(b_i, str(b_i))}"
                            if _record_containment_axiom(
                                "gci0", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
                    data_g0b = td["gci0_bot"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g0b.shape[0], ax_chunk):
                        rows = data_g0b[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c = rows[:, 0]
                        ca = cls_to_col[a_c]
                        max_dep = th.full(
                            (bsz,), _EMPTY_DEPTH_FLOOR, dtype=th.float32, device=device
                        )
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]
                            h_e = gci0_bot(
                                ma_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            max_dep = elem_dep_live[:, ca].amax(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        s_reg = sample_class_region(mod, a_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            da_r_m = th.where(
                                s_reg.valid,
                                da_r,
                                th.full_like(da_r, _EMPTY_DEPTH_FLOOR),
                            ).view(bsz, n_per_ax)
                            max_dep = th.maximum(max_dep, da_r_m.amax(dim=1))
                        viol_0_b = (max_dep > 0.0) | s_reg.valid.view(
                            bsz, n_per_ax
                        ).any(dim=1)
                        for r_row, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            max_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i = int(r_row[0])
                            ax_s = f"{cls_names.get(a_i, str(a_i))} <= bot"
                            if _record_bot_axiom(
                                "gci0_bot", ax_s, float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci1" in td and len(td["gci1"]) > 0:
                    data_g1 = td["gci1"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g1.shape[0], ax_chunk):
                        rows = data_g1[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c, b_c, c_c = rows[:, 0], rows[:, 1], rows[:, 2]
                        ca, cb, cc = cls_to_col[a_c], cls_to_col[b_c], cls_to_col[c_c]
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]
                            mb_e = elem_m_live[:, cb]
                            mc_e = elem_m_live[:, cc]
                            h_e = gci1(
                                ma_e,
                                mb_e,
                                mc_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            da_e = elem_dep_live[:, ca]
                            db_e = elem_dep_live[:, cb]
                            dc_e = elem_dep_live[:, cc]
                            mask_e = (da_e > 0.0) & (db_e > 0.0)
                            min_dep = th.where(
                                mask_e, dc_e, th.full_like(dc_e, float("inf"))
                            ).amin(dim=0)
                            has_lhs = mask_e.any(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        c_rep = c_c.repeat_interleave(n_per_ax)
                        s_reg = sample_pair_region(mod, a_rep, b_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            db_r = mod.depth(s_reg.points, b_rep, crisp=True)
                            dc_r = mod.depth(s_reg.points, c_rep, crisp=True)
                            mask_r = (da_r > 0.0) & (db_r > 0.0) & s_reg.valid
                            dep_r = th.where(
                                mask_r, dc_r, th.full_like(dc_r, float("inf"))
                            ).view(bsz, n_per_ax)
                            min_dep = th.minimum(min_dep, dep_r.amin(dim=1))
                            has_lhs = has_lhs | mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & (min_dep <= 0.0)
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            has_lhs.detach().cpu().tolist(),
                            min_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i, b_i, c_i = int(r_row[0]), int(r_row[1]), int(r_row[2])
                            ax_s = (
                                f"{cls_names.get(a_i, str(a_i))} cap "
                                f"{cls_names.get(b_i, str(b_i))} <= "
                                f"{cls_names.get(c_i, str(c_i))}"
                            )
                            if _record_containment_axiom(
                                "gci1", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
                    data_g1b = td["gci1_bot"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g1b.shape[0], ax_chunk):
                        rows = data_g1b[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c, b_c = rows[:, 0], rows[:, 1]
                        ca, cb = cls_to_col[a_c], cls_to_col[b_c]
                        max_dep = th.full(
                            (bsz,), _EMPTY_DEPTH_FLOOR, dtype=th.float32, device=device
                        )
                        if eval_elem_ids.numel() > 0:
                            ma_e = elem_m_live[:, ca]
                            mb_e = elem_m_live[:, cb]
                            h_e = gci1_bot(
                                ma_e,
                                mb_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            max_dep = th.minimum(
                                elem_dep_live[:, ca], elem_dep_live[:, cb]
                            ).amax(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        s_inter = sample_pair_region(mod, a_rep, b_rep, generator=gen)
                        if bool(s_inter.valid.any().item()):
                            da_r = mod.depth(s_inter.points, a_rep, crisp=True)
                            db_r = mod.depth(s_inter.points, b_rep, crisp=True)
                            ab_r = th.where(
                                s_inter.valid,
                                th.minimum(da_r, db_r),
                                th.full_like(da_r, _EMPTY_DEPTH_FLOOR),
                            ).view(bsz, n_per_ax)
                            max_dep = th.maximum(max_dep, ab_r.amax(dim=1))
                        viol_0_b = (max_dep > 0.0) | s_inter.valid.view(
                            bsz, n_per_ax
                        ).any(dim=1)
                        for r_row, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            max_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i, b_i = int(r_row[0]), int(r_row[1])
                            ax_s = (
                                f"{cls_names.get(a_i, str(a_i))} cap "
                                f"{cls_names.get(b_i, str(b_i))} <= bot"
                            )
                            if _record_bot_axiom(
                                "gci1_bot", ax_s, float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci2" in td and len(td["gci2"]) > 0:
                    data_g2 = td["gci2"][:].to(device=device, dtype=th.long)
                    for s_i in range(0, data_g2.shape[0], ax_chunk):
                        rows = data_g2[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        a_c, r_c, b_c = rows[:, 0], rows[:, 1], rows[:, 2]
                        ca = cls_to_col[a_c]
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if eval_elem_ids.numel() > 0:
                            l_cnt = int(eval_elem_ids.numel())
                            x_mat = eval_elem_ids.unsqueeze(0).expand(bsz, l_cnt)
                            y_mat = b_c.unsqueeze(1)
                            _, r_dep_c = self._grid_role_membership(
                                pos_all, x_mat, y_mat, r_c, need_crisp=True
                            )
                            assert r_dep_c is not None
                            r_dep_e = r_dep_c.view(bsz, l_cnt).t()
                            rho_e = crisp_membership(r_dep_e)
                            ma_e = elem_m_live[:, ca]  # (L, B)
                            db_e = dead_f[b_c].unsqueeze(0).expand_as(ma_e)
                            h_e = gci2(
                                ma_e,
                                rho_e,
                                dead=eval_dead_f.unsqueeze(1).expand_as(ma_e),
                                dead_filler=db_e,
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                            da_e = elem_dep_live[:, ca]
                            rhs_e = th.where(
                                dead_c[b_c].unsqueeze(0), -da_e, r_dep_e
                            )
                            mask_e = da_e > 0.0
                            min_dep = th.where(
                                mask_e, rhs_e, th.full_like(rhs_e, float("inf"))
                            ).amin(dim=0)
                            has_lhs = mask_e.any(dim=0)
                        a_rep = a_c.repeat_interleave(n_per_ax)
                        r_rep = r_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        s_reg = sample_class_region(mod, a_rep, generator=gen)
                        if bool(s_reg.valid.any().item()):
                            da_r = mod.depth(s_reg.points, a_rep, crisp=True)
                            rd_r = mod.role_depth(
                                s_reg.points, pos_all[b_rep], r_rep, crisp=True
                            )
                            rhs_r = th.where(dead_c[b_rep], -da_r, rd_r)
                            mask_r = (da_r > 0.0) & s_reg.valid
                            dep_r = th.where(
                                mask_r, rhs_r, th.full_like(rhs_r, float("inf"))
                            ).view(bsz, n_per_ax)
                            min_dep = th.minimum(min_dep, dep_r.amin(dim=1))
                            has_lhs = has_lhs | mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & ((min_dep <= 0.0) | dead_c[b_c])
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            has_lhs.detach().cpu().tolist(),
                            min_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            a_i, r_i, b_i = int(r_row[0]), int(r_row[1]), int(r_row[2])
                            ax_s = (
                                f"{cls_names.get(a_i, str(a_i))} <= exists "
                                f"{rel_names.get(r_i, str(r_i))}."
                                f"{cls_names.get(b_i, str(b_i))}"
                            )
                            if _record_containment_axiom(
                                "gci2", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci3" in td and len(td["gci3"]) > 0:
                    data_g3 = td["gci3"][:].to(device=device, dtype=th.long)
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for s_i in range(0, data_g3.shape[0], ax_chunk):
                        rows = data_g3[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        r_c, b_c, c_c = rows[:, 0], rows[:, 1], rows[:, 2]
                        cb, cc = cls_to_col[b_c], cls_to_col[c_c]
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            l_cnt = int(eval_elem_ids.numel())
                            x_mat = eval_elem_ids.unsqueeze(0).expand(bsz, l_cnt)
                            y_mat = y_eval_ids.unsqueeze(0).expand(bsz, n_y)
                            _, r_dep_c = self._grid_role_membership(
                                pos_all, x_mat, y_mat, r_c, need_crisp=True
                            )
                            assert r_dep_c is not None
                            rho_xy = crisp_membership(r_dep_c).view(bsz, l_cnt, n_y)
                            mb_y = elem_m_live[:n_y, cb].t().unsqueeze(1).expand(
                                bsz, l_cnt, n_y
                            )
                            mc_x = elem_m_live[:, cc].t().unsqueeze(2).expand(
                                bsz, l_cnt, n_y
                            )
                            dead_x = eval_dead_f.view(1, l_cnt, 1).expand(
                                bsz, l_cnt, n_y
                            )
                            h_e = gci3(rho_xy, mb_y, mc_x, dead=dead_x)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        r_rep = r_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        c_rep = c_c.repeat_interleave(n_per_ax)
                        h_s, t_s = sample_role_region(mod, r_rep, generator=gen)
                        min_dep = th.full((bsz,), float("inf"), dtype=th.float32, device=device)
                        has_lhs = th.zeros((bsz,), dtype=th.bool, device=device)
                        if bool(h_s.valid.any().item()):
                            rd_r = mod.role_depth(
                                h_s.points, t_s.points, r_rep, crisp=True
                            )
                            db_r = mod.depth(t_s.points, b_rep, crisp=True)
                            dc_r = mod.depth(h_s.points, c_rep, crisp=True)
                            mask_r = (rd_r > 0.0) & (db_r > 0.0) & h_s.valid
                            dep_r = th.where(
                                mask_r, dc_r, th.full_like(dc_r, float("inf"))
                            ).view(bsz, n_per_ax)
                            min_dep = dep_r.amin(dim=1)
                            has_lhs = mask_r.view(bsz, n_per_ax).any(dim=1)
                        viol_0_b = has_lhs & (min_dep <= 0.0)
                        for r_row, h_flag, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            has_lhs.detach().cpu().tolist(),
                            min_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            r_i, b_i, c_i = int(r_row[0]), int(r_row[1]), int(r_row[2])
                            ax_s = (
                                f"exists {rel_names.get(r_i, str(r_i))}."
                                f"{cls_names.get(b_i, str(b_i))} <= "
                                f"{cls_names.get(c_i, str(c_i))}"
                            )
                            if _record_containment_axiom(
                                "gci3", ax_s, bool(h_flag), float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

                if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
                    data_g3b = td["gci3_bot"][:].to(device=device, dtype=th.long)
                    n_y = min(int(eval_elem_ids.numel()), 32)
                    y_eval_ids = eval_elem_ids[:n_y]
                    for s_i in range(0, data_g3b.shape[0], ax_chunk):
                        rows = data_g3b[s_i : s_i + ax_chunk]
                        bsz = int(rows.shape[0])
                        r_c, b_c = rows[:, 0], rows[:, 1]
                        cb = cls_to_col[b_c]
                        if eval_elem_ids.numel() > 0 and n_y > 0:
                            l_cnt = int(eval_elem_ids.numel())
                            x_mat = eval_elem_ids.unsqueeze(0).expand(bsz, l_cnt)
                            y_mat = y_eval_ids.unsqueeze(0).expand(bsz, n_y)
                            _, r_dep_c = self._grid_role_membership(
                                pos_all, x_mat, y_mat, r_c, need_crisp=True
                            )
                            assert r_dep_c is not None
                            rho_xy = crisp_membership(r_dep_c).view(bsz, l_cnt, n_y)
                            mb_y = elem_m_live[:n_y, cb].t().unsqueeze(1).expand(
                                bsz, l_cnt, n_y
                            )
                            dead_x = eval_dead_f.view(1, l_cnt, 1).expand(
                                bsz, l_cnt, n_y
                            )
                            h_e = gci3_bot(rho_xy, mb_y, dead=dead_x)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                        r_rep = r_c.repeat_interleave(n_per_ax)
                        b_rep = b_c.repeat_interleave(n_per_ax)
                        h_s, t_s = sample_role_region(mod, r_rep, generator=gen)
                        max_dep = th.full(
                            (bsz,), _EMPTY_DEPTH_FLOOR, dtype=th.float32, device=device
                        )
                        if bool(h_s.valid.any().item()):
                            rd_r = mod.role_depth(
                                h_s.points, t_s.points, r_rep, crisp=True
                            )
                            db_r = mod.depth(t_s.points, b_rep, crisp=True)
                            rb_r = th.where(
                                h_s.valid,
                                th.minimum(rd_r, db_r),
                                th.full_like(rd_r, _EMPTY_DEPTH_FLOOR),
                            ).view(bsz, n_per_ax)
                            max_dep = rb_r.amax(dim=1)
                        viol_0_b = max_dep > 0.0
                        for r_row, m_val, v0_flag in zip(
                            rows.detach().cpu().tolist(),
                            max_dep.detach().cpu().tolist(),
                            viol_0_b.detach().cpu().tolist(),
                        ):
                            r_i, b_i = int(r_row[0]), int(r_row[1])
                            ax_s = (
                                f"exists {rel_names.get(r_i, str(r_i))}."
                                f"{cls_names.get(b_i, str(b_i))} <= bot"
                            )
                            if _record_bot_axiom(
                                "gci3_bot", ax_s, float(m_val), bool(v0_flag)
                            ):
                                reg_viol_axioms += 1
                        reg_total_axioms += bsz

            # 8. Role inclusion: r <= s
            n_per_role_ax = max(16, min(2000, n_eval // max(total_region_axioms, 1)))
            if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
                y_eval_ids = (
                    eval_elem_ids[:32]
                    if eval_elem_ids.numel() > 32
                    else eval_elem_ids
                )
                for row in td["role_inclusion"][:].detach().cpu().tolist():
                    r_id, s_id = int(row[0]), int(row[1])
                    if eval_elem_ids.numel() > 0 and y_eval_ids.numel() > 0:
                        for y_id in y_eval_ids.tolist():
                            rr = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            rs = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], s_id, crisp=True)
                            )
                            h_e = role_inclusion(rr, rs)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    h_s, t_s = sample_role_region(
                        mod, r_id, generator=gen, n_samples=n_per_role_ax
                    )
                    has_lhs = False
                    min_rhs = 0.0
                    viol_0 = False
                    if bool(h_s.valid.any().item()):
                        zx = h_s.points[h_s.valid]
                        zy = t_s.points[t_s.valid]
                        dep_r = mod.role_depth(zx, zy, r_id, crisp=True)
                        dep_s = mod.role_depth(zx, zy, s_id, crisp=True)
                        mask = dep_r > 0.0
                        has_lhs = bool(mask.any().item())
                        if has_lhs:
                            min_rhs = float(dep_s[mask].min().item())
                            viol_0 = bool((dep_s[mask] <= 0.0).any().item())
                    ax_s = f"{rel_names.get(r_id, str(r_id))} <= {rel_names.get(s_id, str(s_id))}"
                    if _record_containment_axiom("role_inclusion", ax_s, has_lhs, min_rhs, viol_0):
                        reg_viol_axioms += 1

            # 9. Role chain: r o s <= t
            if "role_chain" in td and len(td["role_chain"]) > 0:
                c_eval_ids = (
                    eval_elem_ids[:16]
                    if eval_elem_ids.numel() > 16
                    else eval_elem_ids
                )
                c_pos = pos_all[c_eval_ids]
                for row in td["role_chain"][:].detach().cpu().tolist():
                    r_id, s_id, t_id = int(row[0]), int(row[1]), int(row[2])
                    if c_eval_ids.numel() > 0:
                        for y_id in c_eval_ids.tolist():
                            rho_xy = crisp_membership(
                                mod.role_depth(c_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            for z_id in c_eval_ids.tolist():
                                rho_yz = crisp_membership(
                                    mod.role_depth(
                                        pos_all[y_id], pos_all[z_id], s_id, crisp=True
                                    )
                                ).expand_as(rho_xy)
                                rho_xz = crisp_membership(
                                    mod.role_depth(
                                        c_pos, pos_all[z_id], t_id, crisp=True
                                    )
                                )
                                h_e = role_chain(rho_xy, rho_yz, rho_xz)
                                elem_viol_count += int((h_e > 0.0).sum().item())
                                elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    hx, hy = sample_role_region(
                        mod, r_id, generator=gen, n_samples=n_per_role_ax
                    )
                    _, tz = sample_role_region(
                        mod, s_id, generator=gen, n_samples=n_per_role_ax
                    )
                    v_c = hx.valid & tz.valid
                    has_lhs = False
                    min_rhs = 0.0
                    viol_0 = False
                    if bool(v_c.any().item()):
                        zx = hx.points[v_c]
                        zy = hy.points[v_c]
                        zz = tz.points[v_c]
                        dep_r = mod.role_depth(zx, zy, r_id, crisp=True)
                        dep_s = mod.role_depth(zy, zz, s_id, crisp=True)
                        dep_t = mod.role_depth(zx, zz, t_id, crisp=True)
                        mask = (dep_r > 0.0) & (dep_s > 0.0)
                        has_lhs = bool(mask.any().item())
                        if has_lhs:
                            min_rhs = float(dep_t[mask].min().item())
                            viol_0 = bool((dep_t[mask] <= 0.0).any().item())
                    ax_s = (
                        f"{rel_names.get(r_id, str(r_id))} o "
                        f"{rel_names.get(s_id, str(s_id))} <= "
                        f"{rel_names.get(t_id, str(t_id))}"
                    )
                    if _record_containment_axiom("role_chain", ax_s, has_lhs, min_rhs, viol_0):
                        reg_viol_axioms += 1

            # 10. ABox assertions
            if "class_assertion" in td and len(td["class_assertion"]) > 0:
                ca_data = td["class_assertion"][:].to(device=device, dtype=th.long)
                for s_i in range(0, ca_data.shape[0], 4096):
                    rows = ca_data[s_i : s_i + 4096]
                    el_a = self.nb_classes + rows[:, 0]
                    c_id = rows[:, 1]
                    m_ca = crisp_membership(
                        mod.depth(pos_all[el_a], c_id, crisp=True)
                    )
                    h_e = class_assertion(m_ca)
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

            if (
                "object_property_assertion" in td
                and len(td["object_property_assertion"]) > 0
            ):
                opa_data = td["object_property_assertion"][:].to(
                    device=device, dtype=th.long
                )
                for s_i in range(0, opa_data.shape[0], 4096):
                    rows = opa_data[s_i : s_i + 4096]
                    el_a = self.nb_classes + rows[:, 0]
                    r_id = rows[:, 1]
                    el_b = self.nb_classes + rows[:, 2]
                    rho_ab = crisp_membership(
                        mod.role_depth(pos_all[el_a], pos_all[el_b], r_id, crisp=True)
                    )
                    h_e = role_assertion(rho_ab)
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += int(h_e.numel())

            elem_viol = (
                float(elem_viol_count / elem_total_count)
                if elem_total_count > 0
                else 0.0
            )
            region_viol = (
                float(reg_viol_axioms / reg_total_axioms)
                if reg_total_axioms > 0
                else 0.0
            )
            cls_reg_all = mod.class_regions()
            empty_cls_mask = ~cls_reg_all.active.any(dim=1)
            if self.bot_id is not None:
                empty_cls_mask[self.bot_id] = False
            if self.top_id is not None:
                empty_cls_mask[self.top_id] = False
            n_empty_classes = int(empty_cls_mask.sum().item())

            dead_elem_mask = (mod.dead() > 0.5).clone()
            if self.bot_id is not None:
                dead_elem_mask[self.bot_id] = False
            n_dead_elements = int(dead_elem_mask.sum().item())

            return {
                "elem_viol": elem_viol,
                "region_viol": region_viol,
                "worst_depth": worst_depth,
                "n_axioms_violated_by_depth": _ToleranceCountDict(tol_counts),
                "n_empty_classes": n_empty_classes,
                "n_dead_elements": n_dead_elements,
            }

    def evaluate_model(
        self,
        use_reasoner: bool = True,
        n_eval: int = 10000,
        tol: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute §7 evaluation metrics dictionary.

        Lazily imports :class:`~mowl.geometric.completion.CanonicalModel` inside
        this method ONLY when ``use_reasoner=True``.

        Guarantees:
        - Returns a dict with keys ``"elem_viol"``, ``"region_viol"``,
          ``"region_viol_tol1e-4"``, ``"exist"``, ``"dead_ok"``,
          ``"n_empty_classes"``, ``"n_dead_elements"``, ``"n_false_dead"``,
          ``"faithful_named_precision"``, ``"faithful_named_recall"``,
          ``"faithful_named"``, ``"exact"``, and ``"exact_tol1e-4"``.
        - ``exact`` is ``True`` if and only if ``elem_viol == 0.0``,
          ``region_viol == 0.0``, ``exist == 1.0``, and ``dead_ok is True``.
        - ``region_viol_tol1e-4`` and ``exact_tol1e-4`` are always computed at
          tolerance ``1e-4``, independent of ``tol``.

        Non-guarantees:
        - When ``use_reasoner=False``, ``exist`` is evaluated over live classes
          (``dead <= 0.5``) and ``n_false_dead`` and ``faithful_named_*`` are ``None``.

        Parameters
        ----------
        use_reasoner : bool, default True
            Whether to compute ground-truth satisfiability and subsumptions via
            :class:`~mowl.geometric.completion.CanonicalModel`.
        n_eval : int, default 10000
            Sample budget for ``region_viol`` and region inclusion checks.
        tol : float, default 0.0
            Non-negative boundary tolerance :math:`\\ge 0` passed to :meth:`crisp_report`.

        Returns
        -------
        Dict[str, Any]
            §7 evaluation metrics dictionary.
        """
        base = self.crisp_report(n_eval=n_eval, tol=tol)
        elem_viol = base["elem_viol"]
        region_viol = base["region_viol"]
        n_empty_classes = int(base["n_empty_classes"])
        n_dead_elements = int(base["n_dead_elements"])
        td = self.training_datasets
        total_region_axioms = sum(
            len(td[k])
            for k in (
                "gci0",
                "gci0_bot",
                "gci1",
                "gci1_bot",
                "gci2",
                "gci3",
                "gci3_bot",
                "role_inclusion",
                "role_chain",
            )
            if k in td and len(td[k]) > 0
        )
        region_viol_tol1e4 = (
            float(base["n_axioms_violated_by_depth"]["1e-4"] / total_region_axioms)
            if total_region_axioms > 0
            else 0.0
        )

        mod = self.module
        device = self.device
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 2026)

        with th.no_grad():
            pos_all = mod.positions()
            dead_cls = (mod.dead()[: self.nb_classes] > 0.5).cpu()
            if self.bot_id is not None:
                dead_cls[self.bot_id] = True

            self_inside = th.zeros((self.nb_classes,), dtype=th.bool)
            for c_start in range(0, self.nb_classes, 8192):
                c_end = min(c_start + 8192, self.nb_classes)
                c_ids = th.arange(c_start, c_end, dtype=th.long, device=device)
                dep_c = mod.depth(pos_all[c_ids], c_ids, crisp=True).cpu()
                self_inside[c_start:c_end] = (dep_c > 0.0) & (~dead_cls[c_start:c_end])

            if not use_reasoner:
                live_cls = ~dead_cls
                exist_val = (
                    float(self_inside[live_cls].to(th.float32).mean().item())
                    if bool(live_cls.any().item())
                    else 1.0
                )
                dead_ok = True
                exact = bool(
                    elem_viol == 0.0
                    and region_viol == 0.0
                    and exist_val == 1.0
                    and dead_ok
                )
                exact_tol1e4 = bool(
                    elem_viol == 0.0
                    and region_viol_tol1e4 == 0.0
                    and exist_val == 1.0
                    and dead_ok
                )
                return {
                    "elem_viol": elem_viol,
                    "region_viol": region_viol,
                    "region_viol_tol1e-4": region_viol_tol1e4,
                    "exist": exist_val,
                    "dead_ok": dead_ok,
                    "n_empty_classes": n_empty_classes,
                    "n_dead_elements": n_dead_elements,
                    "n_false_dead": None,
                    "faithful_named_precision": None,
                    "faithful_named_recall": None,
                    "faithful_named": (None, None),
                    "exact": exact,
                    "exact_tol1e-4": exact_tol1e4,
                }

            if self.dataset is None or getattr(self.dataset, "ontology", None) is None:
                raise ValueError(
                    "use_reasoner=True requires an OWL dataset; pass use_reasoner=False for table-constructed GroundedEL models"
                )

            from mowl.geometric.completion import CanonicalModel

            cm = CanonicalModel.from_ontology(self.dataset.ontology)

            sat_mask = th.zeros((self.nb_classes,), dtype=th.bool)
            for c_iri, c_idx in self.class_index_dict.items():
                if c_iri == BOTTOM:
                    sat_mask[c_idx] = False
                elif c_iri == TOP:
                    sat_mask[c_idx] = bool(cm.is_satisfiable(TOP))
                else:
                    sat_mask[c_idx] = bool(cm.is_satisfiable(c_iri))

            unsat_mask = ~sat_mask
            dead_ok = bool(th.equal(dead_cls, unsat_mask))
            n_false_dead = int((dead_cls & sat_mask).sum().item())

            if bool(sat_mask.any().item()):
                exist_val = float(
                    self_inside[sat_mask].to(dtype=th.float32).mean().item()
                )
            else:
                exist_val = 1.0

            # Named class region inclusion faithfulness (M2)
            named_items = [
                (iri, idx)
                for iri, idx in self.named_class_index_dict.items()
                if iri not in (TOP, BOTTOM) and bool(sat_mask[idx].item())
            ]
            if len(named_items) > 128:
                named_items = named_items[:128]

            n_named = len(named_items)
            if n_named >= 2:
                n_pts_per_cls = max(16, min(256, n_eval // max(n_named, 1)))
                all_regs = mod.class_regions()
                cls_points: Dict[int, th.Tensor] = {}
                for _, c_idx in named_items:
                    s_reg = sample_class_region(
                        mod, c_idx, generator=gen, n_samples=n_pts_per_cls
                    )
                    corners = self._box_corner_samples(
                        all_regs.cen[c_idx : c_idx + 1],
                        all_regs.w[c_idx : c_idx + 1],
                        all_regs.active[c_idx : c_idx + 1],
                        gen,
                    )
                    pts_parts = [pos_all[c_idx : c_idx + 1]]
                    if bool(s_reg.valid.any().item()):
                        pts_parts.append(s_reg.points[s_reg.valid])
                    if corners.numel() > 0:
                        pts_parts.append(corners)
                    cand_pts = th.cat(pts_parts, dim=0)
                    in_c = mod.depth(cand_pts, c_idx, crisp=True) > 0.0
                    cls_points[c_idx] = cand_pts[in_c]

                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for a_iri, a_idx in named_items:
                    pts_a = cls_points[a_idx]
                    for b_iri, b_idx in named_items:
                        if a_idx == b_idx:
                            continue
                        gt_entailed = bool(
                            cm.extension(a_iri).issubset(cm.extension(b_iri))
                        )
                        if pts_a.shape[0] == 0:
                            pred_inc = True
                        else:
                            pred_inc = bool(
                                (mod.depth(pts_a, b_idx, crisp=True) > 0.0)
                                .all()
                                .item()
                            )
                        if gt_entailed and pred_inc:
                            tp += 1
                        elif not gt_entailed and pred_inc:
                            fp += 1
                        elif not gt_entailed and not pred_inc:
                            tn += 1
                        else:
                            fn += 1

                # Precision/recall of non-entailed pairs being separated & entailed pairs included
                precision = (
                    float(tn / (tn + fn))
                    if (tn + fn) > 0
                    else (1.0 if fp == 0 else 0.0)
                )
                recall = (
                    float(tn / (tn + fp))
                    if (tn + fp) > 0
                    else (1.0 if fn == 0 else 0.0)
                )
            else:
                precision = 1.0
                recall = 1.0

            exact = bool(
                elem_viol == 0.0
                and region_viol == 0.0
                and exist_val == 1.0
                and dead_ok
            )
            exact_tol1e4 = bool(
                elem_viol == 0.0
                and region_viol_tol1e4 == 0.0
                and exist_val == 1.0
                and dead_ok
            )
            return {
                "elem_viol": elem_viol,
                "region_viol": region_viol,
                "region_viol_tol1e-4": region_viol_tol1e4,
                "exist": exist_val,
                "dead_ok": dead_ok,
                "n_empty_classes": n_empty_classes,
                "n_dead_elements": n_dead_elements,
                "n_false_dead": n_false_dead,
                "faithful_named_precision": precision,
                "faithful_named_recall": recall,
                "faithful_named": (precision, recall),
                "exact": exact,
                "exact_tol1e-4": exact_tol1e4,
            }
