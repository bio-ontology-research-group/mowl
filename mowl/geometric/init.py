"""Exact-model initialisation and anchor regularisation for learnable box embeddings.

Provides :class:`ExactBoxInit` (Spec §3, R1), which initialises the class boxes
of axis-parallel box-embedding modules (:class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`,
:class:`~mowl.nn.el.elbe.module.ELBEModule`, and ``MultiBoxELModule``) from the
canonical model of a normalised :math:`\\mathcal{EL}^{++}` ontology, and computes
an unweighted :math:`L_2^2` anchor loss pulling class centres and half-widths
back towards the initial configuration during training.

Guarantees:
- **Containment guarantee (projected mode, k = 1):** When ``method="projected"``
  and ``n_components=1``, every canonical domain element :math:`e \\in \\Delta^{\\mathcal{I}}`
  is mapped to a Gaussian point :math:`p_e = \\text{scale} \\cdot g_e / \\sqrt{d} \\in \\mathbb{R}^d`
  (:math:`g_e \\sim \\mathcal{N}(0, I_d)`), and each satisfiable concept name
  :math:`C` with extension :math:`E_C \\neq \\emptyset` receives the padded
  bounding box :math:`[\\min_{e \\in E_C} p_e - \\text{pad},\\; \\max_{e \\in E_C} p_e + \\text{pad}]`.
  Because :math:`\\mathcal{O} \\models C \\sqsubseteq D \\implies E_C \\subseteq E_D`
  and bounding boxes are monotone under set inclusion, :math:`\\text{box}(C) \\subseteq \\text{box}(D)`
  holds for every entailed atomic subsumption with satisfiable :math:`C`.
  Consequently, every ``gci0`` training axiom has zero :class:`BoxSquaredELModule`
  and :class:`ELBEModule` loss at margin :math:`0` immediately after
  :meth:`ExactBoxInit.apply_to_module`.
- **Degenerate point boxes for unsatisfiable concepts:** ``BOTTOM``
  (``owl:Nothing``) and any concept name in the canonical model's signature that
  is unsatisfiable (:math:`E_C = \\emptyset`) are initialised to the point box
  at the origin (``center = 0``, ``offset = 0`` across all :math:`k` components).
- **Unknown classes skipped:** Concept names present in ``class_to_id`` (for
  example, classes occurring only in validation or test splits) that are absent
  from the canonical model's signature receive ``NaN`` across their row in
  ``centers`` and ``offsets``, are counted in ``report["n_skipped"]``, are left
  untouched by :meth:`ExactBoxInit.apply_to_module` (retaining their random
  initialisation without introducing ``NaN`` parameters), and are excluded from
  :meth:`ExactBoxInit.anchor_loss`.
- **Small-extension duplication (k > 1):** When ``n_components > 1`` in
  ``"projected"`` mode, deterministic k-means (``scipy.cluster.vq.kmeans2`` with
  ``minit="++"`` and ``seed``) partitions :math:`\\{p_e : e \\in E_C\\}` into
  :math:`k' = \\min(k, |E_C|)` clusters, forms one padded bounding box per
  cluster, and duplicates the first component box into the remaining
  :math:`k - k'` components. In particular, when :math:`|E_C| = 1`, all
  :math:`k` components are identical and equal to the :math:`k = 1` box.
- **MILP mode and fallback:** When ``method="milp"``, :class:`~mowl.geometric.boxes.BoxModelBuilder`
  searches for a verified :math:`d'`-dimensional box model (:math:`d' \\le d`).
  If found, coordinates :math:`0 .. d' - 1` hold the MILP boxes unchanged (preserving
  both containment and disjointness) and coordinates :math:`d' .. d - 1` hold the
  shared interval :math:`[-1 - \\text{pad},\\; 1 + \\text{pad}]`. On
  :class:`~mowl.geometric.errors.NotBoxRepresentable` or
  :class:`~mowl.geometric.errors.BoxModelNotFound`, construction falls back to
  ``"projected"`` and records ``report["fallback"] = "<ExceptionClass>: <message>"``.

Non-guarantees:
- **Disjointness under projected initialisation:** Axis-parallel bounding boxes
  of projected simplex vertices intersect strictly more often than the simplex
  faces themselves (and satisfy Helly number 2 for :math:`k = 1`), so
  ``method="projected"`` does *not* guarantee disjoint boxes for disjoint
  concepts :math:`A \\sqcap B \\sqsubseteq \\bot`.
- **MultiBoxEL duck-typed branch:** :meth:`ExactBoxInit.apply_to_module` and
  :meth:`ExactBoxInit.anchor_loss` support ``MultiBoxELModule`` via duck typing
  (checking ``hasattr(module, "n_components")`` and expecting ``class_center`` /
  ``class_offset`` of width ``n_components * embed_dim`` with flattened
  ``(k, d)`` row layout); integration tests with ``MultiBoxELModule`` are run in
  the MultiBoxEL tasks (Tasks 3–4).
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
import torch

from mowl.geometric.boxes import BoxModelBuilder
from mowl.geometric.errors import (
    BoxModelNotFound,
    NotBoxRepresentable,
    UnsupportedRegion,
)
from mowl.geometric.normal_forms import BOTTOM, TOP

if TYPE_CHECKING:  # pragma: no cover
    from mowl.datasets import Dataset
    from mowl.geometric.completion import CanonicalModel


def _iri_key(key: Any) -> str:
    """Convert a string IRI or OWLAPI ``OWLClass`` key into an unbracketed IRI string."""
    if isinstance(key, str):
        if key.startswith("<") and key.endswith(">"):
            return key[1:-1]
        return key
    if hasattr(key, "toStringID"):
        return str(key.toStringID())
    s = str(key)
    if s.startswith("<") and s.endswith(">"):
        return s[1:-1]
    return s


class _ClassToIdDict(dict):
    """``Dict[str, int]`` that also accepts ``OWLClass`` keys on lookup."""

    def __getitem__(self, key: Any) -> int:
        return super().__getitem__(_iri_key(key))

    def __contains__(self, key: Any) -> bool:
        return super().__contains__(_iri_key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(_iri_key(key), default)


def _normalise_class_to_id(class_to_id: Mapping[Any, int]) -> Dict[str, int]:
    """Normalise ``class_to_id`` keys to IRI strings and validate contiguous ``0..n-1`` indices."""
    norm: Dict[str, int] = _ClassToIdDict()
    for raw_k, raw_idx in class_to_id.items():
        if isinstance(raw_idx, bool) or not isinstance(raw_idx, (int, np.integer)):
            raise ValueError(
                f"class_to_id values must be integers, got {raw_idx!r}"
            )
        norm[_iri_key(raw_k)] = int(raw_idx)

    n_classes = len(norm)
    if set(norm.values()) != set(range(n_classes)):
        raise ValueError(
            f"class_to_id values must be a permutation of 0..{n_classes - 1}, "
            f"got {list(norm.values())!r}"
        )
    return norm


@dataclass(frozen=True)
class ExactBoxInit:
    """Exact-model box initialisation and anchor state for :math:`\\mathcal{EL}` box modules.

    Parameters
    ----------
    centers : np.ndarray
        Float32 array of shape ``(n_classes, k, d)``. Rows for classes absent
        from the canonical model contain ``NaN``.
    offsets : np.ndarray
        Float32 array of shape ``(n_classes, k, d)`` with ``offsets >= 0`` on
        all non-``NaN`` entries.
    class_to_id : Dict[str, int]
        Mapping from concept IRI strings to contiguous embedding row indices
        ``0 .. n_classes - 1``.
    method : str
        Actual initialisation method used: ``"projected"`` or ``"milp"``.
    n_components : int
        Number of box components :math:`k \\ge 1` per class.
    seed : Optional[int]
        Random seed used for Gaussian projection and k-means clustering.
    report : Dict[str, Any]
        Diagnostic dictionary recording ``n_classes``, ``n_canonical_names``,
        ``n_initialised``, ``n_skipped``, ``embed_dim``, ``n_components``,
        ``elapsed_s``, and optional ``fallback`` or ``milp_dimension``.
    """

    centers: np.ndarray
    offsets: np.ndarray
    class_to_id: Dict[str, int]
    method: str
    n_components: int
    seed: Optional[int]
    report: Dict[str, Any]

    @property
    def embed_dim(self) -> int:
        """Ambient embedding dimension :math:`d`."""
        return int(self.centers.shape[2])

    @classmethod
    def from_canonical_model(
        cls,
        cm: CanonicalModel,
        class_to_id: Mapping[Any, int],
        embed_dim: int,
        *,
        n_components: int = 1,
        method: str = "projected",
        seed: Optional[int] = 0,
        pad: float = 0.05,
        scale: float = 1.0,
        milp_time_limit: float = 60.0,
        milp_d_max: Optional[int] = None,
    ) -> ExactBoxInit:
        """Construct an :class:`ExactBoxInit` from a :class:`~mowl.geometric.completion.CanonicalModel`.

        Parameters
        ----------
        cm : CanonicalModel
            Canonical model of the training ontology.
        class_to_id : Mapping[Any, int]
            Mapping from concept IRIs (or ``OWLClass`` objects) to row indices
            ``0 .. n_classes - 1``.
        embed_dim : int
            Target Euclidean dimension :math:`d \\ge 1`.
        n_components : int, default 1
            Number of box components :math:`k \\ge 1` per class.
        method : str, default "projected"
            ``"projected"`` (random Gaussian projection of canonical elements)
            or ``"milp"`` (:class:`~mowl.geometric.boxes.BoxModelBuilder` with
            automatic fallback to ``"projected"`` on
            :class:`~mowl.geometric.errors.NotBoxRepresentable` or
            :class:`~mowl.geometric.errors.BoxModelNotFound`).
        seed : Optional[int], default 0
            Seed for ``numpy.random.default_rng(seed)`` and ``kmeans2``.
        pad : float, default 0.05
            Non-negative half-width padding added around element bounding boxes
            in ``"projected"`` mode and around unused coordinates in ``"milp"``
            mode.
        scale : float, default 1.0
            Positive scaling factor for Gaussian element coordinates
            :math:`p_e = \\text{scale} \\cdot g_e / \\sqrt{d}`.
        milp_time_limit : float, default 60.0
            Time limit in seconds forwarded to :class:`~mowl.geometric.boxes.BoxModelBuilder`
            when ``method="milp"``.
        milp_d_max : Optional[int], default None
            Maximum MILP dimension (defaults to ``embed_dim``, capped at ``embed_dim``).

        Returns
        -------
        ExactBoxInit
        """
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim < 1:
            raise ValueError(f"embed_dim must be an int >= 1, got {embed_dim!r}")
        if (
            isinstance(n_components, bool)
            or not isinstance(n_components, int)
            or n_components < 1
        ):
            raise ValueError(f"n_components must be an int >= 1, got {n_components!r}")
        if method not in ("projected", "milp"):
            raise ValueError(
                f"method must be 'projected' or 'milp', got {method!r}"
            )
        pad_f = float(pad)
        if pad_f < 0.0:
            raise ValueError(f"pad must be >= 0, got {pad!r}")
        scale_f = float(scale)
        if scale_f <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale!r}")

        t0 = time.monotonic()
        c2i = _normalise_class_to_id(class_to_id)
        d = int(embed_dim)
        k = int(n_components)

        fallback_msg: Optional[str] = None
        actual_method = method
        milp_dim: Optional[int] = None

        if method == "milp":
            eff_d_max = (
                d
                if milp_d_max is None
                else min(int(milp_d_max), d)
            )
            try:
                box_model = BoxModelBuilder(
                    cm,
                    d_max=eff_d_max,
                    time_limit=float(milp_time_limit),
                ).build()
            except (NotBoxRepresentable, BoxModelNotFound) as exc:
                fallback_msg = f"{type(exc).__name__}: {exc}"
                actual_method = "projected"
            else:
                centers, offsets, n_init, n_skip = cls._build_from_milp(
                    cm=cm,
                    box_model=box_model,
                    c2i=c2i,
                    d=d,
                    k=k,
                    pad=pad_f,
                )
                milp_dim = int(box_model.dimension)

        if actual_method == "projected":
            centers, offsets, n_init, n_skip = cls._build_projected(
                cm=cm,
                c2i=c2i,
                d=d,
                k=k,
                seed=seed,
                pad=pad_f,
                scale=scale_f,
            )

        elapsed = time.monotonic() - t0
        report: Dict[str, Any] = {
            "n_classes": len(c2i),
            "n_canonical_names": len(cm.names),
            "n_initialised": n_init,
            "n_skipped": n_skip,
            "embed_dim": d,
            "n_components": k,
            "requested_method": method,
            "method": actual_method,
            "elapsed_s": elapsed,
        }
        if fallback_msg is not None:
            report["fallback"] = fallback_msg
        if milp_dim is not None:
            report["milp_dimension"] = milp_dim

        return cls(
            centers=centers,
            offsets=offsets,
            class_to_id=c2i,
            method=actual_method,
            n_components=k,
            seed=seed,
            report=report,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        embed_dim: int,
        **kw: Any,
    ) -> ExactBoxInit:
        """Build a :class:`~mowl.geometric.completion.CanonicalModel` from ``dataset`` and return its :class:`ExactBoxInit`."""
        from mowl.geometric.completion import CanonicalModel

        cm = CanonicalModel.from_dataset(dataset)
        return cls.from_canonical_model(cm, dataset.class_to_id, embed_dim, **kw)

    @staticmethod
    def _build_projected(
        cm: CanonicalModel,
        c2i: Dict[str, int],
        d: int,
        k: int,
        seed: Optional[int],
        pad: float,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Compute ``(centers, offsets, n_initialised, n_skipped)`` for ``method='projected'``."""
        n_classes = len(c2i)
        centers = np.full((n_classes, k, d), np.nan, dtype=np.float32)
        offsets = np.full((n_classes, k, d), np.nan, dtype=np.float32)

        rng = np.random.default_rng(seed)
        n_elements = len(cm.names)
        if n_elements > 0:
            g = rng.standard_normal((n_elements, d))
            p = ((scale / np.sqrt(float(d))) * g).astype(np.float64)
        else:
            p = np.zeros((0, d), dtype=np.float64)

        known_concepts = set(cm.concept_names) | set(cm.names) | {TOP, BOTTOM}
        n_init = 0
        n_skip = 0

        for name, idx in c2i.items():
            if name not in known_concepts:
                n_skip += 1
                continue

            n_init += 1
            if name == BOTTOM or not cm.is_satisfiable(name):
                centers[idx, :, :] = 0.0
                offsets[idx, :, :] = 0.0
                continue

            if name == TOP:
                if n_elements == 0:
                    centers[idx, :, :] = 0.0
                    offsets[idx, :, :] = 0.0
                else:
                    lo = np.min(p, axis=0) - pad
                    hi = np.max(p, axis=0) + pad
                    c_vec = (0.5 * (lo + hi)).astype(np.float32)
                    o_vec = np.maximum(0.0, 0.5 * (hi - lo)).astype(np.float32)
                    centers[idx, :, :] = c_vec
                    offsets[idx, :, :] = o_vec
                continue

            ext = sorted(cm.extension(name))
            if not ext:
                centers[idx, :, :] = 0.0
                offsets[idx, :, :] = 0.0
                continue

            pts = p[ext]
            k_prime = min(k, len(ext))
            if k_prime == 1:
                lo = np.min(pts, axis=0) - pad
                hi = np.max(pts, axis=0) + pad
                c_vec = (0.5 * (lo + hi)).astype(np.float32)
                o_vec = np.maximum(0.0, 0.5 * (hi - lo)).astype(np.float32)
                centers[idx, :, :] = c_vec
                offsets[idx, :, :] = o_vec
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    _, labels = kmeans2(pts, k_prime, minit="++", seed=seed)

                comp_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
                for cluster_id in range(k_prime):
                    mask_c = labels == cluster_id
                    if np.any(mask_c):
                        c_pts = pts[mask_c]
                        lo = np.min(c_pts, axis=0) - pad
                        hi = np.max(c_pts, axis=0) + pad
                        c_vec = (0.5 * (lo + hi)).astype(np.float32)
                        o_vec = np.maximum(0.0, 0.5 * (hi - lo)).astype(np.float32)
                        comp_boxes.append((c_vec, o_vec))

                if not comp_boxes:  # pragma: no cover
                    lo = np.min(pts, axis=0) - pad
                    hi = np.max(pts, axis=0) + pad
                    comp_boxes.append(
                        (
                            (0.5 * (lo + hi)).astype(np.float32),
                            np.maximum(0.0, 0.5 * (hi - lo)).astype(np.float32),
                        )
                    )

                while len(comp_boxes) < k:
                    comp_boxes.append(comp_boxes[0])

                for comp_idx in range(k):
                    centers[idx, comp_idx, :] = comp_boxes[comp_idx][0]
                    offsets[idx, comp_idx, :] = comp_boxes[comp_idx][1]

        return centers, offsets, n_init, n_skip

    @staticmethod
    def _build_from_milp(
        cm: CanonicalModel,
        box_model: Any,
        c2i: Dict[str, int],
        d: int,
        k: int,
        pad: float,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Compute ``(centers, offsets, n_initialised, n_skipped)`` from a solved :class:`BoxModel`."""
        n_classes = len(c2i)
        centers = np.full((n_classes, k, d), np.nan, dtype=np.float32)
        offsets = np.full((n_classes, k, d), np.nan, dtype=np.float32)

        d_prime = int(box_model.dimension)
        model_boxes = box_model.boxes()
        known_concepts = set(cm.concept_names) | set(cm.names) | {TOP, BOTTOM}

        n_init = 0
        n_skip = 0

        for name, idx in c2i.items():
            if name not in known_concepts:
                n_skip += 1
                continue

            n_init += 1
            if name == BOTTOM or not cm.is_satisfiable(name) or name not in model_boxes:
                centers[idx, :, :] = 0.0
                offsets[idx, :, :] = 0.0
                continue

            lo_m, hi_m = model_boxes[name]
            lo = np.full(d, -1.0 - pad, dtype=np.float64)
            hi = np.full(d, 1.0 + pad, dtype=np.float64)
            lo[:d_prime] = lo_m[:d_prime]
            hi[:d_prime] = hi_m[:d_prime]

            c_vec = (0.5 * (lo + hi)).astype(np.float32)
            o_vec = np.maximum(0.0, 0.5 * (hi - lo)).astype(np.float32)
            centers[idx, :, :] = c_vec
            offsets[idx, :, :] = o_vec

        return centers, offsets, n_init, n_skip

    def as_tensors(
        self, device: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(centers, offsets, mask)`` as PyTorch tensors on ``device``.

        Returns
        -------
        centers : torch.Tensor
            Float32 tensor of shape ``(n_classes, k, d)``.
        offsets : torch.Tensor
            Float32 tensor of shape ``(n_classes, k, d)``.
        mask : torch.Tensor
            Boolean tensor of shape ``(n_classes,)``, ``True`` where the class
            row was initialised (not ``NaN``).
        """
        mask_np = ~np.isnan(self.centers).any(axis=(1, 2))
        centers_t = torch.from_numpy(self.centers.copy()).to(dtype=torch.float32)
        offsets_t = torch.from_numpy(self.offsets.copy()).to(dtype=torch.float32)
        mask_t = torch.from_numpy(mask_np)
        if device is not None:
            centers_t = centers_t.to(device=device)
            offsets_t = offsets_t.to(device=device)
            mask_t = mask_t.to(device=device)
        return centers_t, offsets_t, mask_t

    def _resolve_module_embeddings(
        self, module: Any
    ) -> Tuple[torch.nn.Embedding, torch.nn.Embedding, str]:
        """Inspect ``module`` and return ``(center_embedding, offset_embedding, layout)``."""
        n_classes = int(self.centers.shape[0])
        k = int(self.n_components)
        d = int(self.embed_dim)

        if hasattr(module, "n_components"):
            if not (hasattr(module, "class_center") and hasattr(module, "class_offset")):
                raise UnsupportedRegion(
                    f"Module {type(module).__name__!r} declares n_components but lacks "
                    f"class_center/class_offset embeddings."
                )
            mod_k = int(getattr(module, "n_components"))
            mod_d = int(getattr(module, "embed_dim", -1))
            if mod_k != k or (mod_d != -1 and mod_d != d):
                raise UnsupportedRegion(
                    f"Dimension/component mismatch: ExactBoxInit has (k={k}, d={d}), "
                    f"module has (n_components={mod_k}, embed_dim={mod_d})."
                )
            c_emb = module.class_center
            o_emb = module.class_offset
            expected_shape = (n_classes, k * d)
            if (
                tuple(c_emb.weight.shape) != expected_shape
                or tuple(o_emb.weight.shape) != expected_shape
            ):
                raise UnsupportedRegion(
                    f"Embedding weight shape mismatch for MultiBoxEL layout: expected "
                    f"{expected_shape}, got {tuple(c_emb.weight.shape)} and "
                    f"{tuple(o_emb.weight.shape)}."
                )
            return c_emb, o_emb, "multiboxel"

        if hasattr(module, "class_center"):
            if not hasattr(module, "class_offset"):
                raise UnsupportedRegion(
                    f"Module {type(module).__name__!r} has class_center but lacks class_offset."
                )
            c_emb = module.class_center
            o_emb = module.class_offset
        elif hasattr(module, "class_embed"):
            if not hasattr(module, "class_offset"):
                raise UnsupportedRegion(
                    f"Module {type(module).__name__!r} has class_embed but lacks class_offset."
                )
            c_emb = module.class_embed
            o_emb = module.class_offset
        else:
            raise UnsupportedRegion(
                f"Module {type(module).__name__!r} exposes neither "
                f"(class_center, class_offset) nor (class_embed, class_offset)."
            )

        if k != 1:
            raise UnsupportedRegion(
                f"Single-box module {type(module).__name__!r} requires "
                f"ExactBoxInit.n_components == 1, got n_components={k}."
            )
        mod_d = int(getattr(module, "embed_dim", c_emb.weight.shape[1]))
        expected_shape = (n_classes, d)
        if (
            mod_d != d
            or tuple(c_emb.weight.shape) != expected_shape
            or tuple(o_emb.weight.shape) != expected_shape
        ):
            raise UnsupportedRegion(
                f"Embedding shape mismatch for {type(module).__name__!r}: expected "
                f"{expected_shape}, got {tuple(c_emb.weight.shape)} and "
                f"{tuple(o_emb.weight.shape)}."
            )
        return c_emb, o_emb, "single"

    def apply_to_module(
        self, module: Any, *, device: Optional[Any] = None
    ) -> int:
        """Write initialised class boxes into ``module`` under ``torch.no_grad()``.

        Supports :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`
        (``class_center``, ``class_offset``),
        :class:`~mowl.nn.el.elbe.module.ELBEModule` (``class_embed``,
        ``class_offset``), and ``MultiBoxELModule`` via duck typing
        (``hasattr(module, "n_components")`` with ``class_center`` and
        ``class_offset`` of width ``n_components * embed_dim`` storing flattened
        ``(k, d)`` rows; tested with ``MultiBoxELModule`` in Tasks 3–4).
        Rows where ``centers`` is ``NaN`` are left untouched.

        Parameters
        ----------
        module : torch.nn.Module
            Target box-embedding module.
        device : Optional[Any], default None
            Optional target device override.

        Returns
        -------
        int
            Number of initialised class rows written into ``module``.

        Raises
        ------
        UnsupportedRegion
            If ``module`` does not expose supported box embeddings or its
            dimensions / component count do not match ``self``.
        """
        c_emb, o_emb, layout = self._resolve_module_embeddings(module)
        if device is not None:
            module.to(device)
        target_device = c_emb.weight.device
        target_dtype = c_emb.weight.dtype

        centers_t, offsets_t, mask_t = self.as_tensors(device=target_device)
        centers_t = centers_t.to(dtype=target_dtype)
        offsets_t = offsets_t.to(dtype=target_dtype)

        n_classes = int(self.centers.shape[0])
        k = int(self.n_components)
        d = int(self.embed_dim)

        with torch.no_grad():
            if layout == "multiboxel":
                flat_c = centers_t.reshape(n_classes, k * d)
                flat_o = offsets_t.reshape(n_classes, k * d)
                c_emb.weight[mask_t] = flat_c[mask_t]
                o_emb.weight[mask_t] = flat_o[mask_t]
            else:
                c_emb.weight[mask_t] = centers_t[mask_t, 0, :]
                o_emb.weight[mask_t] = offsets_t[mask_t, 0, :]

        return int(mask_t.sum().item())

    def anchor_loss(self, module: Any) -> torch.Tensor:
        """Return unweighted mean :math:`\\|c - c_0\\|_2^2 + \\||o| - o_0\\|_2^2` over initialised rows.

        Uninitialised (``NaN``) rows are excluded from the slice so they neither
        contribute ``NaN`` values nor receive gradients from the anchor.

        Parameters
        ----------
        module : torch.nn.Module
            Target box-embedding module.

        Returns
        -------
        torch.Tensor
            Scalar unweighted anchor loss tensor on ``module``'s device.
        """
        c_emb, o_emb, layout = self._resolve_module_embeddings(module)
        target_device = c_emb.weight.device
        target_dtype = c_emb.weight.dtype

        centers_t, offsets_t, mask_t = self.as_tensors(device=target_device)
        centers_t = centers_t.to(dtype=target_dtype)
        offsets_t = offsets_t.to(dtype=target_dtype)

        if not bool(torch.any(mask_t).item()):
            return c_emb.weight.sum() * 0.0

        n_classes = int(self.centers.shape[0])
        k = int(self.n_components)
        d = int(self.embed_dim)

        if layout == "multiboxel":
            c_ref = centers_t.reshape(n_classes, k * d)[mask_t]
            o_ref = offsets_t.reshape(n_classes, k * d)[mask_t]
        else:
            c_ref = centers_t[mask_t, 0, :]
            o_ref = offsets_t[mask_t, 0, :]

        c_cur = c_emb.weight[mask_t]
        o_cur = torch.abs(o_emb.weight[mask_t])

        row_loss = (c_cur - c_ref).square().sum(dim=-1) + (
            o_cur - o_ref
        ).square().sum(dim=-1)
        return row_loss.mean()
