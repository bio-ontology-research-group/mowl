"""Exact-model initialisation and anchor regularisation for learnable box embeddings.

Provides :class:`ExactBoxInit`, which initialises the class boxes
of axis-parallel box-embedding modules (:class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`,
:class:`~mowl.nn.el.elbe.module.ELBEModule`, and ``MultiBoxELModule``) and the
class prisms of :class:`~mowl.nn.el.prism.module.PrismModule` from the
canonical model of a normalised :math:`\\mathcal{EL}^{++}` ontology, and computes
an unweighted :math:`L_2^2` anchor loss pulling class centres and half-widths
(or triangle centres, inradii, and box centres/half-widths) back towards the
initial configuration during training.

Guarantees:
- **Containment guarantee (projected mode, k = 1):** When ``method="projected"``
  and ``n_components=1``, every canonical domain element :math:`e \\in \\Delta^{\\mathcal{I}}`
  is mapped to a Gaussian point :math:`p_e = \\text{scale} \\cdot g_e / \\sqrt{d} \\in \\mathbb{R}^d`
  (:math:`g_e \\sim \\mathcal{N}(0, I_d)`), and each satisfiable concept name
  :math:`C` with extension :math:`E_C \\neq \\emptyset` receives the padded
  bounding box :math:`[\\min_{e \\in E_C} p_e - \\text{pad},\\; \\max_{e \\in E_C} p_e + \\text{pad}]`.
  Because :math:`\\mathcal{O} \\models C \\sqsubseteq D \\implies E_C \\subseteq E_D`
  and bounding boxes are monotone under set inclusion, :math:`\\text{box}(C) \\subseteq \\text{box}(D)`
  holds exactly (up to ``float32`` rounding; tests use ``1e-6``) for every
  entailed atomic subsumption with satisfiable :math:`C`.
  Consequently, every ``gci0`` training axiom whose subclass is satisfiable has
  zero :class:`BoxSquaredELModule` and :class:`ELBEModule` loss (up to
  ``float32`` rounding, ``<= 1e-6``) at margin :math:`0` immediately after
  :meth:`ExactBoxInit.apply_to_module`.
- **Prism box-containment guarantee (k = 1, d >= 3):** When applied to a module
  with ``tri_center`` (:class:`~mowl.nn.el.prism.module.PrismModule`),
  coordinates :math:`2 .. d - 1` of the initialisation box are written to
  ``box_center`` and ``box_offset``, and coordinates :math:`0 .. 1` define the
  equilateral triangle with ``tri_center = center[:2]`` and
  ``tri_inradius = max(offset[0], offset[1]) * sqrt(3)`` (the incircle radius of
  the equilateral triangle circumscribing the rectangle's bounding square).
  Every corner of the initialisation box therefore lies inside the initialised
  prism (up to ``float32`` rounding, ``<= 1e-6``), and :meth:`ExactBoxInit.anchor_loss`
  compares ``(tri_center, |tri_inradius|, box_center, |box_offset|)`` against
  these same initial quantities.
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
  non-empty cluster, records in ``report["n_degenerate_kmeans"]`` the count of
  classes where k-means produced fewer than :math:`k'` non-empty clusters, and
  duplicates the first component box into the remaining components. In
  particular, when :math:`|E_C| = 1`, all :math:`k` components are identical and
  equal to the :math:`k = 1` box.
- **MILP mode and fallback:** When ``method="milp"``, :class:`~mowl.geometric.boxes.BoxModelBuilder`
  searches for a verified :math:`d'`-dimensional box model (:math:`d' \\le d`).
  If found, coordinates :math:`0 .. d' - 1` hold the MILP boxes mapped from the
  solver's universe :math:`[0, M]` onto :math:`[-\\text{scale}, \\text{scale}]`
  by the coordinate-wise affine map
  :math:`x \\mapsto \\text{scale} \\cdot (2x / M - 1)` (so that they live on the
  same unit scale as the padding, the role boxes and the bumps; an increasing
  affine map preserves both containment and disjointness), and coordinates
  :math:`d' .. d - 1` hold the shared interval
  :math:`[-\\text{scale} - \\text{pad},\\; \\text{scale} + \\text{pad}]`. On
  :class:`~mowl.geometric.errors.NotBoxRepresentable` or
  :class:`~mowl.geometric.errors.BoxModelNotFound`, construction falls back to
  ``"projected"`` and records ``report["fallback"] = "<ExceptionClass>: <message>"``.
- **Row matching by IRI:** Initialisation rows are indexed by the dataset's
  ``class_to_id``. :meth:`ExactBoxInit.apply_to_module` matches them to module
  rows through the ``class_index_dict`` passed by the model (the dataset
  classes followed by the auxiliary classes created by normalisation, with
  IRIs under ``http://mowl.borg/aux/``). Module rows without an
  initialisation entry, in particular all auxiliary classes, keep their
  random initialisation and are excluded from
  :meth:`ExactBoxInit.anchor_loss`.

Non-guarantees:
- **Unsatisfiable-subclass ``gci0`` axioms:** Axioms :math:`U \\sqsubseteq A`
  whose subclass :math:`U` is unsatisfiable are not covered by the zero-loss
  ``gci0`` guarantee: unsatisfiable names are placed as points at the origin,
  which in general lies outside :math:`\\text{box}(A)`, and no point placement
  can cover them in general (e.g. :math:`U \\sqsubseteq A` and
  :math:`U \\sqsubseteq B` with :math:`A \\sqcap B \\sqsubseteq \\bot` have no
  common point).
- **Disjointness under projected initialisation:** Axis-parallel bounding boxes
  of projected simplex vertices intersect strictly more often than the simplex
  faces themselves (and satisfy Helly number 2 for :math:`k = 1`), so
  ``method="projected"`` does *not* guarantee disjoint boxes for disjoint
  concepts :math:`A \\sqcap B \\sqsubseteq \\bot`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
import torch

from mowl.geometric.boxes import BoxModelBuilder
from mowl.geometric.errors import (
    BoxModelNotFound,
    InternalConsistencyError,
    NotBoxRepresentable,
    UnsupportedRegion,
)
from mowl.geometric.normal_forms import BOTTOM, TOP

if TYPE_CHECKING:  # pragma: no cover
    from mowl.datasets import Dataset
    from mowl.geometric.completion import CanonicalModel


def _padded_bbox(pts: np.ndarray, pad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(center, offset)`` float32 vectors for the ``pad``-expanded bounding box of ``pts``."""
    lo = np.min(pts, axis=0) - pad
    hi = np.max(pts, axis=0) + pad
    c_vec = (0.5 * (lo + hi)).astype(np.float32)
    o_vec = np.maximum(0.0, 0.5 * (hi - lo)).astype(np.float32)
    return c_vec, o_vec


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


@dataclass(frozen=True, eq=False)
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
        Mapping from concept IRI strings to contiguous row indices
        ``0 .. n_classes - 1`` of ``centers`` and ``offsets`` (typically
        ``dataset.class_to_id``). These rows are matched to the rows of a
        module by IRI in :meth:`apply_to_module`.
    method : str
        Actual initialisation method used: ``"projected"`` or ``"milp"``.
    n_components : int
        Number of box components :math:`k \\ge 1` per class.
    seed : Optional[int]
        Random seed used for Gaussian projection and k-means clustering.
    report : Dict[str, Any]
        Diagnostic dictionary recording ``n_classes``, ``n_canonical_names``,
        ``n_initialised``, ``n_skipped``, ``n_degenerate_kmeans``,
        ``embed_dim``, ``n_components``, ``elapsed_s``, and optional
        ``fallback`` or ``milp_dimension``.
    """

    centers: np.ndarray
    offsets: np.ndarray
    class_to_id: Dict[str, int]
    method: str
    n_components: int
    seed: Optional[int]
    report: Dict[str, Any]
    _row_maps: Dict[int, Tuple[int, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _ref_cache: Dict[Tuple[str, str, Any], Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

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

        With ``method="projected"`` and ``n_components=1``, for every entailed
        atomic subsumption :math:`C \\sqsubseteq D` with :math:`C` satisfiable,
        :math:`\\text{box}(C) \\subseteq \\text{box}(D)` holds exactly (up to
        ``float32`` rounding; tests use ``1e-6``), so every ``gci0`` training
        axiom whose subclass is satisfiable has zero loss at margin :math:`0`
        immediately after :meth:`apply_to_module`. Axioms whose subclass is
        unsatisfiable are not covered (unsatisfiable names are points at the
        origin), and no placement can cover them in general.

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
            :math:`p_e = \\text{scale} \\cdot g_e / \\sqrt{d}` in
            ``"projected"`` mode; in ``"milp"`` mode the MILP universe
            :math:`[0, M]` is mapped affinely onto
            :math:`[-\\text{scale}, \\text{scale}]` in every coordinate.
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
        n_deg_kmeans = 0

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
                    scale=scale_f,
                )
                milp_dim = int(box_model.dimension)

        if actual_method == "projected":
            centers, offsets, n_init, n_skip, n_deg_kmeans = cls._build_projected(
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
            "n_degenerate_kmeans": n_deg_kmeans,
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
    ) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        """Compute ``(centers, offsets, n_initialised, n_skipped, n_degenerate_kmeans)`` for ``method='projected'``."""
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
        n_deg_kmeans = 0

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
                    c_vec, o_vec = _padded_bbox(p, pad)
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
                c_vec, o_vec = _padded_bbox(pts, pad)
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
                        comp_boxes.append(_padded_bbox(pts[mask_c], pad))

                if len(comp_boxes) < k_prime:
                    n_deg_kmeans += 1

                if not comp_boxes:  # pragma: no cover
                    comp_boxes.append(_padded_bbox(pts, pad))

                while len(comp_boxes) < k:
                    comp_boxes.append(comp_boxes[0])

                for comp_idx in range(k):
                    centers[idx, comp_idx, :] = comp_boxes[comp_idx][0]
                    offsets[idx, comp_idx, :] = comp_boxes[comp_idx][1]

        return centers, offsets, n_init, n_skip, n_deg_kmeans

    @staticmethod
    def _build_from_milp(
        cm: CanonicalModel,
        box_model: Any,
        c2i: Dict[str, int],
        d: int,
        k: int,
        pad: float,
        scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Compute ``(centers, offsets, n_initialised, n_skipped)`` from a solved :class:`BoxModel`.

        MILP coordinates in :math:`[0, M]` are mapped to
        :math:`\\text{scale} \\cdot (2x / M - 1) \\in [-\\text{scale}, \\text{scale}]`.
        """
        n_classes = len(c2i)
        centers = np.full((n_classes, k, d), np.nan, dtype=np.float32)
        offsets = np.full((n_classes, k, d), np.nan, dtype=np.float32)

        d_prime = int(box_model.dimension)
        model_boxes = box_model.boxes()
        big_m = float(box_model.M)
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

            if name not in model_boxes:
                raise InternalConsistencyError(
                    f"Satisfiable concept {name!r} is missing from BoxModel.boxes()."
                )

            lo_m, hi_m = model_boxes[name]
            lo = np.full(d, -scale - pad, dtype=np.float64)
            hi = np.full(d, scale + pad, dtype=np.float64)
            lo[:d_prime] = scale * (2.0 * np.asarray(lo_m[:d_prime], dtype=np.float64) / big_m - 1.0)
            hi[:d_prime] = scale * (2.0 * np.asarray(hi_m[:d_prime], dtype=np.float64) / big_m - 1.0)

            c_vec, o_vec = _padded_bbox(np.stack([lo, hi], axis=0), 0.0)
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
        """Inspect ``module`` and return ``(center_embedding, offset_embedding, layout)``.

        Checks the embedding widths against ``(n_components, embed_dim)``; the
        number of rows is checked against the row map in :meth:`_row_map_for`.
        """
        k = int(self.n_components)
        d = int(self.embed_dim)

        if hasattr(module, "tri_center"):
            if not (
                hasattr(module, "tri_inradius")
                and hasattr(module, "box_center")
                and hasattr(module, "box_offset")
            ):
                raise UnsupportedRegion(
                    f"Module {type(module).__name__!r} has tri_center but lacks "
                    f"tri_inradius/box_center/box_offset embeddings."
                )
            if k != 1:
                raise UnsupportedRegion(
                    f"Prism module {type(module).__name__!r} requires "
                    f"ExactBoxInit.n_components == 1, got n_components={k}."
                )
            if d < 3:
                raise UnsupportedRegion(
                    f"Prism module {type(module).__name__!r} requires "
                    f"ExactBoxInit.embed_dim >= 3, got d={d}."
                )
            mod_d = int(
                getattr(
                    module,
                    "embed_dim",
                    int(module.tri_center.weight.shape[1])
                    + int(module.box_center.weight.shape[1]),
                )
            )
            if mod_d != d:
                raise UnsupportedRegion(
                    f"Dimension mismatch for {type(module).__name__!r}: ExactBoxInit has "
                    f"d={d}, module has embed_dim={mod_d}."
                )
            tc_shape = tuple(module.tri_center.weight.shape)
            tr_shape = tuple(module.tri_inradius.weight.shape)
            bc_shape = tuple(module.box_center.weight.shape)
            bo_shape = tuple(module.box_offset.weight.shape)
            n_rows = tc_shape[0] if len(tc_shape) == 2 else -1
            if (
                tc_shape != (n_rows, 2)
                or tr_shape != (n_rows, 1)
                or bc_shape != (n_rows, d - 2)
                or bo_shape != (n_rows, d - 2)
            ):
                raise UnsupportedRegion(
                    f"Embedding shape mismatch for {type(module).__name__!r} (prism layout): "
                    f"expected (n_rows, 2), (n_rows, 1), (n_rows, {d - 2}), (n_rows, {d - 2}), "
                    f"got {tc_shape}, {tr_shape}, {bc_shape}, {bo_shape}."
                )
            return module.tri_center, module.tri_inradius, "prism"

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
            width = k * d
            layout = "multiboxel"
        else:
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
            if mod_d != d:
                raise UnsupportedRegion(
                    f"Dimension mismatch for {type(module).__name__!r}: ExactBoxInit has "
                    f"d={d}, module has embed_dim={mod_d}."
                )
            width = d
            layout = "single"

        c_shape = tuple(c_emb.weight.shape)
        o_shape = tuple(o_emb.weight.shape)
        if len(c_shape) != 2 or c_shape[1] != width or c_shape != o_shape:
            raise UnsupportedRegion(
                f"Embedding shape mismatch for {type(module).__name__!r} ({layout} layout): "
                f"expected two (n_rows, {width}) tables, got {c_shape} and {o_shape}."
            )
        return c_emb, o_emb, layout

    def _init_rows(self) -> np.ndarray:
        """Return the sorted indices of initialised (non-``NaN``) rows of ``centers``."""
        return np.flatnonzero(~np.isnan(self.centers).any(axis=(1, 2)))

    def _build_row_map(
        self, n_module_rows: int, class_index_dict: Optional[Mapping[Any, int]]
    ) -> np.ndarray:
        """Return module row indices for the initialised rows in :meth:`_init_rows` order."""
        init_rows = self._init_rows()
        n_classes = int(self.centers.shape[0])
        if class_index_dict is None:
            if n_module_rows != n_classes:
                raise UnsupportedRegion(
                    f"Module has {n_module_rows} class rows but ExactBoxInit has "
                    f"{n_classes}; pass class_index_dict (e.g. the model's "
                    f"class_index_dict) so that rows can be matched by IRI."
                )
            return init_rows.astype(np.int64)

        module_index = {_iri_key(key): int(idx) for key, idx in class_index_dict.items()}
        id_to_name = {idx: name for name, idx in self.class_to_id.items()}
        rows = np.empty(len(init_rows), dtype=np.int64)
        missing: List[str] = []
        for pos, init_row in enumerate(init_rows):
            name = id_to_name[int(init_row)]
            mod_row = module_index.get(name)
            if mod_row is None:
                missing.append(name)
                continue
            if not 0 <= mod_row < n_module_rows:
                raise UnsupportedRegion(
                    f"class_index_dict maps {name!r} to row {mod_row}, but the module "
                    f"has {n_module_rows} class rows."
                )
            rows[pos] = mod_row
        if missing:
            raise UnsupportedRegion(
                f"{len(missing)} initialised classes are absent from class_index_dict "
                f"(e.g. {missing[:3]!r})."
            )
        return rows

    def _row_map_for(self, module: Any, n_module_rows: int) -> np.ndarray:
        """Return the cached row map for ``module`` (identity if none and sizes match)."""
        cached = self._row_maps.get(id(module))
        if cached is not None and cached[0] == n_module_rows:
            return cached[1]
        return self._build_row_map(n_module_rows, None)

    def _reference(
        self, layout: str, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(c_ref, o_ref)`` of the initialised rows on ``device``."""
        key = (layout, str(device), dtype)
        ref = self._ref_cache.get(key)
        if ref is None:
            init_rows = self._init_rows()
            k = int(self.n_components)
            d = int(self.embed_dim)
            c_np = self.centers[init_rows]
            o_np = self.offsets[init_rows]
            if layout == "multiboxel":
                c_np = c_np.reshape(len(init_rows), k * d)
                o_np = o_np.reshape(len(init_rows), k * d)
            elif layout == "prism":
                c_np = c_np[:, 0, :]
                o_first = o_np[:, 0, :]
                tri_r = (
                    np.max(o_first[:, :2], axis=-1, keepdims=True)
                    * np.sqrt(3.0)
                ).astype(np.float32)
                o_np = np.concatenate([tri_r, o_first[:, 2:]], axis=-1)
            else:
                c_np = c_np[:, 0, :]
                o_np = o_np[:, 0, :]
            ref = (
                torch.from_numpy(np.ascontiguousarray(c_np)).to(device=device, dtype=dtype),
                torch.from_numpy(np.ascontiguousarray(o_np)).to(device=device, dtype=dtype),
            )
            self._ref_cache[key] = ref
        return ref

    def apply_to_module(
        self, module: Any, class_index_dict: Optional[Mapping[Any, int]] = None
    ) -> int:
        """Write initialised class boxes or prisms into ``module`` under ``torch.no_grad()``.

        Supports :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`
        (``class_center``, ``class_offset``),
        :class:`~mowl.nn.el.elbe.module.ELBEModule` (``class_embed``,
        ``class_offset``), :class:`~mowl.nn.el.multiboxel.module.MultiBoxELModule`
        (detected by ``hasattr(module, "n_components")``; ``class_center`` and
        ``class_offset`` of width ``n_components * embed_dim`` storing flattened
        ``(k, d)`` rows), and :class:`~mowl.nn.el.prism.module.PrismModule`
        (detected by ``hasattr(module, "tri_center")``).

        When ``module`` has ``tri_center`` (prism layout), the box part
        (``box_center``, ``box_offset``) is written from coordinates ``2..d``
        of the initialisation box, and the triangle part is written from
        coordinates ``0..1``: ``tri_center`` is set to the box centre of those
        two coordinates, and ``tri_inradius`` is set to
        :math:`\\max(\\text{half-widths of coords } 0, 1) \\cdot \\sqrt{3}` (the
        incircle of the equilateral triangle circumscribing that rectangle's
        bounding square, so the initialised prism contains the initialisation
        box); :meth:`anchor_loss` compares the same quantities.

        Rows are matched by IRI: initialisation row ``class_to_id[C]`` is
        written to module row ``class_index_dict[C]``. Module rows without an
        initialisation entry -- in particular the auxiliary classes that
        :class:`~mowl.base_models.elmodel.EmbeddingELModel` appends to its
        ``class_index_dict`` for the fresh names introduced by normalisation
        (IRIs under ``http://mowl.borg/aux/``) -- keep their random
        initialisation, and so do module rows of classes whose
        initialisation row is ``NaN``. The tensors are written on the
        module's device and dtype; the module is not moved.

        The row map is cached on this object (per module) and reused by
        :meth:`anchor_loss`.

        Parameters
        ----------
        module : torch.nn.Module
            Target box-embedding or prism-embedding module.
        class_index_dict : Optional[Mapping[Any, int]], default None
            Mapping from class IRIs (or ``OWLClass`` objects) to module rows,
            typically ``model.class_index_dict``. If ``None``, the module must
            have exactly ``len(class_to_id)`` class rows, and rows are matched
            by index.

        Returns
        -------
        int
            Number of initialised class rows written into ``module``.

        Raises
        ------
        UnsupportedRegion
            If ``module`` does not expose supported box or prism embeddings, its
            dimensions / component count do not match ``self``, an initialised
            class is missing from ``class_index_dict``, or
            ``class_index_dict`` is ``None`` and the row counts differ.
        """
        c_emb, o_emb, layout = self._resolve_module_embeddings(module)
        n_module_rows = int(c_emb.weight.shape[0])
        rows_np = self._build_row_map(n_module_rows, class_index_dict)
        self._row_maps[id(module)] = (n_module_rows, rows_np)

        c_ref, o_ref = self._reference(layout, c_emb.weight.device, c_emb.weight.dtype)
        rows = torch.from_numpy(rows_np).to(device=c_emb.weight.device)
        with torch.no_grad():
            if layout == "prism":
                module.tri_center.weight[rows] = c_ref[:, :2]
                module.box_center.weight[rows] = c_ref[:, 2:]
                module.tri_inradius.weight[rows] = o_ref[:, :1]
                module.box_offset.weight[rows] = o_ref[:, 1:]
            else:
                c_emb.weight[rows] = c_ref
                o_emb.weight[rows] = o_ref

        return int(len(rows_np))

    def anchor_loss(self, module: Any) -> torch.Tensor:
        """Return unweighted mean :math:`\\|c - c_0\\|_2^2 + \\||o| - o_0\\|_2^2` over initialised rows.

        For prism modules (:class:`~mowl.nn.el.prism.module.PrismModule`),
        compares ``(tri_center, box_center)`` against the initial centres and
        ``(|tri_inradius|, |box_offset|)`` against the initial circumscribing
        inradius :math:`\\max(o_0, o_1) \\sqrt{3}` and box half-widths.

        Uses the row map cached by :meth:`apply_to_module` for ``module``
        (identity when ``module`` has not been seen and has exactly
        ``len(class_to_id)`` rows). Uninitialised (``NaN``) rows and module
        rows without an initialisation entry (auxiliary classes) are excluded,
        so they neither contribute ``NaN`` values nor receive gradients from
        the anchor. Reference tensors are cached per device and dtype.

        Parameters
        ----------
        module : torch.nn.Module
            Target box-embedding or prism-embedding module.

        Returns
        -------
        torch.Tensor
            Scalar unweighted anchor loss tensor on ``module``'s device.

        Raises
        ------
        UnsupportedRegion
            If ``module`` is incompatible, or has a different number of class
            rows than ``class_to_id`` and :meth:`apply_to_module` was not
            called on it with a ``class_index_dict``.
        """
        c_emb, o_emb, layout = self._resolve_module_embeddings(module)
        rows_np = self._row_map_for(module, int(c_emb.weight.shape[0]))
        if len(rows_np) == 0:
            return c_emb.weight.sum() * 0.0

        device = c_emb.weight.device
        c_ref, o_ref = self._reference(layout, device, c_emb.weight.dtype)
        rows = torch.from_numpy(rows_np).to(device=device)

        if layout == "prism":
            c_cur = torch.cat(
                [module.tri_center.weight[rows], module.box_center.weight[rows]],
                dim=-1,
            )
            o_cur = torch.cat(
                [
                    torch.abs(module.tri_inradius.weight[rows]),
                    torch.abs(module.box_offset.weight[rows]),
                ],
                dim=-1,
            )
        else:
            c_cur = c_emb.weight[rows]
            o_cur = torch.abs(o_emb.weight[rows])

        row_loss = (c_cur - c_ref).square().sum(dim=-1) + (
            o_cur - o_ref
        ).square().sum(dim=-1)
        return row_loss.mean()
