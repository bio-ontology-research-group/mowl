"""Conversions from constructive geometric models to array embeddings.

Provides helpers to export class regions from :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`
and :class:`~mowl.geometric.boxes.BoxModel` as NumPy arrays suitable for seeding
or inspecting mOWL embedding modules (such as :class:`~mowl.nn.el.elbe.module.ELBEModule`
/ :class:`~mowl.models.elbe.model.ELBE` and
:class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`).

Guarantees and limitations:
- :func:`class_points` returns the exact generating simplex-vertex rows for each
  concept name in a :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`.
- :func:`to_bounding_boxes` computes the axis-parallel bounding box
  ``(min(pts, axis=0), max(pts, axis=0))`` of each class face in a
  :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`. **This conversion is
  lossy** (axis-parallel bounding boxes of simplex faces intersect strictly more
  often than the faces themselves—for example, on :math:`T_3` the faces of
  :math:`X_1, X_2, X_3` have empty triple intersection, whereas their bounding
  boxes share a common point by Helly's property) and **must not be used as a
  model**. Empty/unsatisfiable classes map to ``(nan, nan)`` arrays of shape
  ``(d,)``.
- :func:`boxes_as_tensors` packs a :class:`~mowl.geometric.boxes.BoxModel` into
  two float arrays of shape ``(len(class_to_id), d)`` ordered by the integer IDs
  in ``class_to_id`` (such as :attr:`mowl.datasets.Dataset.class_to_id`), filling
  rows for concept names absent from the box model (e.g. ``BOTTOM`` or
  unsatisfiable classes) with ``NaN``. Supports both ``(lo, hi)`` corner pairs
  (``as_center_offset=False``, default) and ``(center, offset)`` pairs
  (``as_center_offset=True``, where ``center = (lo + hi) / 2`` and
  ``offset = (hi - lo) / 2 >= 0``) to initialise ``ELBE`` or
  ``BoxSquaredEL`` class embeddings. Note that :class:`~mowl.geometric.boxes.BoxModelBuilder`
  minimises total box width :math:`\\sum (U_{A,i} - L_{A,i})`, so returned boxes
  are frequently degenerate (:math:`\\text{lo} = \\text{hi}`, zero offset/volume);
  users who require positive box volume for gradient-based training should add a
  small positive margin to ``offset`` (or ``hi - lo``) after export or modify the
  MILP objective.

References:
- Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Lacerda, Ozaki & Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12, Thms. 13, 16, 18, 25.
- Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric.boxes import BoxModel
    from mowl.geometric.polyhedral import ConvexPolyhedralModel


def class_points(model: ConvexPolyhedralModel) -> Dict[str, np.ndarray]:
    """Return ``{name: pts}`` mapping each concept name in ``model`` to its ``(m, d)`` vertex array.

    For every concept name in ``model.class_regions`` (including ``TOP`` and
    ``BOTTOM``), returns a copy of :meth:`~mowl.geometric.polyhedral.ConvexPolyhedralModel.class_points`
    whose rows are the canonical-element simplex vertices spanning that concept's
    face in :math:`\\mathbb{R}^d` (:math:`d = \\text{model.dimension}`). Empty
    classes have shape ``(0, d)``.
    """
    return {
        name: model.class_points(name)
        for name in model.class_regions
    }


def to_bounding_boxes(
    model: ConvexPolyhedralModel,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return ``{name: (lo, hi)}`` elementwise bounding boxes of each class face in ``model``.

    .. warning::
       **Lossy conversion — do not use as a geometric model.**
       Replacing simplex faces :math:`F(S)` by their axis-parallel bounding boxes
       :math:`[\\min_{v \\in F(S)} v,\\; \\max_{v \\in F(S)} v]` does *not* preserve
       disjointness or conjunction axioms: bounding boxes of faces intersect
       strictly more often than the faces do (for instance, axis-parallel boxes
       in :math:`\\mathbb{R}^d` have Helly number 2, so the bounding boxes of
       :math:`X_1, X_2, X_3` in :math:`T_3` always share a common point even
       though the faces have empty triple intersection). Use this function only
       as a heuristic box initialiser or summary, never as a verified model.

    Parameters
    ----------
    model : ConvexPolyhedralModel
        Constructed polyhedral model of dimension :math:`d = \\text{model.dimension}`.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Mapping from each concept name in ``model.class_regions`` to a pair of
        1-D float arrays ``(lo, hi)`` of shape ``(d,)``. For non-empty classes,
        ``lo = pts.min(axis=0)`` and ``hi = pts.max(axis=0)`` where ``pts`` is
        ``model.class_points(name)``. For empty/unsatisfiable classes (``pts.shape[0] == 0``),
        both ``lo`` and ``hi`` are ``np.full(d, np.nan)``.
    """
    d = int(model.dimension)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in model.class_regions:
        pts = model.class_points(name)
        if pts.shape[0] == 0:
            nan_vec = np.full(d, np.nan, dtype=float)
            out[name] = (nan_vec, nan_vec.copy())
        else:
            lo = np.min(pts, axis=0).astype(float, copy=True)
            hi = np.max(pts, axis=0).astype(float, copy=True)
            out[name] = (lo, hi)
    return out


def boxes_as_tensors(
    box_model: BoxModel,
    class_to_id: Dict[str, int],
    as_center_offset: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack ``box_model`` into two float arrays of shape ``(len(class_to_id), d)`` ordered by ``class_to_id``.

    The returned arrays are aligned with ``class_to_id`` (such as
    :attr:`mowl.datasets.Dataset.class_to_id`) and can be converted to PyTorch
    tensors (e.g. ``torch.from_numpy(...)``) to initialise class embeddings in
    :class:`~mowl.models.elbe.model.ELBE` /
    :class:`~mowl.nn.el.elbe.module.ELBEModule` or
    :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`:

    - With ``as_center_offset=False`` (default), returns ``(lo, hi)`` where row
      ``idx = class_to_id[name]`` holds the lower and upper box corners
      :math:`(L_A, U_A) \\in \\mathbb{R}^d \\times \\mathbb{R}^d`.
    - With ``as_center_offset=True``, returns ``(center, offset)`` where
      ``center = 0.5 * (lo + hi)`` and ``offset = 0.5 * (hi - lo) >= 0`` match
      the centre/offset parameterisation used by ``ELBE`` and
      ``BoxSquaredEL``.
    - Concept names in ``class_to_id`` that have no non-empty box in
      ``box_model`` (such as ``owl:Nothing``, unsatisfiable classes, or classes
      outside the model's signature) receive ``NaN`` across their entire row in
      both returned arrays.

    .. note::
       :class:`~mowl.geometric.boxes.BoxModelBuilder` minimises total box width
       :math:`\\sum_{A, i} (U_{A,i} - L_{A,i})`, so returned boxes are frequently
       degenerate (:math:`\\text{lo} = \\text{hi}`, i.e. ``offset == 0``). Users
       initialising gradient-based box embeddings that require positive box
       volume should add a small positive floor to ``offset`` (or ``hi - lo``)
       after calling this function, or change the MILP objective.

    Parameters
    ----------
    box_model : BoxModel
        Verified axis-parallel box model of dimension :math:`d = \\text{box\\_model.dimension}`.
    class_to_id : Dict[str, int]
        Mapping from concept name IRI strings to contiguous integer row indices
        ``0 .. len(class_to_id) - 1``.
    as_center_offset : bool, default False
        If ``False``, return ``(lo, hi)``; if ``True``, return
        ``(center, offset) = ((lo + hi) / 2, (hi - lo) / 2)``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two ``float64`` arrays of shape ``(len(class_to_id), box_model.dimension)``.
    """
    n_classes = len(class_to_id)
    d = int(box_model.dimension)
    indices = list(class_to_id.values())
    if any(
        isinstance(idx, bool) or not isinstance(idx, (int, np.integer))
        for idx in indices
    ) or set(int(idx) for idx in indices) != set(range(n_classes)):
        raise ValueError(
            f"class_to_id values must be a permutation of 0..{n_classes - 1}, "
            f"got {indices!r}"
        )

    first = np.full((n_classes, d), np.nan, dtype=float)
    second = np.full((n_classes, d), np.nan, dtype=float)
    model_boxes = box_model.boxes()

    for name, raw_idx in class_to_id.items():
        idx = int(raw_idx)
        if name in model_boxes:
            lo, hi = model_boxes[name]
            if as_center_offset:
                first[idx] = 0.5 * (lo + hi)
                second[idx] = 0.5 * (hi - lo)
            else:
                first[idx] = lo
                second[idx] = hi

    return first, second
