import torch as th
import numpy as np
import matplotlib.patches as mpatches
from .base import ELVisualizer, _local_name


class BoxSquaredELVisualizer(ELVisualizer):
    """2-D visualizer for Box²EL (box geometry).

    Classes are drawn as axis-aligned rectangles using ``module.class_center``
    (centre) and ``|module.class_offset|`` (half-extents).

    Roles are drawn as **two** boxes per relation — the head box (solid edge, soft fill)
    and the tail box (dashed edge, soft fill) — using ``module.head_center`` /
    ``module.head_offset`` and ``module.tail_center`` / ``module.tail_offset`` respectively.

    :param model: A trained :class:`~mowl.models.BoxSquaredEL` model with ``embed_dim=2``.
    """

    def _get_patch(self, iri):
        idx = self._get_class_index(iri)
        with th.no_grad():
            center = self.model.module.class_center.weight[idx].cpu().numpy()
            half = np.abs(self.model.module.class_offset.weight[idx].cpu().numpy())
        ll = center - half
        size = 2 * half
        return mpatches.Rectangle(ll, size[0], size[1], fill=False, edgecolor="steelblue", linewidth=1.5)

    def _patch_center(self, iri):
        idx = self._get_class_index(iri)
        with th.no_grad():
            c = self.model.module.class_center.weight[idx].cpu().numpy()
        return float(c[0]), float(c[1])

    def _get_role_patches(self, iri):
        """Return head box (solid, soft fill) and tail box (dashed, soft fill) for role *iri*.

        :param iri: Full IRI of the object property.
        :type iri: str
        :returns: List of two ``(patch, cx, cy, label)`` tuples —
            one for the head box and one for the tail box.
        :rtype: list
        """
        idx = self._get_role_index(iri)
        local = _local_name(iri)
        with th.no_grad():
            hc = self.model.module.head_center.weight[idx].cpu().numpy()
            ho = np.abs(self.model.module.head_offset.weight[idx].cpu().numpy())
            tc = self.model.module.tail_center.weight[idx].cpu().numpy()
            to_ = np.abs(self.model.module.tail_offset.weight[idx].cpu().numpy())
        head_patch = mpatches.Rectangle(
            hc - ho, 2 * ho[0], 2 * ho[1],
            fill=True, facecolor="tomato", alpha=0.15,
            edgecolor="tomato", linestyle="solid", linewidth=1.5,
        )
        tail_patch = mpatches.Rectangle(
            tc - to_, 2 * to_[0], 2 * to_[1],
            fill=True, facecolor="tomato", alpha=0.15,
            edgecolor="tomato", linestyle="dashed", linewidth=1.5,
        )
        return [
            (head_patch, float(hc[0]), float(hc[1]), f"{local}(head)"),
            (tail_patch, float(tc[0]), float(tc[1]), f"{local}(tail)"),
        ]
