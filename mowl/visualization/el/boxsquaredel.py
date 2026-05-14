import torch as th
import numpy as np
import matplotlib.patches as mpatches
from .base import ELVisualizer


class BoxSquaredELVisualizer(ELVisualizer):
    """2-D visualizer for Box²EL (box geometry).

    Each class is drawn as an axis-aligned rectangle. The center is
    ``module.class_center`` and the half-extents are ``|module.class_offset|``.

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
