import torch as th
import matplotlib.patches as mpatches
from .base import ELVisualizer


class BoxELVisualizer(ELVisualizer):
    """2-D visualizer for BoxEL (box geometry).

    Each class is drawn as an axis-aligned rectangle whose lower-left corner is
    ``module.min_embedding`` and whose size is ``module.delta_embedding``.

    :param model: A trained BoxEL model with ``embed_dim=2``.
    """

    def _get_patch(self, iri):
        idx = self._get_class_index(iri)
        with th.no_grad():
            mn = self.model.module.min_embedding.weight[idx].cpu().numpy()
            delta = self.model.module.delta_embedding.weight[idx].cpu().numpy()
        return mpatches.Rectangle(mn, float(delta[0]), float(delta[1]),
                                  fill=False, edgecolor="steelblue", linewidth=1.5)

    def _patch_center(self, iri):
        idx = self._get_class_index(iri)
        with th.no_grad():
            mn = self.model.module.min_embedding.weight[idx].cpu().numpy()
            delta = self.model.module.delta_embedding.weight[idx].cpu().numpy()
        c = mn + delta / 2
        return float(c[0]), float(c[1])
