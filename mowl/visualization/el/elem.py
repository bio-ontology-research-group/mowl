import torch as th
import matplotlib.patches as mpatches
from .base import ELVisualizer


class ELEmVisualizer(ELVisualizer):
    """2-D visualizer for ELEm (circle geometry).

    Each class is drawn as a circle whose center is ``module.class_embed`` and
    whose radius is the absolute value of ``module.class_rad``.

    :param model: A trained :class:`~mowl.models.ELEmbeddings` model with ``embed_dim=2``.
    """

    def _get_patch(self, iri):
        idx = self._get_class_index(iri)
        with th.no_grad():
            center = self.model.module.class_embed.weight[idx].cpu().numpy()
            radius = float(abs(self.model.module.class_rad.weight[idx].cpu()))
        return mpatches.Circle(center, radius, fill=False, edgecolor="steelblue", linewidth=1.5)

    def _patch_center(self, iri):
        idx = self._get_class_index(iri)
        with th.no_grad():
            c = self.model.module.class_embed.weight[idx].cpu().numpy()
        return float(c[0]), float(c[1])
