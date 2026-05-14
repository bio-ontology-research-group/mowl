from abc import ABC, abstractmethod
import matplotlib.figure


def _local_name(iri):
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.rsplit("/", 1)[-1] or iri


class ELVisualizer(ABC):
    """Abstract base class for 2-D geometric visualizations of EL embedding models.

    Concrete subclasses implement :meth:`_get_patch` and :meth:`_patch_center` for
    each model's class geometry, and optionally :meth:`_get_role_patches` for role
    geometry (e.g., box pairs in Box²EL).

    :param model: A trained EL embedding model with ``embed_dim=2``.
    :type model: :class:`mowl.base_models.elmodel.EmbeddingELModel`
    :raises ValueError: If ``model.embed_dim`` is not 2.
    """

    def __init__(self, model):
        if model.embed_dim != 2:
            raise ValueError(
                f"ELVisualizer requires a model trained with embed_dim=2, "
                f"got embed_dim={model.embed_dim}."
            )
        self.model = model
        self._fig = None
        self._ax = None

    def _get_class_index(self, iri):
        idx = self.model.class_index_dict.get(iri)
        if idx is None:
            raise KeyError(f"Entity not found in model class index: {iri!r}")
        return idx

    def _get_role_index(self, iri):
        idx = self.model.object_property_index_dict.get(iri)
        if idx is None:
            raise KeyError(f"Role not found in model property index: {iri!r}")
        return idx

    @abstractmethod
    def _get_patch(self, iri):
        """Return a matplotlib patch for the geometric shape of *iri*."""

    @abstractmethod
    def _patch_center(self, iri):
        """Return ``(x, y)`` center coordinates for the shape of *iri*."""

    def _get_role_patches(self, iri):
        """Return a list of ``(patch, cx, cy, label)`` tuples for role *iri*.

        Each element represents one geometric shape associated with the role.
        For example, Box²EL returns two entries — the head box and the tail box.

        Subclasses that do not model roles as geometric shapes should leave this
        unimplemented; calling :meth:`plot_roles` on them will raise
        :class:`NotImplementedError`.

        :param iri: Full IRI of the object property to visualize.
        :type iri: str
        :returns: List of ``(patch, cx, cy, label)`` tuples.
        :rtype: list
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support role visualization. "
            "Override _get_role_patches() to add support."
        )

    def plot(self, entities, figsize=(8, 8), ax=None):
        """Draw geometric shapes for the given entity IRIs.

        :param entities: Full IRIs of the classes to visualize.
        :type entities: list of str
        :param figsize: Figure size; ignored when *ax* is provided. Defaults to ``(8, 8)``.
        :type figsize: tuple, optional
        :param ax: Existing axes to draw into. If ``None``, a new figure is created.
        :type ax: matplotlib.axes.Axes, optional
        :returns: The axes containing the plot.
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            self._fig = matplotlib.figure.Figure(figsize=figsize)
            self._ax = self._fig.add_subplot(1, 1, 1)
        else:
            self._fig = ax.get_figure()
            self._ax = ax

        for iri in entities:
            patch = self._get_patch(iri)
            self._ax.add_patch(patch)
            cx, cy = self._patch_center(iri)
            self._ax.annotate(
                _local_name(iri), (cx, cy),
                ha="center", va="center", fontsize=8,
            )

        self._ax.autoscale()
        self._ax.set_aspect("equal")
        return self._ax

    def plot_roles(self, role_iris, figsize=(8, 8), ax=None):
        """Draw geometric shapes for the given role IRIs.

        Each role may produce one or more patches depending on the model geometry.
        For example, Box²EL produces a head box and a tail box per role.

        :param role_iris: Full IRIs of the object properties to visualize.
        :type role_iris: list of str
        :param figsize: Figure size; ignored when *ax* is provided. Defaults to ``(8, 8)``.
        :type figsize: tuple, optional
        :param ax: Existing axes to draw into. If ``None``, a new figure is created.
        :type ax: matplotlib.axes.Axes, optional
        :returns: The axes containing the plot.
        :rtype: matplotlib.axes.Axes
        :raises NotImplementedError: If the subclass does not implement
            :meth:`_get_role_patches`.
        """
        if ax is None:
            self._fig = matplotlib.figure.Figure(figsize=figsize)
            self._ax = self._fig.add_subplot(1, 1, 1)
        else:
            self._fig = ax.get_figure()
            self._ax = ax

        for iri in role_iris:
            for patch, cx, cy, label in self._get_role_patches(iri):
                self._ax.add_patch(patch)
                self._ax.annotate(
                    label, (cx, cy),
                    ha="center", va="center", fontsize=8,
                )

        self._ax.autoscale()
        self._ax.set_aspect("equal")
        return self._ax

    def show(self):
        """Display the plot interactively.

        Requires a GUI-capable matplotlib backend. In headless environments,
        save to a file with :meth:`savefig` instead.
        """
        import matplotlib.pyplot as plt
        plt.show(self._fig)

    def savefig(self, outfile):
        """Save the plot to *outfile*.

        :param outfile: Destination file path (format inferred from extension).
        :type outfile: str
        :raises RuntimeError: If :meth:`plot` has not been called first.
        """
        if self._fig is None:
            raise RuntimeError("Call plot() before savefig().")
        self._fig.savefig(outfile)
