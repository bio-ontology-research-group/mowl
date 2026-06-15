import torch as th
from mowl.owlapi.defaults import BOT, TOP


def _short_name(iri):
    """Returns a short human-readable label for an IRI."""
    for sep in ("#", "/"):
        if sep in iri:
            iri = iri.rsplit(sep, 1)[-1]
    return iri


class FALCONVisualizer:
    """Renders the FALCON membership heatmap, in the style of Figure 1 of [falcon2022]_.

    Given a trained :class:`FALCONModel <mowl.models.falcon.model.FALCONModel>`, it
    computes the fuzzy membership degree :math:`\\mu(C, e)` of every concept ``C`` over a
    set of entities ``e`` (the named individuals of the ontology plus a number of sampled
    anonymous entities) and renders it as a grayscale heatmap: dark cells denote
    membership degrees close to ``1``, light cells close to ``0``.

    :param model: A trained FALCON model.
    :type model: :class:`mowl.models.falcon.model.FALCONModel`
    """

    def __init__(self, model):
        self.model = model

    def _entity_context(self, n_anon):
        module = self.model.module
        n_named = len(self.model.individual_index_dict)
        named = module.e_embedding.weight.detach()[:n_named]
        dim = module.e_embedding.weight.shape[1]
        anon = th.empty(n_anon, dim, device=named.device)
        th.nn.init.xavier_uniform_(anon)
        if named.shape[0] > 0:
            return th.cat([named, anon], dim=0)
        return anon

    def membership_matrix(self, n_anon=4):
        """Computes the concept-by-entity membership matrix.

        :param n_anon: Number of anonymous entities to sample. Defaults to ``4``.
        :type n_anon: int, optional
        :returns: ``(matrix, concept_labels, entity_labels)`` where ``matrix`` is a
            ``(n_concepts, n_entities)`` numpy array of membership degrees.
        """
        module = self.model.module
        module.eval()
        with th.no_grad():
            e_emb = self._entity_context(n_anon)
            c_emb = module.c_embedding.weight.detach()
            fs = module._get_c_fs_batch(c_emb, e_emb).cpu().numpy()

        # Ground the logical constants: bottom (owl:Nothing) is empty and top
        # (owl:Thing) is universal, regardless of what the membership network learned.
        for iri, idx in self.model.class_index_dict.items():
            if iri == BOT:
                fs[idx, :] = 0.0
            elif iri == TOP:
                fs[idx, :] = 1.0

        concept_labels = [_short_name(iri) for iri in self.model.class_index_dict.keys()]
        entity_labels = [_short_name(iri) for iri in self.model.individual_index_dict.keys()]
        entity_labels += [f"anon_{i}" for i in range(n_anon)]
        return fs, concept_labels, entity_labels

    def plot(self, n_anon=4, ax=None, cmap="Greys"):
        """Renders the membership heatmap.

        :param n_anon: Number of anonymous entities to sample. Defaults to ``4``.
        :type n_anon: int, optional
        :param ax: Optional matplotlib axes to draw on. A new figure is created if omitted.
        :param cmap: Matplotlib colormap. Defaults to ``'Greys'``.
        :returns: The matplotlib axes containing the heatmap.
        """
        import matplotlib.pyplot as plt

        fs, concept_labels, entity_labels = self.membership_matrix(n_anon=n_anon)

        if ax is None:
            _, ax = plt.subplots(
                figsize=(max(4, len(entity_labels) * 0.6),
                         max(4, len(concept_labels) * 0.4)))

        im = ax.imshow(fs, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(entity_labels)))
        ax.set_xticklabels(entity_labels, rotation=90)
        ax.set_yticks(range(len(concept_labels)))
        ax.set_yticklabels(concept_labels)
        ax.set_xlabel("Entities")
        ax.set_ylabel("Concepts")
        ax.set_title("FALCON membership degrees")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.figure.tight_layout()
        return ax
