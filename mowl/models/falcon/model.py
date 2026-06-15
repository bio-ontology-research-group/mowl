from mowl.base_models.alcmodel import EmbeddingALCModel
from mowl.nn import FALCONModule


class FALCONModel(EmbeddingALCModel):
    """FALCON model [falcon2022]_: a fuzzy :math:`\\mathcal{ALC}` neural reasoner.

    Each class expression is interpreted as a fuzzy set over a collection of named and
    sampled anonymous entities, and axioms are scored through fuzzy logical operators.
    The training objective combines a TBox term and two ABox terms (concept and relation
    assertions), weighted by ``alpha`` and ``beta`` (see
    :class:`EmbeddingALCModel <mowl.base_models.alcmodel.EmbeddingALCModel>`).

    :param dataset: mOWL dataset to use for training.
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param embed_dim: Dimension of the embeddings. Defaults to ``50``.
    :type embed_dim: int, optional
    :param anon_e: Number of anonymous entities sampled each epoch. Defaults to ``4``.
    :type anon_e: int, optional
    :param t_norm: Fuzzy t-norm: ``'product'``, ``'minmax'`` or ``'Łukasiewicz'``. \
    Defaults to ``'product'``.
    :type t_norm: str, optional
    :param max_measure: Aggregation used by the concept-concept loss. Defaults to ``'max'``.
    :type max_measure: str, optional
    :param num_negs: Number of negative samples per ABox axiom. Defaults to ``4``.
    :type num_negs: int, optional
    """

    def __init__(self, dataset, embed_dim=50, anon_e=4, alpha=0.3, beta=0.3,
                 t_norm='product', max_measure='max', num_negs=4,
                 learning_rate=0.001, batch_size=256, model_filepath=None, device='cpu'):
        super().__init__(dataset, embed_dim=embed_dim, batch_size=batch_size,
                         alpha=alpha, beta=beta, anon_e=anon_e,
                         learning_rate=learning_rate, model_filepath=model_filepath,
                         device=device)

        self.t_norm = t_norm
        self.max_measure = max_measure
        self.num_negs = num_negs
        self.init_module()

    def _build_assertion_dicts(self):
        """Builds the (entity, relation) -> entities dictionaries used for negative \
        sampling of object-property assertions."""
        heads_dict = {}
        tails_dict = {}
        ind_triples = self.training_dataset.get_obj_prop_assertion_data()
        if ind_triples is None:
            return heads_dict, tails_dict

        for item in ind_triples:
            h, r, t = item[0]
            h, r, t = h.item(), r.item(), t.item()
            heads_dict.setdefault((t, r), []).append(h)
            tails_dict.setdefault((h, r), []).append(t)
        return heads_dict, tails_dict

    def init_module(self):
        heads_dict, tails_dict = self._build_assertion_dicts()
        # The membership network always needs at least one named entity row, even for
        # purely terminological (TBox-only) ontologies with no individuals.
        nentities = max(len(self.dataset.individuals), 1)
        self.module = FALCONModule(
            len(self.dataset.classes),
            nentities,
            len(self.dataset.object_properties),
            heads_dict,
            tails_dict,
            embed_dim=self.embed_dim,
            anon_e=self.anon_e,
            t_norm=self.t_norm,
            max_measure=self.max_measure,
            num_negs=self.num_negs,
            device=self.device,
        ).to(self.device)
