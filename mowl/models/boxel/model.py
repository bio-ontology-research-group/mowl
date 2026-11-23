from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import BoxELModule


class BoxEL(EmbeddingELModel):
    """
    Implementation based on [xiong2022]_.

    This model uses box embeddings where each class is represented as an
    axis-aligned box defined by a minimum point and a delta (size) vector.

    :param positive_scale: Use :math:`|s| + \\epsilon` as the scale of a role (default) so
        that no role inverts a box; ``False`` is the signed scale of the original code.
    :param role_chain_composition: ``"in_order"`` (default) or ``"as_released"``, see
        :func:`mowl.baselines.nn.el.boxel.losses.role_chain_loss`.
    :param load_role_axioms: Also train on the role inclusions and role chains of the \
        ontology (the module implements their losses). Default ``False``.
    :param gci3_preimage: ``"corners"`` (default) or ``"as_released"`` (the box of
        :math:`\\exists r.C` of mOWL 2.1.0, a defect), see :class:`~mowl.nn.BoxELModule`.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 min_bounds=(1e-4, 0.2),
                 delta_bounds=(-0.1, 0),
                 relation_bounds=(-0.1, 0.1),
                 scaling_bounds=(0.9, 1.1),
                 temperature=1.0,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 positive_scale=True,
                 role_chain_composition="in_order",
                 load_role_axioms=False,
                 gci3_preimage="corners",
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis,
                         load_role_axioms=load_role_axioms)

        self.min_bounds = list(min_bounds)
        self.delta_bounds = list(delta_bounds)
        self.relation_bounds = list(relation_bounds)
        self.scaling_bounds = list(scaling_bounds)
        self.temperature = temperature
        self.positive_scale = positive_scale
        self.role_chain_composition = role_chain_composition
        self.gci3_preimage = gci3_preimage
        self.init_module()

    def init_module(self):
        self.module = BoxELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            min_bounds=self.min_bounds,
            delta_bounds=self.delta_bounds,
            relation_bounds=self.relation_bounds,
            scaling_bounds=self.scaling_bounds,
            temperature=self.temperature,
            positive_scale=self.positive_scale,
            role_chain_composition=self.role_chain_composition,
            gci3_preimage=self.gci3_preimage,
        ).to(self.device)
