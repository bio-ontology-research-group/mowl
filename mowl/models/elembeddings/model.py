from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.elem.module import ELEmModule


class ELEmbeddings(EmbeddingELModel):
    """
    Implementation based on [kulmanov2019]_.

    The idea of this paper is to embed EL by modeling ontology classes as :math:`n`-dimensional \
    balls (:math:`n`-balls) and ontology object properties as transformations of those \
    :math:`n`-balls. For each of the normal forms, there is a distance function defined that will \
    work as loss functions in the optimization framework.

    :param gci1_radius_term: If ``True`` (default), the loss of :math:`C \\sqcap D \\sqsubseteq E` \
    includes the term :math:`\\max(0, \\min(r_C, r_D) - r_E - \\gamma)` of the published loss. \
    The authors' released code has this term commented out, and mOWL did not have it either; \
    pass ``False`` to train with that loss (the recipes of :mod:`mowl.benchmarks` do so where \
    a publication used such code).
    :type gci1_radius_term: bool, optional

    The module of the model is the combination of parts of the method ``"elem"``
    (:class:`~mowl.nn.ELEmModule`, a :class:`~mowl.representation.ComposedGeometry`): balls,
    translations, the centre-distance measure, ELEmbeddings' objective and the unit-norm
    regulariser. The direct implementation is kept in :mod:`mowl.baselines.models.elembeddings`.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 gci1_radius_term=True
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis)

        self.margin = margin
        self.reg_norm = reg_norm
        self.gci1_radius_term = gci1_radius_term
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        self.module = ELEmModule(
            len(self.class_index_dict),  # number of ontology classes
            len(self.object_property_index_dict),  # number of ontology object properties
            len(self.individual_index_dict),  # number of individuals
            embed_dim=self.embed_dim,
            margin=self.margin,
            gci1_radius_term=self.gci1_radius_term
        ).to(self.device)
