r"""Order embeddings of the class taxonomy."""
from mowl.models.order_parts_a import TaxonomyEmbeddingModel
from mowl.nn.el.order_embeddings.module import OrderEmbeddingsModule


class OrderEmbeddings(TaxonomyEmbeddingModel):
    r"""Order embeddings ([vendrov2016]_) of the subclass order between named classes.

    A class is a non-negative vector; :math:`C \sqsubseteq D` holds when the vector of
    :math:`C` is coordinatewise at least that of :math:`D` (reversed product order),
    violations are penalised by :math:`E(x, y) = \lVert \max(0, y - x) \rVert_2` and the
    loss is the max-margin loss of Eq. 4. The model is a combination of parts
    (``model.module.parts``): the class model of non-negative vectors, the order violation
    as the measure and the hinge objective, defined once in
    :func:`mowl.representation.library.hierarchy_methods.order_embeddings`; see
    :class:`~mowl.nn.el.order_embeddings.OrderEmbeddingsModule`. Any part can be swapped
    (``ComposedEL(dataset, method="order_embeddings", ...)`` or the parts of
    :mod:`mowl.representation`).

    **Supported.** Axioms :math:`C \sqsubseteq D` between named classes (training and
    scoring: subsumption prediction/inference). Nothing else in the ontology is used
    (:class:`~mowl.models.order_parts_a.TaxonomyEmbeddingModel`). This is an
    **adaptation**: the paper embeds hypernymy, entailment and image-caption order; mOWL
    takes the subclass axioms of the ontology as the order pairs. A zero loss on the
    training pairs implies that the embedding preserves the *transitive closure* of the
    stated subclass axioms (the order relation is transitive).

    The paper's WordNet setting is ``embed_dim=50``, ``margin=1``, ``learning_rate=0.01``,
    batches of 500 positive and 500 negative pairs and early stopping; mOWL's generic loop
    is full batch.

    :param dataset: mOWL dataset.
    :param embed_dim: Embedding dimension.
    :param margin: Margin :math:`\alpha` of the negative pairs.
    :param learning_rate: Adam learning rate (paper: 0.01).
    :param batch_size: Kept for interface parity (the generic loop is full batch).
    :param neg_columns: Columns to corrupt for negatives (default: both).
    """

    def __init__(self, dataset, embed_dim=50, margin=1.0, learning_rate=0.01,
                 batch_size=4096, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None, neg_columns=(0, 1)):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath,
                         device=device, learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis, neg_columns=neg_columns)
        self.margin = margin
        self.init_module()

    def init_module(self):
        self.module = OrderEmbeddingsModule(
            len(self.class_index_dict), embed_dim=self.embed_dim,
            margin=self.margin).to(self.device)
