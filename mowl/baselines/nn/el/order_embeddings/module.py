r"""Order embeddings (Vendrov et al., ICLR 2016) as a module over the class taxonomy."""
import torch as th
import torch.nn as nn

from mowl.nn.el.order_embeddings.base import TaxonomyModule


class OrderEmbeddingsModule(TaxonomyModule):
    r"""Every class is a vector :math:`f(C) = |w_C| \in \mathbb{R}^d_+`.

    **From the paper** (Vendrov et al., Section 2 and 3): the *reversed product order* on
    :math:`\mathbb{R}^d_+`, :math:`x \preceq y \iff x_i \ge y_i` for all :math:`i` (a more
    specific concept has larger coordinates, the origin is the top element); the order
    violation penalty (Eq. 2)

    .. math:: E(x, y) = \lVert \max(0, y - x) \rVert_2,

    which is zero iff :math:`x \preceq y`; and the max-margin loss (Eq. 4) that asks
    :math:`E` to be zero for stated pairs and at least :math:`\alpha` for corrupted ones.
    The embedding is non-negative by taking the absolute value of a free parameter (the
    paper does this for the image embeddings, Eq. 6).

    **mOWL.** A subclass axiom :math:`C \sqsubseteq D` is the pair :math:`f(C) \preceq f(D)`,
    so its cost is :math:`E(f(C), f(D))`.

    :param nb_ont_classes: Number of classes (named and auxiliary).
    :param embed_dim: Dimension :math:`d`.
    :param margin: The margin :math:`\alpha`. The paper uses 1 for WordNet.
    :param init_scale: Parameters are initialised uniformly in ``[0, init_scale]``.
    """

    def __init__(self, nb_ont_classes, embed_dim=50, margin=1.0, init_scale=1.0):
        super().__init__(nb_ont_classes)
        self.embed_dim = embed_dim
        self.margin = float(margin)
        self.class_embed = nn.Embedding(nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=0.0, b=init_scale)

    def vectors(self, idx):
        """The non-negative vectors of the classes ``idx``."""
        return self.class_embed(idx).abs()

    @staticmethod
    def violation(x, y):
        r""":math:`E(x, y) = \lVert \max(0, y - x) \rVert_2` per row."""
        return th.sqrt((th.relu(y - x) ** 2).sum(dim=-1) + 1e-24) - 1e-12

    def positive_loss(self, sub, sup):
        return self.violation(self.vectors(sub), self.vectors(sup)).clamp(min=0.0)

    def negative_loss(self, sub, sup):
        return th.relu(self.margin - self.violation(self.vectors(sub), self.vectors(sup)))
