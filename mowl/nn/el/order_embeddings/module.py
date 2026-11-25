r"""Order embeddings (Vendrov et al., ICLR 2016) over the class taxonomy, built from parts."""

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.hierarchy_methods import order_embeddings
from mowl.representation.library.hierarchy_measures import OrderViolation
from mowl.representation.library.hierarchy_modules import FlatLossOption, adopt


class OrderEmbeddingsModule(FlatLossOption, ComposedGeometry):
    r"""Every class is a vector :math:`f(C) = |w_C| \in \mathbb{R}^d_+`; a subclass axiom
    :math:`C \sqsubseteq D` is the pair :math:`f(C) \preceq f(D)` in the reversed product
    order (:math:`x \preceq y` iff :math:`x_i \ge y_i` for all :math:`i`).

    The module is the combination that :func:`mowl.representation.library.hierarchy_methods.
    order_embeddings` names (read ``module.parts``): a class model of non-negative vectors
    (:class:`~mowl.representation.library.hierarchy_classes.OrderPoints`), the order
    violation :math:`E(x, y) = \lVert \max(0, y - x) \rVert_2` of Vendrov et al. (Eq. 2) as
    the measure
    (:class:`~mowl.representation.library.hierarchy_measures.OrderViolation`) and the
    max-margin loss of Eq. 4 as the objective (cost :math:`E` for a stated pair,
    :math:`\max(0, \alpha - E)` for a corrupted one). It reads ``gci0`` only.

    The earlier hand-written module is kept for comparison as
    ``mowl.baselines.nn.el.order_embeddings.OrderEmbeddingsModule``.

    :param nb_ont_classes: Number of classes (named and auxiliary).
    :param embed_dim: Dimension :math:`d`.
    :param margin: The margin :math:`\alpha`. The paper uses 1 for WordNet.
    :param init_scale: Parameters are initialised uniformly in ``[0, init_scale]``.
    """

    def __init__(self, nb_ont_classes, embed_dim=50, margin=1.0, init_scale=1.0):
        adopt(self, order_embeddings(nb_ont_classes, embed_dim=embed_dim, margin=margin,
                                     init_scale=init_scale))
        self.nb_ont_classes = nb_ont_classes
        self.embed_dim = embed_dim

    @staticmethod
    def violation(x, y):
        r""":math:`E(x, y) = \lVert \max(0, y - x) \rVert_2` per row, for vectors."""
        return OrderViolation("reversed", 2).violation(x, y)

    def vectors(self, idx):
        """The non-negative vectors of the classes ``idx``."""
        return self.classes(idx).centre
