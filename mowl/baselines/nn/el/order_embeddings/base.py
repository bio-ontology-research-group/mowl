r"""Base module for order and hierarchy embeddings of the class taxonomy.

The methods of this family (order embeddings, box lattices, Gumbel boxes) embed a partial
order, not a description logic. mOWL connects them to an ontology through the subclass
axioms between named classes: ``C ⊑ D`` (normal form ``gci0``) is an order pair
``C ≤ D``. No other normal form has a meaning in these geometries, so
:class:`TaxonomyModule` implements ``gci0`` only.
"""
from mowl.baselines.nn.el.elmodule import ELModule


class TaxonomyModule(ELModule):
    r"""An :class:`~mowl.nn.ELModule` that implements ``gci0`` (:math:`C \sqsubseteq D`)
    and nothing else.

    A subclass provides :meth:`positive_loss` (a non-negative cost of the pair, lower is
    better and zero exactly when the order relation holds) and :meth:`negative_loss` (a
    cost that is low when the pair is *not* ordered). The positive cost is also the score
    that the ranking regimes use for a candidate subclass axiom.
    """

    neg_capable_gcis = frozenset({"gci0"})
    role_axiom_capable = False

    def __init__(self, nb_ont_classes):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes

    def positive_loss(self, sub, sup):
        raise NotImplementedError

    def negative_loss(self, sub, sup):
        raise NotImplementedError

    def gci0_loss(self, gci, neg=False):
        if neg:
            return self.negative_loss(gci[:, 0], gci[:, 1])
        return self.positive_loss(gci[:, 0], gci[:, 1])
