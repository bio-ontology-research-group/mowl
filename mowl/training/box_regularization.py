"""The box regulariser of the Box\\ :sup:`2`\\ EL code, as a regime.

The BoxEL baseline of the Box\\ :sup:`2`\\ EL code (``model/BoxEL.py``, ``l2_side_regularizer``)
regularises *every box of every batch* of a step: for the classes of an axiom, the intersection
of :math:`C \\sqcap D` and the translated box of :math:`C`, including the boxes of corrupted
axioms, it adds the mean excess of the upper corners over 1 plus the excess of the norm of the
lower corners over 1 (one scalar per batch of boxes). This is not a function of the parameters
alone, but of the batches of the step, so it cannot be a penalty on the tables of the model
(:mod:`mowl.training.regularization`): the regime records the batches that the training loop
scores and computes the term when the loop asks for the regularisation of the step.
"""

import torch as th

from mowl.training.regime import TrainingRegime


def side_regulariser(box):
    """The regulariser of one batch of boxes (one scalar per batch, not per box)."""
    upper = th.mean(th.relu(box.upper - 1 + 1e-8))
    return upper + th.relu(th.linalg.norm(box.lower) - 1)


def axiom_boxes(module, gci_name, data):
    """The boxes that the Box2EL code regularises for a batch of one normal form: the classes
    of the axiom, the intersection of :math:`C \\sqcap D` and the translated box of :math:`C`
    (read through the class and relation model of a geometry with box classes and an affine
    relation model)."""
    def box(col):
        return module.classes(data[:, col])

    if gci_name in ("gci0", "gci1_bot"):
        return [box(0), box(1)]
    if gci_name == "gci1":
        c, d = box(0), box(1)
        return [c.intersect(d)[0], c, d, box(2)]
    if gci_name == "gci2":
        c, d = box(0), box(2)
        return [module.relation_model.image(data[:, 1], c), c, d]
    if gci_name == "gci3":
        c, d = box(1), box(2)
        return [module.relation_model.preimage(data[:, 0], c), c, d]
    return []


class BoxBatchRegulariser(TrainingRegime):
    """The regulariser of the batches of the step: the sum over the boxes of every positive
    and corrupted batch that the loop scores (:func:`axiom_boxes`) of :func:`side_regulariser`,
    which replaces the regularisation term of the model for the step (the Box\\ :sup:`2`\\ EL
    code regularises this way instead of the penalty on the tables).

    Needs a geometry whose class model gives boxes (``module.classes``) and whose relation model
    has ``image`` and ``preimage`` (an affine map): BoxEL and BoxSEL.
    """

    key = "box-batch-regulariser"

    def __init__(self):
        super().__init__()
        self._batches = []

    def attach(self, model):
        module = model.module
        if not hasattr(module, "classes") or not hasattr(getattr(module, "relation_model", None),
                                                          "image"):
            raise TypeError(f"{type(module).__name__} has no box classes with an affine "
                            f"relation model: the batch regulariser of the Box2EL code "
                            f"needs a geometry such as BoxEL's.")
        super().attach(model)
        self._batches = []

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        # called once per normal form and step, with the positives and the corruptions that
        # the loop scores
        self._batches.append((gci_name, gci_dataset[:]))
        if negatives is not None:
            self._batches.append((gci_name, negatives))
        return negatives

    def regularization(self, default):
        batches, self._batches = self._batches, []
        module = self.model.module
        if not (module.training and th.is_grad_enabled()):
            return NotImplemented
        total = 0
        for gci_name, rows in batches:
            for box in axiom_boxes(module, gci_name, rows):
                total = total + side_regulariser(box)
        if isinstance(total, int):
            return th.zeros((), device=self.model.device)
        return total
