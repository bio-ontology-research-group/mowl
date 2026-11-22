"""The training design of the CatE reference code (Zhapa-Camacho and Hoehndorf,
arXiv:2305.07163v3, ``src/cat.py``, ``src/cat_completion.py``, ``src/cat_membership.py``
and ``src/cat_ppi_evaluator.py``).

The design is the combination of regimes that :class:`mowl.models.cate.CatE` names by
default (:attr:`~mowl.models.cate.CatE.default_regimes`) with the options of the reference
code; attaching this regime switches the model to it. It exists so that a published number can
be reproduced by naming one regime, as :class:`mowl.training.reference.Box2ELCodeTraining`
does for Box\\ :sup:`2`\\ EL.
"""

from mowl.training.regime import ComposedRegime


class CateCodeTraining(ComposedRegime):
    """How the CatE code trains, and how the numbers of Zhapa-Camacho and Hoehndorf
    (arXiv:2305.07163v3, Tables 2-4) were obtained.

    * One epoch is a pass over *all* morphisms of the lattice in random order, in
      minibatches of ``batch_size`` morphisms (the code's ``FastTensorDataLoader`` over
      the released edgelist; the trivial reflexive and top/bottom morphisms are trained
      on like any other).
    * Negatives replace the superconcept of every morphism by an element drawn
      uniformly from *all* lattice elements (fixed-column corruption), ``num_negs``
      times; corruptions that are themselves morphisms (e.g. :math:`A \\sqsubseteq
      \\top`) are *kept* as negatives — with them a zero loss is unattainable, but the
      reference loss never filters them.
    * The loss is Eq. 7 of the paper as the code computes it
      (``loss_type="normal"``): the mean positive order violation plus
      :math:`\\max(0, \\gamma - \\overline{s})` where :math:`\\overline{s}` is the mean
      violation of the ``num_negs`` corruptions of one morphism (the paper's sum over
      corruptions is averaged; ``negative_aggregation="mean"``,
      ``filter_negatives=False``). The code also offers a
      Vendrov-et-al. log-sigmoid loss (``loss_type="bpr"``) whose sign convention
      differs between ``src/cat.py`` and ``src/cat_ppi_evaluator.py``; which one
      produced the published numbers is not recorded, and it is not offered here.
    * Adam with weight decay :math:`10^{-4}`, a triangular cyclic learning rate between
      ``learning_rate / 10`` and ``learning_rate`` with 20 steps per half cycle,
      stepped per batch, Xavier-uniform initialisation (the PyKEEN default of the
      reference model).
    * Model selection: every ``validate_every`` epochs the raw (unfiltered) MRR of the
      validation queries is computed and the best parameters are kept; training stops
      after ``patience`` validations without improvement. (The code validates at
      epochs 0, 100, 200, ... and resets its tolerance counter also when only the
      validation mean rank improves; mOWL validates at epochs ``validate_every``,
      ``2 \\\\times validate_every``, ... and counts the MRR only.)

    With ``interaction_loss=True`` (the PPI task): every batch additionally gets the
    reference PPI term — the mean violation of its morphisms whose head is one of the
    dataset's head evaluation classes (the proteins), plus a hinge on one corruption
    whose tail is drawn uniformly among the elements :math:`\\exists r.p`
    (``src/cat_ppi_evaluator.py``, ``train``). Not described in the paper; the published
    Table 4 was produced with it.

    :param interaction_loss: Add the PPI term of the reference training loop.
    """

    key = "cate-code-training"

    def __init__(self, interaction_loss=False):
        super().__init__()
        self.key = type(self).key
        self._interaction_loss = bool(interaction_loss)

    def attach(self, model):
        from mowl.models.cate import CatE
        if not isinstance(model, CatE):
            raise TypeError(f"CateCodeTraining needs a CatE model, got {type(model).__name__}.")
        # Eq. 7 as the reference code computes it; the options of the model read back it
        model.negative_aggregation = "mean"
        model.filter_negatives = False
        model.interaction_loss = self._interaction_loss
        self.regimes = tuple(make() for make in model._regimes(
            "mean", False, self._interaction_loss))
        super().attach(model)
