"""Training designs of reference implementations, as named compositions of regimes.

A paper's numbers depend on how its code trains, and that is often not what the paper's
text or a re-implementation does by default. The regimes here bundle the training design
of one code base, so that a published number can be reproduced by naming one regime.
"""

from mowl.training.regime import ComposedRegime
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.negatives import PerColumnCorruption
from mowl.training.losses import MSETargetLoss
from mowl.training.margin import SlackMargin
from mowl.training.schedules import StepDecay
from mowl.training.selection import ValidationRankSelection


class Box2ELCodeTraining(ComposedRegime):
    """How ``train.py`` of the Box\\ :sup:`2`\\ EL code (KRR-Oxford/BoxSquaredEL) trains
    Box\\ :sup:`2`\\ EL and its baselines, and how the numbers of Jackermeier et al.
    (WWW 2024) were obtained.

    * One optimisation step per "epoch", on ``batch_size`` axioms drawn at random (with
      replacement) from every normal form
      (:class:`~mowl.training.PerNormalFormMiniBatches`); every normal form has the same
      weight, whatever its size.
    * Negatives only for :math:`C \\sqsubseteq \\exists r.D`: for every axiom,
      ``num_negatives`` copies with :math:`D` replaced and ``num_negatives`` with
      :math:`C` replaced (:class:`~mowl.training.PerColumnCorruption`).
    * Slack margin (:class:`~mowl.training.SlackMargin`), if ``margin`` is given: the
      paper's :math:`\\gamma` loosens inclusions, so the embedding need not be a model.
    * For the ELBE baseline (``neg_target``): squared distance of every score to a
      target, 0 for positives and ``neg_target`` (2 in the code) for negatives
      (:class:`~mowl.training.MSETargetLoss`).
    * Optionally the learning rate is multiplied by 0.1 at ``lr_milestones``
      (:class:`~mowl.training.StepDecay`).
    * Every ``validate_every`` epochs the median rank of the first 1000 validation
      axioms :math:`C \\sqsubseteq D` is computed and the best parameters are kept
      (:class:`~mowl.training.ValidationRankSelection`). Model selection thus looks at
      one normal form only.

    Deviation: the code draws the negatives of the baselines once before training; here
    they are drawn anew at every step for all models.

    :param batch_size: Axioms per normal form and step (512 in the code).
    :param num_negatives: Negatives per side and positive (``num_neg`` of the code).
    :param margin: Slack :math:`\\gamma`; ``None`` leaves the model's margin unchanged.
    :param neg_target: Target of the negative scores for a per-axiom squared loss, or
        ``None`` to keep the model's own combination of positive and negative losses.
    :param lr_milestones: Epochs at which the learning rate is multiplied by 0.1.
    :param validate_every: Epochs between validations.
    :param select: Set to ``False`` to train without model selection.
    :param seed: Seed of the batch and negative samplers.
    """

    key = "box2el-code-training"
    _default_batch_size = 512

    def __init__(self, batch_size=None, num_negatives=1, margin=None, neg_target=None,
                 lr_milestones=None, validate_every=100, select=True, seed=None):
        batch_size = self._default_batch_size if batch_size is None else batch_size
        parts = [PerNormalFormMiniBatches(batch_size=batch_size, order="random",
                                          seed=seed),
                 PerColumnCorruption(columns={"gci2": (0, 2)},
                                     num_negatives=num_negatives, seed=seed)]
        if margin is not None:
            parts.append(SlackMargin(margin))
        if neg_target is not None:
            parts.append(MSETargetLoss(pos_target=0.0, neg_target=neg_target))
        if lr_milestones:
            parts.append(StepDecay(lr_milestones, gamma=0.1))
        if select:
            parts.append(ValidationRankSelection(every=validate_every))
        super().__init__(*parts)
        self.key = type(self).key
        self._select = bool(select)

    def attach(self, model):
        super().attach(model)
        # The generic loop needs to know which normal form to validate on; the code
        # validates on C ⊑ D.
        if self._select and getattr(model, "eval_gci_name", "") is None:
            model.eval_gci_name = "gci0"


class TransBoxCodeTraining(Box2ELCodeTraining):
    """How the TransBox code (HuiYang1997/TransBox, ``model/TransBox_reg.py``) trains,
    and how the numbers of Yang et al. (WWW 2025) were obtained: the design of
    :class:`Box2ELCodeTraining` (the code is derived from the Box\\ :sup:`2`\\ EL code)
    with 1024 axioms per normal form and step. As there, the margin is a slack margin,
    although the paper's loss (Section 4.3) is written with a strict one.
    """

    key = "transbox-code-training"
    _default_batch_size = 1024
