"""Training designs of reference implementations, as named compositions of regimes.

A paper's numbers depend on how its code trains, and that is often not what the paper's
text or a re-implementation does by default. The regimes here bundle the training design
of one code base, so that a published number can be reproduced by naming one regime.
"""

from mowl.training.regime import ComposedRegime
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.negatives import PerColumnCorruption
from mowl.training.losses import MSETargetLoss, SoftmaxEnergyContrastiveLoss
from mowl.training.margin import SlackMargin
from mowl.training.schedules import StepDecay
from mowl.training.selection import ValidationRankSelection, ValidationF1Selection


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


class RegDCodeTraining(ComposedRegime):
    """How ``EXP_ontology/baseline_sweep.py`` of the RegD code (HuiYang1997/RegD) trains
    RegD and the ELBE / ELEm baselines, and how the numbers of Yang and Chen
    (arXiv:2501.17518, Tables 3 and 4) were obtained.

    * One optimisation step per "epoch", on ``batch_size`` axioms drawn at random (with
      replacement) from every normal form
      (:class:`~mowl.training.PerNormalFormMiniBatches`).
    * ``design="regd"``: the loss of the code (``PluginRegD.loss_cross``:
      :class:`~mowl.training.SoftmaxEnergyContrastiveLoss`) on the normal form of the
      task only -- ``gci0`` for ``task="inference"``, ``gci2`` for
      ``task="prediction"`` -- with the model's own negative sampling (10 random
      classes per positive, replacing the superclass of ``gci0`` and the filler of
      ``gci2``). Every other normal form keeps the host model's squared-error loss of
      the RegD energy (:class:`~mowl.training.MSETargetLoss`, MSE to 0, per axiom). For
      the prediction task give the model ``flip_gci2=True``, which is the reversed
      inclusion the code's ``nf3Loss_ours`` trains.
    * ``design="baseline"``: the ELBE / ELEm baselines of the Box\\ :sup:`2`\\ EL code
      trained in the same loop: both-side corruptions of ``gci2``
      (:class:`~mowl.training.PerColumnCorruption`) with the model's own loss (ELBE:
      pass ``neg_target=2`` for its MSE-to-2 negative loss; ``margin`` is a slack
      margin, :class:`~mowl.training.SlackMargin`).
    * Every ``validate_every`` epochs the F1 of the first 1,000 validation axioms of
      the task's normal form is computed as at test time (threshold tuned on that
      split) and the best parameters kept (:class:`~mowl.training.ValidationF1Selection`).
      Model selection thus optimises the reported metric, not the training loss.

    Deviations: the code clips the offsets of the parent at ``exp(-1e-4)`` inside the
    plugin energy and (for the ball host) uses softplus radii; mOWL's RegD modules do
    neither. The RegD models still need ``depth_on_containment=False`` (the code adds the
    depth term only to non-contained pairs) and ``neg_sampling_gcis`` restricted to the
    task's normal form; both are model arguments of the recipe.

    :param task: ``"inference"`` (evaluate on ``C ⊑ D``) or ``"prediction"``
        (evaluate on ``A ⊑ ∃r.B``).
    :param design: ``"regd"`` or ``"baseline"``.
    :param batch_size: Axioms per normal form and step (512 in the code).
    :param num_negatives: Negatives per positive and column (baselines: 1, as in the
        code's ``sample_negatives``; RegD: 10, from the model's own sampler).
    :param margin: Slack margin for the baseline host losses (the code sweeps
        0, 0.05, 0.1, 0.15); ``None`` leaves the model's margin.
    :param neg_target: Target of the baseline negative scores (2 for ELBE in the code;
        ``None`` keeps the model's own loss combination).
    :param validate_every: Epochs between validations (100 in the code).
    :param select: Set to ``False`` to train without model selection.
    :param seed: Seed of the batch and negative samplers.
    """

    key = "regd-code-training"
    _TASK_GCI = {"inference": "gci0", "prediction": "gci2"}

    def __init__(self, task="inference", design="regd", batch_size=512,
                 num_negatives=1, margin=None, neg_target=None, validate_every=100,
                 select=True, seed=None):
        if task not in self._TASK_GCI:
            raise ValueError("task must be 'inference' or 'prediction'.")
        if design not in ("regd", "baseline"):
            raise ValueError("design must be 'regd' or 'baseline'.")
        gci = self._TASK_GCI[task]
        parts = [PerNormalFormMiniBatches(batch_size=batch_size, order="random",
                                          seed=seed)]
        if design == "regd":
            parts.append(SoftmaxEnergyContrastiveLoss(normal_forms=(gci,)))
            parts.append(MSETargetLoss(pos_target=0.0, reduce="per_axiom"))
        else:
            parts.append(PerColumnCorruption(columns={"gci2": (0, 2)},
                                             num_negatives=num_negatives, seed=seed))
            if margin is not None:
                parts.append(SlackMargin(margin))
            if neg_target is not None:
                parts.append(MSETargetLoss(pos_target=0.0, neg_target=neg_target,
                                           reduce="per_axiom"))
        if select:
            parts.append(ValidationF1Selection(every=validate_every, gci_name=gci))
        super().__init__(*parts)
        self.key = type(self).key
