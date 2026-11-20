"""Training designs of reference implementations, as named compositions of regimes.

A paper's numbers depend on how its code trains, and that is often not what the paper's
text or a re-implementation does by default. The regimes here bundle the training design
of one code base, so that a published number can be reproduced by naming one regime.
"""

from mowl.training.regime import ComposedRegime
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.negatives import PerColumnCorruption
from mowl.training.losses import MSETargetLoss, SoftmaxEnergyContrastiveLoss, SumLoss
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


def joint_unit_init(module):
    """Initialise the class regions as the RegD code (and the ELBE / ELEm code it
    adapts) does: one embedding row per class holding the centre *and* the size
    (offsets of a box, radius of a ball), drawn uniformly from [-1, 1] and normalised to
    unit length as a whole. mOWL's ELBE and ELEm modules normalise the centre and the
    size separately, which makes the initial regions larger (unit-norm offsets, radii of
    +-1).

    :param module: A module with ``class_embed`` and ``class_offset`` (boxes) or
        ``class_rad`` (balls).
    """
    import torch as th
    size = getattr(module, "class_offset", None)
    if size is None:
        size = getattr(module, "class_rad", None)
    if size is None:
        raise TypeError(f"{type(module).__name__} has neither class_offset nor class_rad.")
    center = module.class_embed
    with th.no_grad():
        dim = center.weight.shape[1]
        w = th.empty(center.weight.shape[0], dim + size.weight.shape[1],
                     device=center.weight.device).uniform_(-1, 1)
        w /= th.linalg.norm(w, dim=1, keepdim=True)
        center.weight.copy_(w[:, :dim])
        size.weight.copy_(w[:, dim:])


class RegDCodeTraining(ComposedRegime):
    """How ``EXP_ontology/baseline_sweep.py`` of the RegD code (HuiYang1997/RegD) trains
    RegD and the ELBE / ELEm baselines, and how the numbers of Yang and Chen
    (arXiv:2501.17518, Tables 3 and 4) were obtained.

    * Class regions initialised as in the code (:func:`joint_unit_init`, ``init="joint"``).
    * One optimisation step per "epoch", on ``batch_size`` axioms drawn at random (with
      replacement) from every normal form
      (:class:`~mowl.training.PerNormalFormMiniBatches`).
    * ``design="regd"``: the loss of the code (``PluginRegD.loss_cross``:
      :class:`~mowl.training.SoftmaxEnergyContrastiveLoss`) on the normal form of the
      task only -- ``gci0`` for ``task="inference"``, ``gci2`` for
      ``task="prediction"`` -- with the model's own negative sampling (10 random
      classes per positive, redrawn every step, replacing the superclass of ``gci0`` and
      the filler of ``gci2``). Every other normal form is trained on its *raw* RegD
      energy (``raw_energy=True``: the code's ``our_dist`` is not rectified, so a
      contained pair is pulled back towards the boundary): squared error to 0 for the
      box host (``host_loss="mse"``, the code's ``Elbe.forward``), plain mean for the
      ball host (``host_loss="mean"``, ``Elem.forward``, which adds the centre-norm
      regulariser that mOWL's ELEm module applies to all classes). For the prediction
      task give the model ``flip_gci2=True``: the code's ``nf3Loss_ours`` trains the
      reversed inclusion.
    * ``design="baseline"``: the ELBE / ELEm baselines (the code with no plugin): both-
      side corruptions of ``gci2`` (:class:`~mowl.training.PerColumnCorruption`) with
      the model's own loss (ELBE: pass ``neg_target=2`` for its MSE-to-2 negative loss;
      ``margin`` is a slack margin, :class:`~mowl.training.SlackMargin`).
    * Every ``validate_every`` epochs the F1 of the first 1,000 validation axioms of
      the task's normal form is computed as at test time (threshold tuned on that
      split, the code's positive score for the prediction task) and the best parameters
      are kept (:class:`~mowl.training.ValidationF1Selection`). Model selection thus
      optimises the reported metric, not the training loss.

    Deviations (see the regime card): the code clips the parent offsets at
    ``exp(-1e-4)`` inside the plugin energy; the baselines' negatives are drawn once
    before training in the code and every step here; the ELEm baseline of the code uses
    softplus radii, an unrectified negative loss and a containment loss for
    :math:`\\exists r.C \\sqsubseteq D`, mOWL's ELEm module does not. The RegD models need
    ``depth_on_containment=False`` and ``neg_sampling_gcis`` restricted to the task's
    normal form; both are model arguments of the recipe.

    :param task: ``"inference"`` (evaluate on ``C ⊑ D``) or ``"prediction"``
        (evaluate on ``A ⊑ ∃r.B``).
    :param design: ``"regd"`` or ``"baseline"``.
    :param batch_size: Axioms per normal form and step (512 in the code).
    :param num_negatives: Negatives per positive and column for the baselines (1, as in
        the code's ``sample_negatives``).
    :param margin: Slack margin for the baseline host losses (the code sweeps
        0, 0.05, 0.1, 0.15); ``None`` leaves the model's margin.
    :param neg_target: Target of the baseline negative scores (2 for ELBE in the code;
        ``None`` keeps the model's own loss combination).
    :param host_loss: ``"mse"`` or ``"mean"``: loss of the non-task normal forms for
        ``design="regd"``.
    :param raw_energy: Train the RegD normal forms on the raw energy (the code) instead
        of :math:`\\max\\{E, \\gamma_1\\}` (the paper).
    :param init: ``"joint"`` (the code's initialisation) or ``None`` (the module's).
    :param reference_ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``, for the code's
        positive score of the prediction task in model selection; without it, model
        selection scores the axiom itself.
    :param validate_every: Epochs between validations (100 in the code).
    :param select: Set to ``False`` to train without model selection.
    :param seed: Seed of the batch and negative samplers.
    """

    key = "regd-code-training"
    _TASK_GCI = {"inference": "gci0", "prediction": "gci2"}

    def __init__(self, task="inference", design="regd", batch_size=512,
                 num_negatives=1, margin=None, neg_target=None, host_loss="mse",
                 raw_energy=True, init="joint", reference_ontology=None,
                 validate_every=100, select=True, seed=None):
        if task not in self._TASK_GCI:
            raise ValueError("task must be 'inference' or 'prediction'.")
        if design not in ("regd", "baseline"):
            raise ValueError("design must be 'regd' or 'baseline'.")
        if host_loss not in ("mse", "mean"):
            raise ValueError("host_loss must be 'mse' or 'mean'.")
        if init not in ("joint", None):
            raise ValueError("init must be 'joint' or None.")
        gci = self._TASK_GCI[task]
        parts = [PerNormalFormMiniBatches(batch_size=batch_size, order="random",
                                          seed=seed)]
        if design == "regd":
            parts.append(SoftmaxEnergyContrastiveLoss(normal_forms=(gci,)))
            if host_loss == "mse":
                parts.append(MSETargetLoss(pos_target=0.0, reduce="per_axiom"))
            else:
                parts.append(SumLoss())
        else:
            parts.append(PerColumnCorruption(columns={"gci2": (0, 2)},
                                             num_negatives=num_negatives, seed=seed))
            if margin is not None:
                parts.append(SlackMargin(margin))
            if neg_target is not None:
                parts.append(MSETargetLoss(pos_target=0.0, neg_target=neg_target,
                                           reduce="per_axiom"))
        if select:
            parts.append(ValidationF1Selection(
                every=validate_every, gci_name=gci,
                positive_scores="reference" if reference_ontology else "pair",
                reference_ontology=reference_ontology))
        super().__init__(*parts)
        self.key = type(self).key
        self.task = task
        self.design = design
        self.raw_energy = bool(raw_energy) and design == "regd"
        self.init = init

    def attach(self, model):
        super().attach(model)
        module = model.module
        if self.init == "joint":
            joint_unit_init(module)
        if self.raw_energy:
            if not hasattr(module, "energy_mode"):
                raise TypeError("raw_energy=True needs a RegD module (mowl.nn.el.regd).")
            module.energy_mode = True

    def on_train_end(self):
        super().on_train_end()
        if self.raw_energy:
            self.model.module.energy_mode = False
