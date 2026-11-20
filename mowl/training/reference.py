"""Training designs of reference implementations, as named compositions of regimes.

A paper's numbers depend on how its code trains, and that is often not what the paper's
text or a re-implementation does by default. The regimes here bundle the training design
of one code base, so that a published number can be reproduced by naming one regime.
"""

from mowl.training.regime import ComposedRegime
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.negatives import PerColumnCorruption, FixedNegatives
from mowl.training.initialisation import JointNormalisedInitialisation
from mowl.training.loss_forms import Box2ELBaselineLosses
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

    The baselines (ELEm, EmEL++, ELBE, BoxEL; ``model/*.py`` of the code) are trained
    differently from Box\\ :sup:`2`\\ EL in three respects, each an option here and all
    three switched on by ``baseline_code=True``:

    * ``fixed_negatives``: the negatives are drawn once before training, and every step
      takes ``batch_size`` of them at random (:class:`~mowl.training.FixedNegatives`);
      Box\\ :sup:`2`\\ EL redraws them at every step;
    * ``initialisation="joint"``: centre and radius (offset) of a class are initialised as
      one vector of unit norm (:class:`~mowl.training.JointNormalisedInitialisation`), so
      the ELEm radii start near 0.06 instead of 1;
    * ``baseline_losses``: the loss details of the baseline code
      (:class:`~mowl.training.Box2ELBaselineLosses`: no hinge on the negative loss of
      ELEm and EmEL++, per-axiom unit-norm penalty of ELEm, no radius term in EmEL++'s
      :math:`C \\sqcap D \\sqsubseteq E`, summed losses and the code's regulariser for
      BoxEL).

    :param batch_size: Axioms per normal form and step (512 in the code).
    :param num_negatives: Negatives per side and positive (``num_neg`` of the code).
    :param margin: Slack :math:`\\gamma`; ``None`` leaves the model's margin unchanged.
    :param neg_target: Target of the negative scores for a per-axiom squared loss, or
        ``None`` to keep the model's own combination of positive and negative losses.
    :param lr_milestones: Epochs at which the learning rate is multiplied by 0.1.
    :param validate_every: Epochs between validations.
    :param select: Set to ``False`` to train without model selection.
    :param fixed_negatives: Draw the negatives once before training (baselines).
    :param initialisation: ``None`` (the module's own) or ``"joint"`` (baselines).
    :param baseline_losses: Apply :class:`~mowl.training.Box2ELBaselineLosses`.
    :param baseline_code: Shorthand for all three baseline options.
    :param seed: Seed of the batch and negative samplers.
    """

    key = "box2el-code-training"
    _default_batch_size = 512

    def __init__(self, batch_size=None, num_negatives=1, margin=None, neg_target=None,
                 lr_milestones=None, validate_every=100, select=True, fixed_negatives=False,
                 initialisation=None, baseline_losses=False, baseline_code=False,
                 seed=None):
        batch_size = self._default_batch_size if batch_size is None else batch_size
        if baseline_code:
            fixed_negatives, baseline_losses = True, True
            initialisation = "joint" if initialisation is None else initialisation
        if initialisation not in (None, "joint"):
            raise ValueError("initialisation must be None or 'joint'.")
        corruption = PerColumnCorruption(columns={"gci2": (0, 2)},
                                         num_negatives=num_negatives, seed=seed)
        if fixed_negatives:
            corruption = FixedNegatives(corruption, batch_size=batch_size, seed=seed)
        parts = [PerNormalFormMiniBatches(batch_size=batch_size, order="random",
                                          seed=seed),
                 corruption]
        if initialisation == "joint":
            # BoxEL has no such pair; the code initialises it as mOWL does.
            parts.append(JointNormalisedInitialisation(strict=False, seed=seed))
        if baseline_losses:
            parts.append(Box2ELBaselineLosses())
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


class EmELppCodeTraining(ComposedRegime):
    """How the Keras code of EmEL++ (kracr/EmELpp, ``experiments/*/models/*_EmEL.py``)
    trains EmEL++, and its ``*_EL.py`` scripts train the ELEmbeddings baseline; the
    numbers of Mondal et al. (AAAI-MAKE 2021, Table 3) were obtained this way.

    * Training data: the normalised ontology without the test subsumptions but *with* the
      validation subsumptions (:class:`~mowl.training.ValidationAxiomsAsPositives`).
    * A step draws ``batch_size`` (256) axioms with replacement from every normal form
      (:class:`~mowl.training.PerNormalFormMiniBatches`, ``sample_small=True``); an
      epoch is :math:`\\lceil N / 256 \\rceil` steps, :math:`N` the size of the largest
      input table, and the code trains 1,000 epochs (the recipes give the step counts).
    * Negatives of :math:`C \\sqsubseteq \\exists R.D` are drawn once before training,
      one with :math:`C` and one with :math:`D` replaced by a *different* class from the
      classes of the subsumption pairs (:class:`~mowl.training.FixedNegatives`).
    * Centre and radius are initialised jointly to unit norm, with the radius coordinate
      drawn from :math:`[0, 1]` for EmEL++ and :math:`[-1, 1]` for ELEm
      (:class:`~mowl.training.JointNormalisedInitialisation`).
    * The objective is the Keras one (:class:`~mowl.training.EmELppCodeObjective`): the
      squared row sums of all losses, the negative loss without hinge, a learned
      translation for :math:`C \\sqsubseteq D`, the radius of ``owl:Thing`` pulled to 5.
    * After every epoch (``validate_every`` steps) the mean rank of :math:`D` for the first
      100 validation axioms :math:`C \\sqsubseteq D` (which are training axioms) is
      computed with the model's :math:`C \\sqsubseteq D` loss, and the best parameters are
      kept (:class:`~mowl.training.ValidationRankSelection`).

    Deviations: the code draws the replacement classes from the classes of the training,
    validation *and test* pairs; here only training and validation pairs are used. The
    role inclusion and chain losses of the paper compare the role vectors row by row; the
    code normalises along the batch axis (``tf.nn.l2_normalize(r, 0)``), which turns the
    direction term into one number per batch. This bug is reproduced by default
    (``batch_direction=True``) because the published numbers were obtained with it.

    :param batch_size: Axioms per normal form and step.
    :param validate_every: Steps between validations (one epoch of the code).
    :param select: Set to ``False`` to train without model selection.
    :param batch_direction: See :class:`~mowl.training.EmELppCodeObjective` (``True``:
        the direction term of the role axiom losses as computed by the code).
    :param seed: Seed of the samplers and of the initialisation.
    """

    key = "emelpp-code-training"

    def __init__(self, batch_size=256, validate_every=1, select=True, batch_direction=True,
                 seed=None):
        from mowl.training.positives import ValidationAxiomsAsPositives
        from mowl.training.loss_forms import EmELppCodeObjective
        self._corruption = PerColumnCorruption(columns={"gci2": (0, 2)}, different=True,
                                               seed=seed)
        self._init = JointNormalisedInitialisation(seed=seed)
        parts = [ValidationAxiomsAsPositives(),
                 PerNormalFormMiniBatches(batch_size=batch_size, order="random",
                                          sample_small=True, seed=seed),
                 FixedNegatives(self._corruption, batch_size=batch_size, seed=seed),
                 self._init,
                 EmELppCodeObjective(batch_size=batch_size,
                                     batch_direction=batch_direction, seed=seed)]
        if select:
            parts.append(ValidationRankSelection(every=validate_every, statistic="mean",
                                                 max_queries=100, scoring="loss",
                                                 target_col=1, ties="average"))
        super().__init__(*parts)
        self.key = type(self).key
        self._select = bool(select)

    def attach(self, model):
        import numpy as np
        module = model.module
        is_emelpp = any(c.__name__ == "EmELPPModule" for c in type(module).__mro__)
        self._init.size_low = 0.0 if is_emelpp else -1.0
        # Replacement classes: the classes of the subsumption pairs (training and
        # validation), as the code's 'prot_ids'.
        named = set(getattr(model, "named_class_index_dict", {}).values())
        ids = set()
        for split in ("training_datasets", "validation_datasets"):
            try:
                ds = (getattr(model, split) or {}).get("gci0")
            except AttributeError:  # no validation ontology
                ds = None
            if ds is not None and len(ds) > 0:
                ids.update(int(i) for i in ds[:].flatten().tolist())
        if named:
            ids &= named
        if ids:
            pool = np.asarray(sorted(ids), dtype=np.int64)
            self._corruption.pools = {"gci2": {0: pool, 2: pool}}
        super().attach(model)
        if self._select and getattr(model, "eval_gci_name", "") is None:
            model.eval_gci_name = "gci0"
