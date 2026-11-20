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
    """How the TransBox code (HuiYang1997/TransBox, ``model/TransBox_reg.py`` and
    ``baseline_sweep.py``) trains TransBox, and how the numbers of Yang et al. (WWW 2025)
    were obtained.

    * One step per "epoch" on 1024 axioms drawn at random from every normal form,
      including role inclusions and chains (:class:`~mowl.training.PerNormalFormMiniBatches`).
    * Every per-axiom loss is squared and averaged, :math:`\\mathrm{mean}(\\ell^2)`
      (the paper's Eq. 14); negatives of :math:`C \\sqsubseteq \\exists r.D` (each side
      corrupted, :class:`~mowl.training.PerColumnCorruption`) contribute
      :math:`\\mathrm{mean}((\\delta - \\ell^-)^2)` with :math:`\\delta` = ``neg_dist``
      (:class:`~mowl.training.MSETargetLoss`).
    * Slack margin (:class:`~mowl.training.SlackMargin`), if ``margin`` is given (the
      paper writes a strict one).
    * Adam, no schedule; every ``validate_every`` epochs the median rank of the first
      1000 validation axioms :math:`C \\sqsubseteq D` selects the parameters
      (:class:`~mowl.training.ValidationRankSelection`).

    The per-axiom losses themselves are those of the code: for :class:`~mowl.models.TransBox`
    this regime requires ``formulation="reference"``
    (:mod:`mowl.nn.el.transbox.code_losses`), whose negative loss is an overlap that the
    squared target loss pushes to ``neg_dist``.

    Deviations, not replicated: the code draws the negatives once before training and
    takes 1024 of them per step independently of the positives (here: drawn anew from
    the positives of the step); the code draws 1024 axioms with replacement also from
    normal forms with fewer axioms (here: all of them, same expectation); in the code the
    loss of :math:`C \\sqcap D \\sqsubseteq E` is, by a broadcasting slip, the mean over
    all pairs :math:`(i, j)` of :math:`(p_i + q_j)^2` (emptiness penalty of axiom
    :math:`i`, inclusion term of axiom :math:`j`) instead of :math:`(p_i + q_i)^2`, which
    changes only the cross term :math:`2\\bar p \\bar q` versus
    :math:`2\\overline{pq}`.

    :param batch_size: Axioms per normal form and step (1024 in the code).
    :param num_negatives: Negatives per side and positive (``num_neg``).
    :param margin: Slack :math:`\\gamma`; ``None`` leaves the model's margin unchanged.
    :param neg_dist: Target of the negative losses (``neg_dist``, 1 in all configurations).
    :param lr_milestones: Epochs at which the learning rate is multiplied by 0.1 (none in
        the code).
    :param validate_every: Epochs between validations (100).
    :param select: Set to ``False`` to train without model selection.
    :param seed: Seed of the batch and negative samplers.
    """

    key = "transbox-code-training"
    _default_batch_size = 1024

    def __init__(self, batch_size=None, num_negatives=1, margin=None, neg_dist=1.0,
                 lr_milestones=None, validate_every=100, select=True, seed=None):
        super().__init__(batch_size=batch_size, num_negatives=num_negatives,
                         margin=margin, neg_target=neg_dist, lr_milestones=lr_milestones,
                         validate_every=validate_every, select=select, seed=seed)
        self.key = type(self).key
        self.neg_dist = float(neg_dist)

    def attach(self, model):
        formulation = getattr(getattr(model, "module", None), "formulation", None)
        if formulation is not None and formulation != "reference":
            raise TypeError(
                "transbox-code-training squares the per-axiom losses of the TransBox code "
                "and pushes negatives to neg_dist; build the model with "
                "formulation='reference' (mOWL's default TransBox losses already square "
                "their negative loss).")
        super().attach(model)
