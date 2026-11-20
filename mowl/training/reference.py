"""Training designs of reference implementations, as named compositions of regimes.

A paper's numbers depend on how its code trains, and that is often not what the paper's
text or a re-implementation does by default. The regimes here bundle the training design
of one code base, so that a published number can be reproduced by naming one regime.
"""

from mowl.training.regime import ComposedRegime, TrainingRegime
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.negatives import PerColumnCorruption, FixedColumnCorruption, \
    EitherSideCorruption
from mowl.training.losses import MSETargetLoss, PearsonCorrelationLoss
from mowl.training.margin import SlackMargin
from mowl.training.schedules import StepDecay
from mowl.training.selection import ValidationRankSelection, ValidationSelection, \
    validation_rank


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


class _ELKEmAxiomPreparation(TrainingRegime):
    """Training axioms as the ELK-Em code prepares them: ``gci0`` rows whose subclass is
    fixed to the empty box are dropped (they hold trivially) and ``gci1``/``gci3`` rows
    whose superclass is fixed empty are trained as ``gci1_bot``/``gci3_bot``. This is
    :meth:`mowl.models.elkem.model.ELKEm._training_tensors` as a regime hook.
    """

    def training_datasets(self, datasets):
        return self.model._training_tensors()


def _elkem_validation_median_rank(model):
    """Median rank of all validation ``gci0`` axioms as the ELK-Em code computes it:
    candidates are the named classes except :math:`\\top`, the subclass is removed from
    its own query, scores are margin-free and ties are averaged
    (``paper/ranking/eval.py``, ``evaluate_ranking_metrics``)."""
    import torch as th
    top = set(getattr(model, "_top_ids", None) or ())
    cands = th.tensor(sorted(i for i in model.named_class_index_dict.values()
                              if i not in top), dtype=th.long)
    return validation_rank(model, gci_name="gci0", target_col=1, statistic="median",
                           max_queries=None, candidates=cands, scoring="margin_free",
                           ties="average", exclude_cols=(0,))


class ELKEmCodeTraining(ComposedRegime):
    """How ``paper/ranking/train.py`` of the ELK-Em code
    (city-artificial-intelligence/elk-em) trains, and how the numbers of Singh,
    Jiménez-Ruiz & Weyde (KG-NeSy 2026, Table 3) were obtained.

    * One optimisation step per "epoch", on ``batch_size`` axioms drawn at random with
      replacement from every normal form
      (:class:`~mowl.training.PerNormalFormMiniBatches`).
    * Axioms of concepts fixed to the empty box are dropped or rerouted to their
      ``_bot`` normal form, and ``nf1_bot`` is not trained (the boxes are fixed);
      :class:`~mowl.models.elkem.model.ELKEm` implements this itself.
    * The positive per-axiom losses of every normal form are summed (the model's
      ``compute_loss``; the code squares them before averaging, see ``notes``).
    * Negatives only for the Pearson contrastive term
      (:class:`~mowl.training.PearsonCorrelationLoss`, weight
      ``contrastive_factor`` = 3.0 in the code): ``k_neg`` corruptions per positive of
      :math:`C \\sqsubseteq D` with :math:`D` replaced
      (:class:`~mowl.training.FixedColumnCorruption`) and of
      :math:`C \\sqsubseteq \\exists r.D` with either side corrupted
      (:class:`~mowl.training.EitherSideCorruption`), drawn from the named classes
      except :math:`\\top`, :math:`\\bot` and the unsatisfiable concepts. Corruptions
      are not filtered against entailed axioms.
    * The learning rate is multiplied by 0.1 after each epoch of ``lr_milestones``
      (:class:`~mowl.training.StepDecay`; 5,000 in the paper).
    * Every ``validate_every`` epochs the median rank of *all* validation subsumptions
      is computed (margin-free scores, average ties, :math:`\\top` and the query removed
      from the candidates) and the best parameters are kept
      (:class:`~mowl.training.ValidationSelection`).

    Only the TBox design of the ranking experiments; the ABox losses and the lexical
    regulariser of the protein-function experiments stay with the model's own loop.

    Differences from the authors' code, accepted deliberately (see the regime card
    'elkem-code-training'):

    * the correlation is computed from margin-ful module scores, whereas the code sets
      the margin to zero while scoring positives and corruptions;
    * the per-axiom losses are not squared and not multiplied by 1,000;
    * ``different=True`` redraws a replacement equal to the corrupted column only, so a
      rare negative :math:`C \\sqsubseteq C` can appear (the code also redraws
      :math:`D = C`);
    * the corruption pool excludes :math:`\\top`, :math:`\\bot` and the unsatisfiable
      concepts; the code draws from all class indices except the first and the last.

    :param batch_size: Axioms per normal form and step (1,024 in the code).
    :param k_neg: Corruptions per positive (3 for GALEN and Anatomy, 5 for GO).
    :param contrastive_factor: Weight of the Pearson term (3.0 in the code).
    :param lr_milestones: Epochs at which the learning rate is multiplied by 0.1.
    :param validate_every: Epochs between validations.
    :param select: Set to ``False`` to train without model selection.
    :param seed: Seed of the batch and negative samplers.
    """

    key = "elkem-code-training"

    def __init__(self, batch_size=1024, k_neg=3, contrastive_factor=3.0,
                 lr_milestones=(5000,), validate_every=100, select=True, seed=None):
        super().__init__()
        self.batch_size = int(batch_size)
        self.k_neg = int(k_neg)
        self.contrastive_factor = float(contrastive_factor)
        self.lr_milestones = tuple(lr_milestones or ())
        self.validate_every = max(1, int(validate_every))
        self.select = bool(select)
        self.seed = seed
        self.key = type(self).key

    def _parts(self):
        parts = [_ELKEmAxiomPreparation(),
                 PerNormalFormMiniBatches(batch_size=self.batch_size, order="random",
                                          seed=self.seed)]
        return parts

    def attach(self, model):
        pool = sorted(i for i in model.named_class_index_dict.values()
                      if i not in set(model._bot_ids) and i not in set(model._top_ids))
        parts = self._parts() + [
            FixedColumnCorruption(
                columns={"gci0": (1,)}, num_negatives=self.k_neg, different=True,
                require_negative_loss=False, pools={"gci0": {1: pool}}, seed=self.seed),
            EitherSideCorruption(
                columns={"gci2": (0, 2)}, num_negatives=self.k_neg, different=True,
                require_negative_loss=False, pools={"gci2": {0: pool, 2: pool}},
                seed=self.seed),
            PearsonCorrelationLoss(weight=self.contrastive_factor,
                                   normal_forms=("gci0", "gci2")),
        ]
        if self.lr_milestones:
            parts.append(StepDecay(self.lr_milestones, gamma=0.1))
        if self.select:
            parts.append(ValidationSelection(metric=_elkem_validation_median_rank,
                                             every=self.validate_every))
            parts[-1].key = "validation-rank-selection"
        self.regimes = tuple(parts)
        super().attach(model)
        if self.select and getattr(model, "eval_gci_name", "") is None:
            model.eval_gci_name = "gci0"

