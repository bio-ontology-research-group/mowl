"""Training designs of reference implementations, as named compositions of regimes.

A paper's numbers depend on how its code trains, and that is often not what the paper's
text or a re-implementation does by default. The regimes here bundle the training design
of one code base, so that a published number can be reproduced by naming one regime.
"""

from mowl.training.regime import ComposedRegime, TrainingRegime
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.negatives import PerColumnCorruption, FixedColumnCorruption, \
    EitherSideCorruption, FixedNegatives
from mowl.training.initialisation import JointNormalisedInitialisation
from mowl.training.loss_forms import Box2ELBaselineLosses
from mowl.training.losses import MSETargetLoss, PearsonCorrelationLoss, \
    SoftmaxEnergyContrastiveLoss, SumLoss
from mowl.training.margin import SlackMargin
from mowl.training.schedules import StepDecay
from mowl.training.selection import ValidationRankSelection, ValidationSelection, \
    ValidationF1Selection, validation_rank


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
    :param nf2_pairwise: Replicate the broadcasting slip of the code for
        :math:`C \\sqcap D \\sqsubseteq E`: loss :math:`\\mathrm{mean}_{i,j}(p_i + q_j)^2`
        instead of :math:`\\mathrm{mean}_i(p_i + q_i)^2` (default ``False``).
    """

    key = "transbox-code-training"
    _default_batch_size = 1024

    def __init__(self, batch_size=None, num_negatives=1, margin=None, neg_dist=1.0,
                 lr_milestones=None, validate_every=100, select=True, seed=None,
                 nf2_pairwise=False):
        super().__init__(batch_size=batch_size, num_negatives=num_negatives,
                         margin=margin, neg_target=neg_dist, lr_milestones=lr_milestones,
                         validate_every=validate_every, select=select, seed=seed)
        self.key = type(self).key
        self.neg_dist = float(neg_dist)
        self.nf2_pairwise = bool(nf2_pairwise)
        self._datasets = {}

    def training_datasets(self, datasets):
        self._datasets = super().training_datasets(datasets)
        return self._datasets

    def loss(self, gci_name, pos_scores, neg_scores):
        if self.nf2_pairwise and gci_name == "gci1" and "gci1" in self._datasets:
            from mowl.nn.el.transbox.code_losses import gci1_terms
            module = self.model.module
            p, q = gci1_terms(self._datasets["gci1"][:], module.class_center,
                              module.class_offset, module.margin)
            return ((p[:, None] + q[None, :]) ** 2).mean()
        return super().loss(gci_name, pos_scores, neg_scores)

    def attach(self, model):
        formulation = getattr(getattr(model, "module", None), "formulation", None)
        if formulation is not None and formulation != "reference":
            raise TypeError(
                "transbox-code-training squares the per-axiom losses of the TransBox code "
                "and pushes negatives to neg_dist; build the model with "
                "formulation='reference' (mOWL's default TransBox losses already square "
                "their negative loss).")
        super().attach(model)


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
    ids = sorted(i for i in model.named_class_index_dict.values() if i not in top)
    cands = th.tensor(ids, dtype=th.long)
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
    * The per-axiom losses of every normal form are squared and averaged, and the means
      are summed (``positive_power=2``: ``loss_fn(batch).pow(2).mean()`` in the code;
      :class:`~mowl.training.MSETargetLoss` with target 0 for the normal forms without
      negatives). ``positive_power=1`` gives the plain sum of Table 2 of the paper.
    * Negatives only for the Pearson contrastive term
      (:class:`~mowl.training.PearsonCorrelationLoss`, weight
      ``contrastive_factor`` = 3.0 in the code): ``k_neg`` corruptions per positive of
      :math:`C \\sqsubseteq D` with :math:`D` replaced
      (:class:`~mowl.training.FixedColumnCorruption`) and of
      :math:`C \\sqsubseteq \\exists r.D` with either side corrupted
      (:class:`~mowl.training.EitherSideCorruption`). The correlation is computed from
      the *margin-free* scores (``margin_free_contrast=True``; the code sets the margin
      to 0 while scoring positives and corruptions). Corruptions are not filtered
      against entailed axioms. They are drawn from all classes except :math:`\\top` and
      ``owl:Nothing`` (``corruption_pool="code"``: the code draws class indices
      ``1 .. M-2``), or, with ``corruption_pool="satisfiable"``, also without the
      classes fixed to the empty box.
    * The learning rate is multiplied by 0.1 after each epoch of ``lr_milestones``
      (:class:`~mowl.training.StepDecay`; 5,000 in the paper).
    * Every ``validate_every`` epochs the median rank of *all* validation subsumptions
      is computed (margin-free scores, average ties, :math:`\\top` and the query removed
      from the candidates) and the best parameters are kept
      (:class:`~mowl.training.ValidationSelection`).

    The geometry of the code (interval-arithmetic existential boxes, an overlap term in
    the ``gci2`` loss) is a property of the model, not of the training: construct
    :class:`~mowl.models.elkem.model.ELKEm` with ``variant="reference"``.

    Only the TBox design of the ranking experiments; the ABox losses and the lexical
    regulariser of the protein-function experiments stay with the model's own loop.

    Remaining differences from the authors' code (see the regime card
    'elkem-code-training'):

    * the code multiplies the total loss by 1,000; Adam is invariant to a constant
      factor of the loss (up to its :math:`\\epsilon`), so this is left out;
    * ``different=True`` redraws a replacement equal to the corrupted column only, so a
      rare negative :math:`C \\sqsubseteq C` can appear (the code also redraws
      :math:`D = C`);
    * the code draws the ``gci2`` positives of the Pearson term and of the positive
      loss as two independent batches; here they are the same batch.

    :param batch_size: Axioms per normal form and step (1,024 in the code).
    :param k_neg: Corruptions per positive (3 for GALEN and Anatomy, 5 for GO).
    :param contrastive_factor: Weight of the Pearson term (3.0 in the code).
    :param lr_milestones: Epochs at which the learning rate is multiplied by 0.1.
    :param validate_every: Epochs between validations.
    :param select: Set to ``False`` to train without model selection.
    :param positive_power: 2 (code: squared per-axiom losses) or 1 (paper).
    :param margin_free_contrast: Correlate margin-free scores (code) or the
        training-mode scores with margin.
    :param corruption_pool: ``"code"`` or ``"satisfiable"`` (see above).
    :param seed: Seed of the batch and negative samplers.
    """

    key = "elkem-code-training"

    def __init__(self, batch_size=1024, k_neg=3, contrastive_factor=3.0,
                 lr_milestones=(5000,), validate_every=100, select=True, positive_power=2,
                 margin_free_contrast=True, corruption_pool="code", seed=None):
        super().__init__()
        if corruption_pool not in ("code", "satisfiable"):
            raise ValueError("corruption_pool must be 'code' or 'satisfiable'.")
        self.batch_size = int(batch_size)
        self.k_neg = int(k_neg)
        self.contrastive_factor = float(contrastive_factor)
        self.lr_milestones = tuple(lr_milestones or ())
        self.validate_every = max(1, int(validate_every))
        self.select = bool(select)
        self.positive_power = int(positive_power)
        self.margin_free_contrast = bool(margin_free_contrast)
        self.corruption_pool = corruption_pool
        self.seed = seed
        self.key = type(self).key

    def _parts(self):
        parts = [_ELKEmAxiomPreparation(),
                 PerNormalFormMiniBatches(batch_size=self.batch_size, order="random",
                                          seed=self.seed)]
        return parts

    def _pool(self, model):
        excluded = set(model._top_ids)
        if self.corruption_pool == "code":
            nothing = model.class_index_dict.get("http://www.w3.org/2002/07/owl#Nothing")
            if nothing is not None:
                excluded.add(nothing)
        else:
            excluded |= set(model._bot_ids)
        return sorted(i for i in model.named_class_index_dict.values() if i not in excluded)

    def attach(self, model):
        pool = self._pool(model)
        parts = self._parts() + [
            FixedColumnCorruption(
                columns={"gci0": (1,)}, num_negatives=self.k_neg, different=True,
                require_negative_loss=False, pools={"gci0": {1: pool}}, seed=self.seed),
            EitherSideCorruption(
                columns={"gci2": (0, 2)}, num_negatives=self.k_neg, different=True,
                require_negative_loss=False, pools={"gci2": {0: pool, 2: pool}},
                seed=self.seed),
            PearsonCorrelationLoss(weight=self.contrastive_factor,
                                   normal_forms=("gci0", "gci2"),
                                   positive_power=self.positive_power,
                                   margin_free=self.margin_free_contrast),
        ]
        if self.positive_power == 2:
            # every other normal form: mean of the squared per-axiom losses
            parts.append(MSETargetLoss(pos_target=0.0, reduce="per_axiom"))
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
    :param rectified: The variant with ``relu`` applied to every RegD energy (the line
        commented out in ``PluginRegD.our_dist``): the other normal forms train on
        :math:`\\max\\{E, 0\\}` and the task's negatives exert no force. Not the
        released code; an ablation.
    :param score: Score of model selection, ``"boundary"`` (the released code) or
        ``"energy"`` (its ``depth_type='norm'`` branch); use the same as the evaluation.
    """

    key = "regd-code-training"
    _TASK_GCI = {"inference": "gci0", "prediction": "gci2"}

    def __init__(self, task="inference", design="regd", batch_size=512,
                 num_negatives=1, margin=None, neg_target=None, host_loss="mse",
                 raw_energy=True, init="joint", reference_ontology=None,
                 validate_every=100, select=True, seed=None, score="boundary",
                 rectified=False):
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
            parts.append(SoftmaxEnergyContrastiveLoss(normal_forms=(gci,),
                                                      rectify_negatives=rectified))
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
                reference_ontology=reference_ontology, score=score))
        super().__init__(*parts)
        self.key = type(self).key
        self.task = task
        self.design = design
        self.raw_energy = bool(raw_energy) and design == "regd" and not rectified
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
