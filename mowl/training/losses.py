"""Regimes that decide how the scores of positives and negatives are combined into a loss.

In mOWL the module returns, for every axiom, a non-negative *loss score* that is 0 when the
axiom holds in the embedding (lower is "more true"). Papers combine these scores in
different ways, and the combination is part of the training regime:

* :class:`SumLoss`: mean positive loss plus (weighted) mean negative loss -- the default of
  mOWL's models, ELEmbeddings, EmEL++, DELE, BoxEL (negative weight :math:`\\phi`);
* :class:`MSETargetLoss`: squared distance of every score to a target (0 for positives, a
  target :math:`t` for negatives) -- the reference implementation of ELBE;
* :class:`PairwiseMarginRankingLoss`: :math:`\\max(0, \\gamma + s^+ - s^-)` -- the triplet
  loss of HiT and OnT, TransE-style ranking;
* :class:`LogSigmoidLoss`: :math:`-\\log\\sigma(\\gamma - s^+) - \\sum_i \\alpha_i
  \\log\\sigma(s^-_i - \\gamma)`, optionally with self-adversarial weights
  :math:`\\alpha` -- Query2Box, ConE, AConE, octagon embeddings, binary-code boxes;
* :class:`BPRLoss`: :math:`-\\log\\sigma(s^- - s^+)` -- FALCON's ABox loss;
* :class:`LogSumExpContrastiveLoss`: the contrastive loss of RegD (from shadow cones);
* :class:`PearsonCorrelationLoss`: one minus the Pearson correlation of scores and labels
  -- ELK-Em;
* :class:`ProbabilisticBCELoss`: binary cross-entropy of :math:`p = e^{-s}` -- binary-code
  boxes, Gumbel boxes;
* :class:`NormalFormFrequencyWeighting`: weights the loss of every normal form by its
  share of the training data -- the NeSy 2024 paper of Mashkova et al.

**Scores of negatives.** The generic training loop scores negatives with
``module(neg_data, gci_name, neg=True)``: the module's *negative loss*, which is small when
the negative does *not* hold. A sum of losses (:class:`SumLoss`, :class:`MSETargetLoss`)
wants exactly that. All other losses here compare positives and negatives on the *same*
scale and need the ordinary score of the negatives, ``module(neg_data, gci_name)``. Since
the loss hook does not receive the negative rows, these regimes record them in the
``filter_negatives`` hook and re-score them (``neg_scores="positive"``, the default).
Put such a regime *last* in a :class:`~mowl.training.ComposedRegime`, so that it records
the negatives after every other filter. With ``neg_scores="module"`` they use the scores
the loop passes in.

**Alignment.** Several losses pair every positive with its own negatives. They assume the
layout of :mod:`mowl.training.negatives` and of the model's default sampler: ``K * n``
negative rows for ``n`` positives, row ``i`` belonging to positive ``i mod n``. When the
number of negatives is not a multiple of the number of positives (a filter dropped some),
pairwise losses fall back to comparing every positive with the mean over all negatives
and record this in :attr:`num_unaligned`.
"""

import math

import torch as th
import torch.nn.functional as F

from mowl.training.regime import TrainingRegime
from mowl.training.terms import pearson_correlation


def _flat(x):
    return None if x is None else x.reshape(len(x), -1).mean(dim=1)


class _ScoreLoss(TrainingRegime):
    """Base of the loss regimes. Subclasses implement :meth:`combine`."""

    #: Whether the loss needs ordinary scores of the negatives (see module docstring).
    needs_positive_form = True

    def __init__(self, normal_forms=None, neg_scores="positive"):
        super().__init__()
        if neg_scores not in ("positive", "module"):
            raise ValueError("neg_scores must be 'positive' or 'module'.")
        self.normal_forms = None if normal_forms is None else tuple(normal_forms)
        self.neg_scores = neg_scores
        self._last_negatives = {}
        self.num_unaligned = 0

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        self._last_negatives[gci_name] = negatives
        return negatives

    def scores_negatives_like_positives(self, gci_name):
        # The loss re-scores the negatives with the ordinary score anyway; the loop must not
        # ask the module for a negative loss it may not have (and would not use).
        if self.normal_forms is not None and gci_name not in self.normal_forms:
            return None
        return self.neg_scores == "positive" and self.needs_positive_form

    def _negative_scores(self, gci_name, neg_scores):
        if neg_scores is None:
            return None
        if self.neg_scores == "module" or not self.needs_positive_form:
            return _flat(neg_scores)
        rows = self._last_negatives.get(gci_name)
        if rows is None or len(rows) != len(neg_scores):
            raise RuntimeError(
                f"{type(self).__name__} could not find the negatives of {gci_name!r} it "
                f"needs to re-score. Put it last in a ComposedRegime (it records the "
                f"negatives in filter_negatives), or pass neg_scores='module'.")
        return _flat(self.model.module(rows, gci_name))

    def loss(self, gci_name, pos_scores, neg_scores):
        if self.normal_forms is not None and gci_name not in self.normal_forms:
            return NotImplemented
        pos = _flat(pos_scores)
        neg = self._negative_scores(gci_name, neg_scores)
        return self.combine(gci_name, pos, neg)

    def combine(self, gci_name, pos, neg):
        raise NotImplementedError

    def _paired(self, pos, neg):
        """Negatives as an ``(n, K)`` matrix aligned with the positives, or ``None``."""
        n = len(pos)
        if neg is None or n == 0 or len(neg) % n != 0:
            if neg is not None:
                self.num_unaligned += 1
            return None
        return neg.reshape(-1, n).t()


class SumLoss(_ScoreLoss):
    """Mean loss of the positives plus ``neg_weight`` times the mean negative loss.

    The neutral combination: with ``neg_weight=1`` it equals
    :meth:`~mowl.base_models.elmodel.EmbeddingELModel.compute_loss`, the loss of
    ELEmbeddings (Kulmanov et al. 2019, Section 3.4: "the sum of the loss functions"),
    EmEL++ (Section 3.3) and DELE (Mashkova et al., Eq. 23). BoxEL (Xiong et al.,
    Section 5.2) multiplies its non-subsumption loss by :math:`\\phi = 0.05`
    (``neg_weight=0.05``). Attach it to ELBE to replace ELBE's MSE loss by the plain sum.
    Negatives are scored by the module's negative loss. ``pos_weight`` weights the mean
    positive loss (ELK-Em weights its assertion losses, ``abox_pos_weight``).
    ``positive_floor=True`` takes the hinge :math:`\\max(0, s^+)` of every positive score
    first, for modules whose score of a stated axiom is a signed energy (disk embeddings,
    Eq. 10 and 11).
    """

    key = "loss-sum"
    needs_positive_form = False

    def __init__(self, neg_weight=1.0, normal_forms=None, pos_weight=1.0, positive_floor=False):
        super().__init__(normal_forms=normal_forms, neg_scores="module")
        self.neg_weight = float(neg_weight)
        self.pos_weight = float(pos_weight)
        self.positive_floor = bool(positive_floor)

    def combine(self, gci_name, pos, neg):
        if self.positive_floor:
            pos = th.relu(pos)
        loss = pos.mean() if self.pos_weight == 1.0 else self.pos_weight * pos.mean()
        if neg is not None:
            loss = loss + self.neg_weight * neg.mean()
        return loss


class MSETargetLoss(_ScoreLoss):
    """Squared distance of the scores to targets: :math:`\\mathrm{mean}((s^+ - t^+)^2) +
    \\mathrm{mean}((s^- - t^-)^2)`, negatives scored by the module's negative loss.

    The reference implementation of ELBE (Peng et al. 2022; ``ELBoxlModel.py``) applies
    ``MSELoss`` per axiom with target 0 to the positive losses and target 2 to the
    negative ones. mOWL's :class:`~mowl.models.ELBE` has a variant built in: it squares
    the *mean* score, ``MSE(mean(s+), 0) + MSE(mean(s-), 1)`` (``reduce="mean_first"``,
    ``neg_target=1`` reproduces it exactly). Box\\ :sup:`2`\\ EL squares per axiom inside
    its module (:math:`(\\delta - \\mu)^2` for negatives), so the neutral
    :class:`SumLoss` already gives its objective.

    :param pos_target: Target :math:`t^+` of the positive scores.
    :param neg_target: Target :math:`t^-` of the negative scores.
    :param reduce: ``"per_axiom"`` (mean of squared errors; ELBE's reference code) or
        ``"mean_first"`` (squared error of the mean; mOWL's ELBE).
    """

    key = "mse-to-target"
    needs_positive_form = False

    def __init__(self, pos_target=0.0, neg_target=2.0, reduce="per_axiom", normal_forms=None):
        super().__init__(normal_forms=normal_forms, neg_scores="module")
        if reduce not in ("per_axiom", "mean_first"):
            raise ValueError("reduce must be 'per_axiom' or 'mean_first'.")
        self.pos_target = float(pos_target)
        self.neg_target = float(neg_target)
        self.reduce = reduce

    def _term(self, x, target):
        if self.reduce == "mean_first":
            return (x.mean() - target) ** 2
        return ((x - target) ** 2).mean()

    def combine(self, gci_name, pos, neg):
        loss = self._term(pos, self.pos_target)
        if neg is not None:
            loss = loss + self._term(neg, self.neg_target)
        return loss


class PairwiseMarginRankingLoss(_ScoreLoss):
    """:math:`\\mathrm{mean}_{i,k} \\max(0, \\gamma + s^+_i - s^-_{ik})`: every positive
    must score at least :math:`\\gamma` better than each of its negatives.

    The "hierarchical contrastive" triplet loss of HiT (He et al. 2024, Eq. 3) and OnT
    (Yang et al. 2025, Section 3, hierarchy loss) with the hyperbolic distance as score,
    and the margin ranking loss of TransE-style models. Positives without negatives
    contribute their mean score.

    :param margin: :math:`\\gamma`.
    """

    key = "pairwise-margin-ranking"

    def __init__(self, margin=1.0, normal_forms=None, neg_scores="positive"):
        super().__init__(normal_forms=normal_forms, neg_scores=neg_scores)
        self.margin = float(margin)

    def combine(self, gci_name, pos, neg):
        if neg is None:
            return pos.mean()
        paired = self._paired(pos, neg)
        if paired is None:
            return F.relu(self.margin + pos - neg.mean()).mean()
        return F.relu(self.margin + pos.unsqueeze(1) - paired).mean()


class LogSigmoidLoss(_ScoreLoss):
    """Negative-sampling loss with a log-sigmoid, optionally self-adversarial:

    .. math::
        -\\log\\sigma(\\gamma - s^+) - \\sum_{k} \\alpha_k \\log\\sigma(s^-_k - \\gamma)

    with :math:`\\alpha = 1/K` (uniform; Query2Box, Ren et al. 2020, Eq. 4; ConE; AConE,
    He et al. 2024, Eq. 4) or, with ``adversarial_temperature`` :math:`T`, the
    self-adversarial weights :math:`\\alpha = \\mathrm{softmax}(-T s^-)` over the
    negatives of each positive, detached from the gradient (Sun et al. 2019; used by
    octagon embeddings, Charpenay and Schockaert 2024, Section 5, and binary-code boxes,
    Zhang et al. 2022, Section 4.4). Scores are distances (lower is better), so a
    negative with a *small* score (a hard negative) gets a large weight.

    :param margin: :math:`\\gamma`.
    :param adversarial_temperature: ``None`` for uniform weights.
    """

    key = "logsigmoid-negative-sampling"

    def __init__(self, margin=1.0, adversarial_temperature=None, normal_forms=None,
                 neg_scores="positive"):
        super().__init__(normal_forms=normal_forms, neg_scores=neg_scores)
        self.margin = float(margin)
        self.adversarial_temperature = (None if adversarial_temperature is None
                                        else float(adversarial_temperature))

    def combine(self, gci_name, pos, neg):
        loss = -F.logsigmoid(self.margin - pos).mean()
        if neg is None:
            return loss
        paired = self._paired(pos, neg)
        if paired is None:
            return loss - F.logsigmoid(neg - self.margin).mean()
        if self.adversarial_temperature is None:
            weights = th.full_like(paired, 1.0 / paired.shape[1])
        else:
            weights = th.softmax(-self.adversarial_temperature * paired.detach(), dim=1)
        return loss - (weights * F.logsigmoid(paired - self.margin)).sum(dim=1).mean()


class BPRLoss(_ScoreLoss):
    """Bayesian personalised ranking (Rendle et al. 2012): :math:`-\\mathrm{mean}_{i,k}
    \\log\\sigma(s^-_{ik} - s^+_i)`.

    FALCON (Tang et al. 2022, Section 5.1) uses the BPR loss with uniformly sampled
    negatives "for ABox optimization" (4, 8 or 16 per positive). Scores are distances,
    so the positive should have the smaller score.
    """

    key = "bpr-loss"

    def combine(self, gci_name, pos, neg):
        if neg is None:
            return pos.mean()
        paired = self._paired(pos, neg)
        if paired is None:
            return -F.logsigmoid(neg.mean() - pos).mean()
        return -F.logsigmoid(paired - pos.unsqueeze(1)).mean()


class LogSumExpContrastiveLoss(_ScoreLoss):
    """The contrastive loss RegD (Yang and Chen, Section 3.3, Eq. 9) takes from shadow
    cones (Yu et al. 2024):

    .. math::
        \\sum_{(u,v) \\in P} \\max\\{E(u,v), \\gamma_1\\}
        + \\log \\sum_{(u,v') \\in N} e^{\\max\\{\\gamma_2 - d(u, v'), 0\\}}

    :math:`E` is the score of a positive and :math:`d` the score of a negative. RegD uses
    its energy :math:`E` for positives but only the boundary dissimilarity :math:`d_{bd}`
    for negatives; here both are the module's scores (a model can supply different ones
    through its module). The sums run over the axioms of one normal form in one step;
    the first sum is averaged over the positives here so that the loss does not grow with
    the batch, which RegD does not specify.

    :param gamma1: :math:`\\gamma_1` (positive scores below it are not pushed further).
    :param gamma2: :math:`\\gamma_2` (how far negatives are pushed).
    """

    key = "logsumexp-contrastive"

    def __init__(self, gamma1=0.0, gamma2=1.0, normal_forms=None, neg_scores="positive"):
        super().__init__(normal_forms=normal_forms, neg_scores=neg_scores)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)

    def combine(self, gci_name, pos, neg):
        loss = th.clamp(pos, min=self.gamma1).mean()
        if neg is not None and len(neg):
            loss = loss + th.logsumexp(F.relu(self.gamma2 - neg), dim=0)
        return loss


class SoftmaxEnergyContrastiveLoss(_ScoreLoss):
    r"""The loss the RegD *code* trains with (``PluginRegD.loss_cross`` of
    HuiYang1997/RegD, inherited from the ShadowCones code): for every positive pair,
    the cross-entropy of :math:`e^{-E}` against its :math:`K` sampled negatives,

    .. math::
        -\log \frac{e^{-s^+}}{e^{-s^+} + \sum_{k} e^{-s^-_k}}
        \qquad
        s^+ = \max\{E^+, \gamma_1\}, \quad
        s^-_k = \min\{E^-_k - \gamma_2, 0\},

    averaged over the positive pairs. A positive is satisfied when its energy reaches
    :math:`\gamma_1` (with :math:`\gamma_1 = 0`: when the child region is contained) and
    a negative when its energy reaches :math:`\gamma_2` (with :math:`\gamma_2 = 0`: when
    it is *no longer* contained) -- the loss does not push negatives further out. Both
    terms use the full energy :math:`E = d_{bd} + \lambda d_{dep}`, unlike Eq. 10 of the
    paper, whose negatives use :math:`d_{bd}` only and whose combination is a max-margin
    rather than a softmax (:class:`LogSumExpContrastiveLoss`).

    The positive energies are the module's scores as the training loop passes them,
    clamped here at :math:`\gamma_1`; the negative energies are re-scored *raw*, which
    requires a module with the scoring context of :mod:`mowl.nn.el.regd`
    (:meth:`RegDELBEModule.scoring`).

    :param gamma1: The margin :math:`\gamma_1` at which a positive energy is clamped (0 in
        the RegD code).

    :param gamma2: The margin :math:`\gamma_2` subtracted from the negative energies
        before clamping (0 in the RegD code).
    :param rectify_negatives: Rectify the negative energies before the clamp (the
        released code's commented-out ``relu`` in ``PluginRegD.our_dist``); with
        :math:`\gamma_2 = 0` the negatives then exert no force.
    :param normal_forms: The axiom types trained with this loss (in the RegD code, only
        the normal form of the task: ``gci0`` for the inference task, ``gci2`` for the
        prediction task); the others fall back to the model's own loss.
    """

    key = "softmax-energy-contrastive"

    def __init__(self, gamma1=0.0, gamma2=0.0, normal_forms=None, rectify_negatives=False):
        super().__init__(normal_forms=normal_forms, neg_scores="positive")
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.rectify_negatives = bool(rectify_negatives)

    def _negative_scores(self, gci_name, neg_scores):
        if neg_scores is None:
            return None
        rows = self._last_negatives.get(gci_name)
        if rows is None or len(rows) != len(neg_scores):
            raise RuntimeError(
                f"{type(self).__name__} could not find the negatives of {gci_name!r} it "
                f"needs to re-score. Put it last in a ComposedRegime (it records the "
                f"negatives in filter_negatives).")
        module = self.model.module
        scoring = getattr(module, "scoring", None)
        if scoring is None:
            raise TypeError(
                f"{type(self).__name__} needs the raw energies of the negatives; the "
                f"module {type(module).__name__} has no scoring context. It is the loss "
                "of the RegD code and applies to RegD modules "
                "(mowl.nn.el.regd).")
        with scoring():
            energy = module(rows, gci_name)
        return energy.reshape(len(rows), -1).mean(dim=1)

    def combine(self, gci_name, pos, neg):
        # The module may pass raw energies (RegDCodeTraining trains in energy mode); the
        # code clamps the positive energy in the loss itself (idempotent otherwise).
        pos = th.clamp(pos, min=self.gamma1)
        if neg is None or not len(neg):
            return pos.mean()
        if self.rectify_negatives:
            # The variant with the line `our_dist = relu(our_dist + margin)` of
            # PluginRegD.our_dist active (commented out in the released code): negative
            # energies are rectified first, so with gamma2 = 0 they contribute a constant.
            neg = th.relu(neg)
        neg = th.clamp(neg - self.gamma2, max=0.0)
        paired = self._paired(pos, neg)
        if paired is None:
            # Unaligned negatives: one shared pool against every positive.
            logits = th.cat([-pos.reshape(-1, 1),
                             -neg.reshape(1, -1).expand(len(pos), -1)], dim=1)
        else:
            logits = th.cat([-pos.reshape(-1, 1), -paired], dim=1)
        target = th.zeros(len(logits), dtype=th.long, device=logits.device)
        return F.cross_entropy(logits, target)


class GroupedLogSumExpContrastiveLoss(_ScoreLoss):
    r"""Each positive against its own group of negatives: the mean over the positives of
    :math:`\max(E^+, \gamma) + \log\sum_k \exp(h_k)`, where :math:`h_k` is the module's
    negative loss of the :math:`k`-th corruption of that positive (a hinge, for example
    :math:`\max(0, \gamma_1 - E^-_k)`). It is Eq. 1 of the shadow cones paper (the softmax
    cross-entropy of a positive against ``K`` negatives, with :math:`\gamma = \gamma_2`)
    and Eq. 10 of RegD (with :math:`\gamma = \gamma_1`), and differs from
    :class:`LogSumExpContrastiveLoss`, which pools all negatives of the step.

    The negatives of a form must be grouped as the corrupting regimes produce them: ``K``
    blocks of one corruption per positive (a number of rows that is a multiple of the number
    of positives); otherwise the loss refuses.

    :param gamma: Clamp of the positive score from below.
    :param negative_form: ``"module"``: the module's negative loss is the hinge :math:`h_k`;
        ``"energy"``: the module returns raw energies :math:`E^-` and the hinge
        :math:`\max(0, \gamma_{neg} - E^-)` is taken here.
    :param gamma_neg: The margin of the hinge for ``negative_form="energy"``.
    :param normal_forms: The forms trained with this loss; the others fall back to the
        model's loss (RegD: ``gci0`` and ``gci2``).
    """

    key = "grouped-logsumexp-contrastive"
    needs_positive_form = False

    def __init__(self, gamma=0.0, negative_form="module", gamma_neg=0.0, normal_forms=None):
        super().__init__(normal_forms=normal_forms, neg_scores="module")
        if negative_form not in ("module", "energy"):
            raise ValueError("negative_form must be 'module' or 'energy'.")
        self.gamma = float(gamma)
        self.negative_form = negative_form
        self.gamma_neg = float(gamma_neg)

    def combine(self, gci_name, pos, neg):
        pos = th.clamp(pos, min=self.gamma)
        if neg is None:
            return pos.mean()
        paired = self._paired(pos, neg)
        if paired is None:
            raise ValueError(
                f"{type(self).__name__} needs a whole number of negatives per positive "
                f"({len(neg)} negatives for {len(pos)} positives of {gci_name!r}).")
        if self.negative_form == "energy":
            paired = F.relu(self.gamma_neg - paired)
        return (pos + th.logsumexp(paired, dim=1)).mean()


class PearsonCorrelationLoss(_ScoreLoss):
    """ELK-Em's contrastive loss (Singh et al. 2026, Section 4.2, Eq. 5):
    :math:`1 - \\frac{(\\sigma - \\bar\\sigma)^\\top (t - \\bar t)}
    {\\|\\sigma - \\bar\\sigma\\| \\|t - \\bar t\\| + \\varepsilon}` over the scores
    :math:`\\sigma` of the positives and their corruptions and the labels :math:`t`
    (1 for positives, 0 for corruptions).

    ELK-Em correlates the margin-free *axiom scores* :math:`s(\\alpha) = e^{-\\ell_0(\\alpha)}`
    (Eq. 4; higher is better) with the labels. ``score_transform="exp_neg"`` (default)
    applies :math:`e^{-\\ell}` to the module's loss scores; ``"neg"`` uses :math:`-\\ell`.
    The module's scores include its margin, whereas ELK-Em's use :math:`\\gamma = 0`
    (``margin_free=True`` below). ELK-Em uses this loss in addition to (not instead of) the
    positive per-axiom losses and only when training on the TBox alone:
    ``include_positive_loss=True`` (default) adds the mean positive loss.
    :class:`~mowl.models.elkem.model.ELKEm` has this loss built in.

    The authors' code differs from the paper's text in two points, both available here:
    it squares the per-axiom positive losses before averaging (``positive_power=2``), and
    it correlates the *margin-free* scores :math:`e^{-\\ell^0}` while the positive loss keeps
    the margin (``margin_free=True``: positives and corruptions are re-scored with the
    module's ``zero_margin_loss``).

    :param eps: :math:`\\varepsilon`.
    :param weight: Factor of the correlation term.
    :param positive_power: Power of the per-axiom positive losses before averaging (1: the
        paper's sum; 2: ``loss.pow(2).mean()`` of the ELK-Em code).
    :param margin_free: Correlate the margin-free scores (requires a module method
        ``zero_margin_loss(rows, gci_name)``) instead of the training-mode scores.
    """

    key = "pearson-correlation-loss"

    def __init__(self, eps=1e-8, weight=1.0, score_transform="exp_neg",
                 include_positive_loss=True, normal_forms=None, neg_scores="positive",
                 positive_power=1, margin_free=False):
        super().__init__(normal_forms=normal_forms, neg_scores=neg_scores)
        if score_transform not in ("exp_neg", "neg"):
            raise ValueError("score_transform must be 'exp_neg' or 'neg'.")
        if positive_power not in (1, 2):
            raise ValueError("positive_power must be 1 or 2.")
        self.eps = float(eps)
        self.weight = float(weight)
        self.score_transform = score_transform
        self.include_positive_loss = bool(include_positive_loss)
        self.positive_power = int(positive_power)
        self.margin_free = bool(margin_free)
        self._last_positives = {}

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        if self.margin_free:
            self._last_positives[gci_name] = gci_dataset[:]
        return super().filter_negatives(gci_name, gci_dataset, negatives)

    def _margin_free(self, rows, gci_name):
        module = self.model.module
        if not hasattr(module, "zero_margin_loss"):
            raise ValueError(f"margin_free=True needs {type(module).__name__}."
                             "zero_margin_loss(rows, gci_name).")
        return _flat(module.zero_margin_loss(rows, gci_name))

    def loss(self, gci_name, pos_scores, neg_scores):
        if self.normal_forms is not None and gci_name not in self.normal_forms:
            return NotImplemented
        if not self.margin_free or neg_scores is None:
            return super().loss(gci_name, pos_scores, neg_scores)
        pos_rows = self._last_positives.get(gci_name)
        neg_rows = self._last_negatives.get(gci_name)
        if pos_rows is None or neg_rows is None or len(pos_rows) != len(pos_scores):
            raise RuntimeError("PearsonCorrelationLoss(margin_free=True) could not find the "
                               f"rows of {gci_name!r}; put it last in a ComposedRegime.")
        loss = self._positive_term(_flat(pos_scores))
        if len(neg_rows) == 0:
            return loss
        corr = self.correlation(self._margin_free(pos_rows, gci_name),
                                self._margin_free(neg_rows, gci_name))
        return loss + self.weight * (1.0 - corr)

    def _positive_term(self, pos):
        if not self.include_positive_loss:
            return pos.sum() * 0.0
        return pos.pow(self.positive_power).mean()

    def correlation(self, pos, neg):
        """The Pearson correlation of the transformed scores with the labels."""
        scores = th.cat([pos, neg])
        sigma = th.exp(-scores) if self.score_transform == "exp_neg" else -scores
        t = th.cat([th.ones_like(pos), th.zeros_like(neg)])
        return pearson_correlation(sigma, t, self.eps)

    def combine(self, gci_name, pos, neg):
        loss = self._positive_term(pos)
        if neg is None or len(neg) == 0:
            return loss
        return loss + self.weight * (1.0 - self.correlation(pos, neg))


class ProbabilisticBCELoss(_ScoreLoss):
    """Binary cross-entropy with the score read as a negative log-probability,
    :math:`p(\\alpha) = e^{-s(\\alpha)}`:

    .. math::
        \\mathrm{mean}_i\\, s^+_i - \\mathrm{mean}_{i,k} \\log(1 - e^{-s^-_{ik}})

    Binary-code box embeddings (Zhang et al. 2022, Section 4.4, Eq. 10) train with this
    objective, :math:`E` being the negative log conditional probability of an edge; Gumbel
    boxes (Dasgupta et al. 2020, Section 5) minimise the cross-entropy of predicted
    conditional probabilities. Optional self-adversarial weights as in
    :class:`LogSigmoidLoss`.

    :param adversarial_temperature: ``None`` for uniform weights over the negatives.
    :param eps: Clamp of :math:`1 - p` inside the logarithm.
    """

    key = "probabilistic-bce"

    def __init__(self, adversarial_temperature=None, eps=1e-7, normal_forms=None,
                 neg_scores="positive"):
        super().__init__(normal_forms=normal_forms, neg_scores=neg_scores)
        self.adversarial_temperature = adversarial_temperature
        self.eps = float(eps)

    def combine(self, gci_name, pos, neg):
        loss = pos.mean()
        if neg is None:
            return loss
        log_not = th.log(th.clamp(-th.expm1(-neg), min=self.eps))
        paired = self._paired(pos, log_not)
        if paired is None or self.adversarial_temperature is None:
            return loss - log_not.mean()
        negs = self._paired(pos, neg)
        weights = th.softmax(-float(self.adversarial_temperature) * negs.detach(), dim=1)
        return loss - (weights * paired).sum(dim=1).mean()


class NormalFormFrequencyWeighting(TrainingRegime):
    """Weight the loss of every normal form by its share of the training data.

    Mashkova et al. (NeSy 2024, Section 4.4, Eq. 4) "weigh individual GCI losses based on
    frequency of the axiom types sampled during one epoch" to counter the imbalance
    between normal forms. The paper gives no formula; the reference implementation
    (``geometric_embeddings``, ``ELEmbeddings.train``) uses
    :math:`w_t = \\lceil n_t / b \\rceil / \\sum_u \\lceil n_u / b \\rceil`, the number of
    mini-batches of type :math:`t` over the number of all mini-batches, for batch size
    :math:`b`. ``batch_size=None`` (default) uses the model's ``batch_size``;
    ``batch_size=1`` gives :math:`w_t = n_t / \\sum_u n_u`.

    The regime wraps the loss of another regime (``base``) or, without one, the model's
    :meth:`~mowl.base_models.elmodel.EmbeddingELModel.compute_loss`. Weights are computed
    from the datasets actually trained on (after the other regimes' ``training_datasets``
    transformations, so place it after them in a composition).

    :param base: A loss regime whose loss is weighted, or ``None``.
    :param batch_size: See above.
    :param normal_forms: Axiom types counted and weighted (default: all trained on).
        Axiom types outside this list keep weight 1.
    """

    key = "normal-form-frequency-weighting"

    def __init__(self, base=None, batch_size=None, normal_forms=None):
        super().__init__()
        if base is not None and not isinstance(base, TrainingRegime):
            raise TypeError("base must be a TrainingRegime or None.")
        self.base = base
        self.batch_size = batch_size
        self.normal_forms = None if normal_forms is None else tuple(normal_forms)
        self.weights = {}

    def attach(self, model):
        super().attach(model)
        if self.base is not None:
            self.base.attach(model)

    def training_datasets(self, datasets):
        if self.base is not None:
            datasets = self.base.training_datasets(datasets)
        b = self.batch_size if self.batch_size is not None else self.model.batch_size
        b = max(1, int(b))
        steps = {k: math.ceil(len(v) / b) for k, v in datasets.items()
                 if (self.normal_forms is None or k in self.normal_forms) and len(v) > 0}
        total = sum(steps.values())
        self.weights = {k: s / total for k, s in steps.items()} if total else {}
        return datasets

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        if self.base is not None:
            return self.base.filter_negatives(gci_name, gci_dataset, negatives)
        return negatives

    def negatives(self, gci_name, gci_dataset):
        if self.base is not None:
            return self.base.negatives(gci_name, gci_dataset)
        return NotImplemented

    def on_epoch_start(self, epoch, epochs):
        if self.base is not None:
            self.base.on_epoch_start(epoch, epochs)

    def additional_loss(self, epoch):
        return None if self.base is None else self.base.additional_loss(epoch)

    def optimizer(self):
        return NotImplemented if self.base is None else self.base.optimizer()

    def loss(self, gci_name, pos_scores, neg_scores):
        out = NotImplemented
        if self.base is not None:
            out = self.base.loss(gci_name, pos_scores, neg_scores)
        if out is NotImplemented:
            out = self.model.compute_loss(pos_scores, neg_scores)
        return self.weights.get(gci_name, 1.0) * out
