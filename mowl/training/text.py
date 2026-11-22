"""Training regimes of the text-encoder hierarchy methods (HiT, OnT), as parts.

HiT and OnT fine-tune a text encoder on verbalised ``(child, parent, negative)`` triplets with
AdamW, a linear warm-up followed by a linear decay of the learning rate, and the checkpoint of
the best validation epoch. The batching of the triplets is :class:`~mowl.training.EpochBatches`
(with the random generator of the model), the selection of the best epoch
:class:`~mowl.training.ValidationPatience`; what is new is

* :class:`LinearWarmupDecay`: the schedule of the transformers library
  (``get_linear_schedule_with_warmup``) that the reference codes use;
* :class:`SampledLogicalLosses`: the logical losses of OnT (Eq. 6 and 7), each on a random sample
  of the size of the batch that reuses the negatives of the batch.
"""

import torch as th

from mowl.training.regime import TrainingRegime
from mowl.training.schedules import _Schedule


class LinearWarmupDecay(_Schedule):
    """The learning rate rises linearly from 0 to its base value in ``warmup_steps`` steps and
    falls linearly to 0 at the last step (``get_linear_schedule_with_warmup`` of Hugging Face
    transformers, the default scheduler of the trainer of the HiT and OnT codes), stepped once
    per optimisation step. The number of steps is the one of the call of ``train``.

    :param warmup_steps: Steps of the warm-up (default: the model's ``warmup_steps``).
    """

    key = "linear-warmup-decay"

    def __init__(self, warmup_steps=None, base_optimizer=None):
        super().__init__(base_optimizer)
        self.warmup_steps = warmup_steps
        self.scheduler = None

    def setup(self, opt):
        warm = self.warmup_steps if self.warmup_steps is not None \
            else self.model.warmup_steps
        warm = max(0, int(warm))
        self.total = getattr(self.model, "training_steps", None)

        def schedule(step):
            if step < warm:
                return step / max(1, warm)
            if self.total is None:
                return 1.0
            return max(0.0, (self.total - step) / max(1, self.total - warm))
        self.scheduler = th.optim.lr_scheduler.LambdaLR(opt, schedule)

    def update(self, epoch, epochs):
        self.total = epochs
        if epoch > 0:
            self.scheduler.step()


class SampledLogicalLosses(TrainingRegime):
    """The logical losses of OnT (Yang et al. 2025, Eq. 6 and 7), added to every step. On a
    random sample of the size of the batch of the triplets (all of them if there are fewer)
    drawn from the existential restrictions :math:`\\exists r.D` of the ontology the loss
    :math:`L_r` is taken, on a sample from its conjunctions the loss :math:`L_\\sqcap`; the
    negatives of both are those of the batch (a random permutation of them for
    :math:`L_r`, a random pick for every conjunct for :math:`L_\\sqcap`). The step takes
    ``existential_weight`` times the mean :math:`L_r` and ``conjunction_weight`` times
    :math:`L_\\sqcap` (the mean over the conjunctions of the mean over their conjuncts).

    The model gives ``existentials`` (a list of ``(text of \\exists r.D, text of r, text of D)``),
    ``conjunctions`` (a list of ``(text, [conjunct texts])``), ``rng`` (with ``sample``),
    ``texts(ids)`` and a module with ``classes``, ``compound``, ``existential_loss`` and
    ``hierarchy_loss`` (:class:`~mowl.representation.library.hierarchy_geometry.TextGeometry`).

    :param existential_weight, conjunction_weight: The weights of the terms.
    :param form: The form of the triplets of the batch.
    """

    key = "sampled-logical-losses"

    def __init__(self, existential_weight=1.0, conjunction_weight=1.0, form="triplet"):
        super().__init__()
        self.existential_weight = existential_weight
        self.conjunction_weight = conjunction_weight
        self.form = form
        self._rows = None
        self.terms = {}

    def filter_negatives(self, form, dataset, negatives):
        if form == self.form:
            self._rows = dataset[:]
        return negatives

    def _sample(self, items, n):
        if len(items) <= n:
            return list(items)
        return self.model.rng.sample(items, n)

    def additional_loss(self, epoch):
        rows, self._rows = self._rows, None
        self.terms = {}
        model = self.model
        if rows is None or not (model.existentials or model.conjunctions):
            return None
        g = model.module
        manifold = g.class_model.manifold
        from mowl.representation.library.hierarchy_regions import ManifoldPoint
        n = len(rows)
        negatives = g.classes(model.texts(rows[:, 2])).centre
        loss = None
        if model.existentials:
            ex = self._sample(model.existentials, n)
            k = len(ex)
            neg1 = negatives[th.randperm(n, device=negatives.device)[:k]]
            neg2 = negatives[th.randperm(n, device=negatives.device)[:k]]
            l_r = g.existential_loss(g.compound([e[0] for e in ex]), [e[1] for e in ex],
                                     g.classes([e[2] for e in ex]), ManifoldPoint(neg1, manifold),
                                     ManifoldPoint(neg2, manifold)).mean()
            self.terms["existential"] = float(l_r.detach())
            loss = self.existential_weight * l_r
        if model.conjunctions:
            cj = self._sample(model.conjunctions, n)
            conj_texts, part_texts, weights = [], [], []
            for text, ops in cj:
                for o in ops:
                    conj_texts.append(text)
                    part_texts.append(o)
                    weights.append(1.0 / len(ops))
            idx = th.randint(0, n, (len(conj_texts),), device=negatives.device)
            per_row = g.hierarchy_loss(g.compound(conj_texts), g.classes(part_texts),
                                       ManifoldPoint(negatives[idx], manifold)).squeeze(-1)
            w = th.tensor(weights, device=per_row.device, dtype=per_row.dtype)
            l_c = (w * per_row).sum() / len(cj)
            self.terms["conjunction"] = float(l_c.detach())
            term = self.conjunction_weight * l_c
            loss = term if loss is None else loss + term
        return loss
