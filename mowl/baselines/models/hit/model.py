"""The direct implementation of HiT: its own loss, AdamW loop and validation in the model class.
Kept for the comparison in ``tests/baselines``; the taxonomy, the verbalisation, the training
triplets and the scoring are shared (:mod:`mowl.baselines.models.hit.base`)."""

import logging
import math

import torch as th

from mowl.baselines.models.hit.base import DEFAULT_CENTRI_GRID, HiTBase  # noqa: F401

logger = logging.getLogger(__name__)


class HiT(HiTBase):
    """HiT with the loss and the loop written out (see :class:`mowl.models.HiT` for the
    parameters)."""

    # ------------------------------------------------------------------ set-up
    def init_module(self):
        from mowl.baselines.nn.lm.hit import HiTModule
        return HiTModule(self.encoder, curvature=self.curvature,
                         clustering_margin=self.clustering_margin,
                         centripetal_margin=self.centripetal_margin,
                         clustering_weight=self.clustering_weight,
                         centripetal_weight=self.centripetal_weight)

    # -- what depends on the module of this side ----------------------------------------
    def _embed_points(self, texts):
        return self.module.embed(texts)

    def _cost(self, sub, sup, lam):
        from mowl.baselines.nn.lm import hyperbolic as hb
        c = self.c
        return hb.distance(sub, sup, c) + lam * (hb.hyperbolic_norm(sup, c)
                                                 - hb.hyperbolic_norm(sub, c))

    def _ranking_scores(self, query, candidates, candidates_are_children, lam):
        from mowl.baselines.nn.lm import hyperbolic as hb
        c, dtype = self.c, self.score_dtype
        q_norm = hb.hyperbolic_norm(query.to(dtype), c).unsqueeze(1)
        dist = hb.pairwise_distance(query, candidates, c, dtype=dtype)
        c_norm = hb.hyperbolic_norm(candidates.to(dtype), c).unsqueeze(0)
        if candidates_are_children:
            return -(dist + lam * (q_norm - c_norm))
        return -(dist + lam * (c_norm - q_norm))

    def _score_pair(self, x, lam):
        return float(self.module.score(x[0], x[1], lam))

    # ------------------------------------------------------------------ training
    def _optimiser(self, total_steps):
        params = [p for p in self.module.parameters() if p.requires_grad]
        opt = th.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        warm = max(0, int(self.warmup_steps))

        def schedule(step):
            # get_linear_schedule_with_warmup of Hugging Face transformers (the default
            # scheduler of the trainer used by the reference implementations)
            if step < warm:
                return step / max(1, warm)
            return max(0.0, (total_steps - step) / max(1, total_steps - warm))
        return opt, th.optim.lr_scheduler.LambdaLR(opt, schedule)

    def batch_loss(self, batch, negatives):
        """Mean training loss of one batch of triplets (HiT: :math:`L_{HiT}`, Eq. 5).

        :param batch: List of ``(child, parent, negative)`` texts.
        :param negatives: Embeddings of the negatives of the batch.
        :return: ``(loss, parts)`` with ``parts`` a dict of floats for logging.
        """
        child = self.module.embed([t[0] for t in batch])
        parent = self.module.embed([t[1] for t in batch])
        loss = self.module.hierarchy_loss(child, parent, negatives).mean()
        return loss, {"hierarchy": float(loss.detach())}

    def train(self, epochs=1, validate=True, verbose=False):
        """Train the encoder.

        :param epochs: Number of passes over the training triplets (20 in HiT, 1 in OnT).
        :param validate: Select :math:`\\lambda` and the best epoch on validation data
            (if ``dataset.validation`` has atomic subsumptions).
        :param verbose: Log the loss every 100 steps.
        :return: The mean training loss of the last epoch.
        """
        triplets = self.training_triplets()
        steps_per_epoch = math.ceil(len(triplets) / self.batch_size)
        opt, sched = self._optimiser(steps_per_epoch * epochs)
        has_valid = validate and self._validation_rows() is not None
        best = None
        mean_loss = float("nan")
        for epoch in range(epochs):
            self.module.train()
            self._invalidate()
            order = list(range(len(triplets)))
            self.rng.shuffle(order)
            total = 0.0
            for step, start in enumerate(range(0, len(order), self.batch_size)):
                batch = [triplets[i] for i in order[start:start + self.batch_size]]
                negatives = self.module.embed([t[2] for t in batch])
                loss, parts = self.batch_loss(batch, negatives)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach())
                if verbose and step % 100 == 0:
                    logger.info("epoch %d step %d %s", epoch, step, parts)
            self._invalidate()
            mean_loss = total / max(1, steps_per_epoch)
            record = {"epoch": epoch, "loss": mean_loss}
            if has_valid:
                mrr, lam = self.validate()
                record.update({"valid_mrr": mrr, "centri_weight": lam})
                if best is None or mrr > best[0]:
                    best = (mrr, lam, epoch)
                    self._save_state()
            self.history.append(record)
            if verbose:
                logger.info("%s", record)
        if best is not None:
            self.load_best_model()
            if not self._fixed_centri_weight:
                self.centri_weight = best[1]
            self.best_validation = {"mrr": best[0], "centri_weight": best[1],
                                    "epoch": best[2]}
        elif self.centri_weight is None:
            self.centri_weight = 0.0
        return mean_loss

