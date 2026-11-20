"""Model selection (checkpoint selection) on validation data, as a training regime.

Papers select the parameters they report in two different ways:

* by the **validation loss** of the predicted normal form -- mOWL's built-in training loop
  (it saves the best ``state_dict`` to ``model_filepath``), DELE (Section 6.3) and the NeSy
  2024 paper (Section 4.4);
* by a **validation rank metric** -- Box\\ :sup:`2`\\ EL ("We evaluate the models on a
  fraction of the validation set every 100 epochs and choose the embeddings that achieve
  the best performance", Section 4.2; the reference ``train.py`` uses the median rank of
  the first 1,000 validation ``nf1`` axioms) and ELK-Em ("Checkpoints are selected by
  median rank on the validation split", Section 5.1).

The validation loss and a rank metric can select very different checkpoints: the loss of
the positive validation axioms alone keeps decreasing when all regions grow, which a rank
metric penalises.

:class:`ValidationSelection` evaluates the parameters reached after epoch ``e - 1`` in the
``on_epoch_start`` hook of epoch ``e`` and keeps the best ``state_dict`` in memory. After
the last epoch (``on_train_end``) it evaluates the final parameters and loads the best
ones into the module; :meth:`ValidationSelection.finalize` does the same by hand.
"""

import copy

import numpy as np
import torch as th

from mowl.training.regime import TrainingRegime
from mowl.training.schedules import validation_loss


def validation_rank(model, gci_name="gci0", target_col=None, statistic="median",
                    max_queries=1000, candidates=None, scoring="auto", ties="ordinal",
                    split="valid"):
    """A rank statistic of the validation axioms of one normal form (lower is better).

    For every validation axiom (the first ``max_queries``, as in the reference code of
    Box\\ :sup:`2`\\ EL) the entity in ``target_col`` is replaced by every candidate and
    the rank of the true axiom is computed with
    :func:`mowl.evaluation.axiom_ranking.rank_queries`.

    :param gci_name: Normal form of the validation axioms.
    :param target_col: Ranked column (default: the last column).
    :param statistic: ``"median"``, ``"mean"``, ``"mrr"`` (returned negated, so that lower
        is better) or ``"hits@k"`` (negated).
    :param candidates: Candidate class indices (default: the model's named classes).
    :param scoring: See :func:`mowl.evaluation.el_scoring.resolve_scorer`.
    :param split: ``"valid"`` (the model's validation axioms) or ``"train"`` (its training
        axioms; useful for ontologies without a validation part).
    """
    from mowl.evaluation.axiom_ranking import rank_queries
    from mowl.evaluation.el_scoring import resolve_scorer
    if split == "valid":
        rows = model.validation_datasets[gci_name][:]
    elif split == "train":
        rows = model.training_datasets[gci_name][:]
    else:
        raise ValueError("split must be 'valid' or 'train'.")
    if max_queries is not None:
        rows = rows[:int(max_queries)]
    if target_col is None:
        target_col = rows.shape[1] - 1
    if candidates is None:
        candidates = th.tensor(sorted(model.named_class_index_dict.values()), dtype=th.long)
    scorer, _ = resolve_scorer(model, scoring)
    ranks, _, _ = rank_queries(model, gci_name, rows.cpu(), target_col, candidates, scorer,
                               ties=ties)
    r = ranks.double().numpy()
    if len(r) == 0:
        return float("inf")
    if statistic == "median":
        return float(np.median(r))
    if statistic == "mean":
        return float(np.mean(r))
    if statistic == "mrr":
        return -float(np.mean(1.0 / r))
    if statistic.startswith("hits@"):
        return -float(np.mean(r <= int(statistic[5:])))
    raise ValueError(f"Unknown statistic {statistic!r}.")


class ValidationSelection(TrainingRegime):
    """Keep the parameters with the best validation quantity.

    :param metric: ``"loss"`` (:func:`~mowl.training.schedules.validation_loss`),
        ``"rank"`` (:func:`validation_rank` with ``rank_kwargs``) or a callable
        ``f(model) -> float``. Lower is better.
    :param every: Evaluate every ``every`` epochs (Box\\ :sup:`2`\\ EL: 100).
    :param rank_kwargs: Keyword arguments of :func:`validation_rank`.
    """

    key = "validation-loss-selection"

    def __init__(self, metric="loss", every=1, rank_kwargs=None):
        super().__init__()
        if metric == "loss":
            self.metric = validation_loss
        elif metric == "rank":
            kwargs = dict(rank_kwargs or {})
            self.metric = lambda model: validation_rank(model, **kwargs)
            self.key = "validation-rank-selection"
        elif callable(metric):
            self.metric = metric
        else:
            raise ValueError("metric must be 'loss', 'rank' or a callable.")
        self.every = max(1, int(every))
        self.best_value = float("inf")
        self.best_epoch = None
        self.best_state = None
        self.history = []

    def attach(self, model):
        super().attach(model)
        self.best_value, self.best_epoch, self.best_state = float("inf"), None, None
        self.history = []
        self._epochs_done = 0

    def _evaluate(self, trained_epochs):
        value = float(self.metric(self.model))
        self.history.append((trained_epochs, value))
        if value <= self.best_value:
            self.best_value = value
            self.best_epoch = trained_epochs
            self.best_state = copy.deepcopy(self.model.module.state_dict())

    def on_epoch_start(self, epoch, epochs):
        # Parameters now are those after `epoch` optimisation steps.
        self._epochs_done = epoch + 1
        if epoch > 0 and epoch % self.every == 0:
            self._evaluate(epoch)

    def finalize(self, restore=True):
        """Evaluate the final parameters, then (``restore=True``) load the best ones.

        :return: The number of training epochs after which the best parameters were reached.
        """
        if self.model is None:
            raise RuntimeError("The regime is not attached to a model.")
        done = getattr(self, "_epochs_done", 0)
        if not self.history or self.history[-1][0] != done:
            self._evaluate(done)
        if restore and self.best_state is not None:
            self.model.module.load_state_dict(self.best_state)
        return self.best_epoch

    def on_train_end(self):
        self.finalize(restore=True)


class ValidationRankSelection(ValidationSelection):
    """:class:`ValidationSelection` with ``metric="rank"``: the median rank of the first
    1,000 validation ``gci0`` axioms by default, as in Box\\ :sup:`2`\\ EL's reference code
    and ELK-Em.

    :param every: Evaluation interval in epochs.
    :param rank_kwargs: Keyword arguments of :func:`validation_rank`.
    """

    key = "validation-rank-selection"

    def __init__(self, every=1, **rank_kwargs):
        super().__init__(metric="rank", every=every, rank_kwargs=rank_kwargs)


def validation_f1(model, gci_name="gci0", num_positives=1000, num_negatives=10,
                  positive_scores="reference", reference_ontology=None, seed=0):
    """Best achievable F1 on the first validation axioms of one normal form, negated
    (so that lower is better): the model-selection quantity of the RegD code
    (``train`` of ``EXP_ontology/baseline_sweep.py`` computes
    ``compute_ranks(val_data[nf][:1000], threshold=None)`` every 100 epochs and keeps
    the checkpoint with the best F1).

    The F1 is computed exactly as at test time
    (:func:`mowl.evaluation.sampled_f1.validation_best_f1`): the threshold is tuned on
    the validation pairs themselves, and with ``positive_scores="reference"`` the
    positive score of a ``gci2`` row is the score the code reads (the class indexed by
    the role in the reference data, see
    :func:`mowl.evaluation.sampled_f1.reference_role_classes`).

    :param positive_scores: ``"reference"`` (the code) or ``"pair"`` (the axiom's own
        score; identical for ``gci0``).
    :param reference_ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``; needed for
        ``gci2`` with ``"reference"``.
    :param seed: Seed of the negative sampling.
    """
    from mowl.evaluation.sampled_f1 import validation_best_f1
    f1, _ = validation_best_f1(model, gci_name=gci_name, num_positives=num_positives,
                               num_negatives=num_negatives,
                               positive_scores=positive_scores,
                               reference_ontology=reference_ontology, seed=seed)
    return -f1


class ValidationF1Selection(ValidationSelection):
    """:class:`ValidationSelection` with :func:`validation_f1`: keep the parameters with
    the best F1 on the first 1,000 validation axioms of the task's normal form, the
    threshold tuned on that split, as the RegD code selects its checkpoint every 100
    epochs.

    Ties are resolved towards the *later* epoch (``<=``), as in the code (``>=``).

    :param every: Evaluation interval in epochs.
    :param gci_name: Normal form of the validation axioms (the task's: ``gci0`` for
        RegD's inference task, ``gci2`` for the prediction task).
    :param num_positives: Validation axioms to score (the first ones, as in the code).
    :param num_negatives: Random candidates per validation axiom.
    :param positive_scores: See :func:`validation_f1`.
    :param reference_ontology: See :func:`validation_f1`.
    :param seed: Seed of the negative sampling.
    """

    key = "validation-f1-selection"

    def __init__(self, every=1, gci_name="gci0", num_positives=1000, num_negatives=10,
                 positive_scores="reference", reference_ontology=None, seed=0):
        super().__init__(
            metric=lambda model: validation_f1(
                model, gci_name=gci_name, num_positives=num_positives,
                num_negatives=num_negatives, positive_scores=positive_scores,
                reference_ontology=reference_ontology, seed=seed),
            every=every)
        self.key = type(self).key
