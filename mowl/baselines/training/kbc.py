"""The training design of the reference code of the faithfulness benchmark (Mashkova &
Hoehndorf 2026, ``bio-ontology-research-group/kbc``, ``run_el.py`` with ``box2el.py``,
``elembeddings.py``, ``emelpp.py`` and ``dataloader.py``), as named regimes.

The three EL models of that code base share one training loop, which differs from mOWL's
defaults in three points:

* **pooled mean** (``weight_mode="uniform"``, the default of ``run_el.py``): the per-axiom
  losses of all positives of all normal forms and of all negatives are summed and divided
  by their total number (:class:`PooledAxiomMeanLoss`), instead of summing per-form means;
* **fixed negatives**: one corruption of the superclass of every ``gci2`` axiom
  (``neg_types=["gci2"]``), drawn from the classes; the data loader calls
  ``np.random.seed(0)`` at every batch, so with one batch per epoch every epoch sees the
  same negatives (:class:`DrawOnce`);
* **plateau schedule and early stopping** on the validation loss of the normal form under
  test: ``ReduceLROnPlateau(patience=10)``, stop after 20 epochs without improvement,
  keep the parameters of the best validation loss.
"""

from mowl.baselines.training.regime import ComposedRegime, TrainingRegime
from mowl.baselines.training.losses import _ScoreLoss
from mowl.baselines.training.negatives import FixedColumnCorruption
from mowl.baselines.training.schedules import ReduceOnPlateau


class DrawOnce(TrainingRegime):
    """Draw the negatives of a sampler regime once per normal form and reuse them.

    The data loader of the faithfulness reference code reseeds NumPy with 0 before every
    batch, so a full-batch run trains on the same corruptions in every epoch. The cache is
    keyed by normal form and only reused while the positives have the same length (with
    mini-batches the negatives are drawn anew for every batch of a new size).

    :param base: The sampler regime whose first draw is kept.
    """

    key = "draw-negatives-once"

    def __init__(self, base):
        super().__init__()
        if not isinstance(base, TrainingRegime):
            raise TypeError("base must be a TrainingRegime.")
        self.base = base
        self._cache = {}

    def attach(self, model):
        super().attach(model)
        self.base.attach(model)
        self._cache = {}

    def negatives(self, gci_name, gci_dataset):
        n = len(gci_dataset)
        cached = self._cache.get(gci_name)
        if cached is not None and cached[0] == n:
            return cached[1]
        out = self.base.negatives(gci_name, gci_dataset)
        if out is not NotImplemented and out is not None:
            self._cache[gci_name] = (n, out)
        return out


class PooledAxiomMeanLoss(_ScoreLoss):
    """Sum of the per-axiom losses of all positives and negatives of all normal forms,
    divided by their total number (``weight_mode="uniform"`` of the faithfulness reference
    code; DELE's code has the same option).

    Normal forms with many axioms weigh more than with :class:`~mowl.training.SumLoss`
    (a sum of per-form means). The denominator is computed from the datasets trained on
    (:meth:`training_datasets`) and the negatives seen in the first epoch, so the regime
    assumes a fixed number of rows per epoch (full batch, as in the reference runs).
    Negatives are scored with the module's negative loss.
    """

    key = "pooled-axiom-mean"
    needs_positive_form = False

    def __init__(self, normal_forms=None):
        super().__init__(normal_forms=normal_forms, neg_scores="module")
        self.num_positives = 0
        self._num_negatives = {}

    def training_datasets(self, datasets):
        self.num_positives = sum(len(v) for k, v in datasets.items()
                                 if self.normal_forms is None or k in self.normal_forms)
        self._num_negatives = {}
        return datasets

    @property
    def total(self):
        return max(1, self.num_positives + sum(self._num_negatives.values()))

    def combine(self, gci_name, pos, neg):
        loss = pos.sum()
        if neg is not None and len(neg):
            self._num_negatives.setdefault(gci_name, len(neg))
            loss = loss + neg.sum()
        return loss / self.total


class KBCCodeTraining(ComposedRegime):
    """How ``run_el.py`` of the faithfulness reference code trains ELEmbeddings, EmEL++
    and Box\\ :sup:`2`\\ EL (Mashkova & Hoehndorf 2026), with its default arguments
    (``neg_types=["gci2"]``, ``weight_mode="uniform"``, ``val_mode="new"``,
    ``random_neg_fraction=1``):

    * one corruption per ``gci2`` axiom, superclass replaced by a uniformly drawn class,
      drawn once and reused (:class:`DrawOnce` around
      :class:`~mowl.training.FixedColumnCorruption`); no negatives for the other forms;
    * :class:`PooledAxiomMeanLoss` over all positives and negatives, plus the model's
      regulariser;
    * Adam with :class:`~mowl.training.ReduceOnPlateau` (patience 10) on the validation
      loss of the normal form under test, early stopping after ``stop_patience`` epochs
      without improvement; the benchmark runner keeps the parameters with the best
      validation loss (the model's own checkpointing).

    Batching is the model's full batch: the reference batch size (32,768 per normal form)
    exceeds every split of the benchmark, so one step is one epoch.

    :param patience: Plateau patience (10).
    :param stop_patience: Epochs without improvement before stopping (20).
    :param seed: Seed of the one draw of negatives (the code always uses 0).
    """

    key = "kbc-code-training"

    def __init__(self, patience=10, stop_patience=20, seed=0):
        super().__init__()
        self.patience = int(patience)
        self.stop_patience = int(stop_patience)
        self.seed = seed
        self.key = type(self).key

    def attach(self, model):
        sampler = FixedColumnCorruption(columns={"gci2": (2,)}, num_negatives=1,
                                        require_negative_loss=False, seed=self.seed)
        self.regimes = (
            DrawOnce(sampler),
            _NoOtherNegatives(("gci2",)),
            PooledAxiomMeanLoss(),
            ReduceOnPlateau(patience=self.patience, stop_patience=self.stop_patience,
                            early_stop=True),
        )
        super().attach(model)


class _NoOtherNegatives(TrainingRegime):
    """No negatives for the normal forms outside ``keep`` (the model's own sampler is
    bypassed)."""

    def __init__(self, keep):
        super().__init__()
        self.keep = tuple(keep)

    def negatives(self, gci_name, gci_dataset):
        if gci_name in self.keep:
            return NotImplemented
        return None
