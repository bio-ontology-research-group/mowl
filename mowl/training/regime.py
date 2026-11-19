"""Training regimes: how a model is trained, separately from what the model is.

A publication on ontology embeddings usually bundles three things: a *representation*
(balls, boxes, cones, ...), a *training regime* (which negatives are sampled, which
axioms count as positives, how losses are combined, how the optimiser is scheduled) and
an *evaluation regime*. mOWL keeps them apart. The representation is the model
(:mod:`mowl.models`), evaluation regimes live in :mod:`mowl.evaluation`, and a training
regime is a :class:`TrainingRegime` object that can be attached to any compatible model:

.. code-block:: python

   model = ELBE(dataset)
   model.training_regime = SomeRegime(...)
   model.train(epochs=100)

A regime customises the generic training loop through a small set of hooks. Every hook
has a neutral default (``NotImplemented`` or a no-op), meaning "use what the model does
by default", so a regime only implements what it changes, and regimes that change
different things can be combined with :class:`ComposedRegime`.

Regimes are described in the catalogue by a :class:`mowl.catalog.RegimeCard`, whether
they were published on their own (like deductive-closure filtering) or as part of a
method paper (like a particular negative sampling scheme).
"""


class TrainingRegime:
    """Base class of training regimes.

    Subclasses override the hooks they need. ``self.model`` is available after
    :meth:`attach`.
    """

    #: Key of the :class:`mowl.catalog.RegimeCard` describing this regime.
    key = ""

    def __init__(self):
        self.model = None

    def attach(self, model):
        """Bind the regime to ``model``. Called once, when the regime is assigned to the
        model. Raise :class:`TypeError` if the model is not compatible. Expensive
        preparation (for example computing a deductive closure) belongs here or should
        be done lazily on first use.
        """
        self.model = model

    def training_datasets(self, datasets):
        """Transform the training data.

        :param datasets: Mapping from axiom type name (``"gci0"``, ...) to dataset, as
            produced by the model.
        :return: The mapping to train on (for example with entailed axioms added as
            positives). The default returns ``datasets`` unchanged.
        """
        return datasets

    def negatives(self, gci_name, gci_dataset):
        """Return negative samples for one axiom type.

        :return: A tensor of corrupted rows, ``None`` for "no negatives for this axiom
            type", or ``NotImplemented`` to use the model's own negative sampling.
        """
        return NotImplemented

    def loss(self, gci_name, pos_scores, neg_scores):
        """Combine the scores of positives and negatives of one axiom type into a loss.

        :return: A scalar tensor, or ``NotImplemented`` to use the model's
            ``compute_loss``.
        """
        return NotImplemented

    def on_epoch_start(self, epoch, epochs):
        """Called at the beginning of every epoch (schedules, annealing, resampling)."""

    def additional_loss(self, epoch):
        """Extra loss term added once per epoch, or ``None``."""
        return None

    def optimizer(self):
        """Return the optimiser, or ``NotImplemented`` to use the model's."""
        return NotImplemented


class ComposedRegime(TrainingRegime):
    """Combination of several regimes.

    Dataset transformations are applied in order, per-epoch hooks are all called,
    additional losses are summed, and for :meth:`negatives`, :meth:`loss` and
    :meth:`optimizer` the first regime that does not return ``NotImplemented`` decides.

    :param regimes: The regimes to combine, in order of precedence.
    """

    def __init__(self, *regimes):
        super().__init__()
        for r in regimes:
            if not isinstance(r, TrainingRegime):
                raise TypeError("ComposedRegime takes TrainingRegime objects.")
        self.regimes = tuple(regimes)
        self.key = "+".join(r.key for r in regimes if r.key)

    def attach(self, model):
        super().attach(model)
        for r in self.regimes:
            r.attach(model)

    def training_datasets(self, datasets):
        for r in self.regimes:
            datasets = r.training_datasets(datasets)
        return datasets

    def _first(self, name, *args):
        for r in self.regimes:
            out = getattr(r, name)(*args)
            if out is not NotImplemented:
                return out
        return NotImplemented

    def negatives(self, gci_name, gci_dataset):
        return self._first("negatives", gci_name, gci_dataset)

    def loss(self, gci_name, pos_scores, neg_scores):
        return self._first("loss", gci_name, pos_scores, neg_scores)

    def optimizer(self):
        return self._first("optimizer")

    def on_epoch_start(self, epoch, epochs):
        for r in self.regimes:
            r.on_epoch_start(epoch, epochs)

    def additional_loss(self, epoch):
        total = None
        for r in self.regimes:
            extra = r.additional_loss(epoch)
            if extra is not None:
                total = extra if total is None else total + extra
        return total
