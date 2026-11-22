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


#: The hooks of a :class:`TrainingRegime` a training loop can consult. A model lists the ones
#: its loop consults in ``training_hooks``; a regime that overrides a hook the model does not
#: consult is refused when it is assigned, instead of being ignored.
HOOKS = ("training_datasets", "negatives", "filter_negatives", "loss", "optimizer",
         "on_epoch_start", "on_epoch_end", "should_stop", "on_train_end", "additional_loss",
         "regularization", "after_step")

#: Hooks that concern the optimiser and the course of training only, not the data, the
#: negatives or the loss. A loop with an objective of its own can still consult these.
SCHEDULE_HOOKS = frozenset({"optimizer", "on_epoch_start", "on_epoch_end", "should_stop",
                            "on_train_end", "additional_loss"})


def unsupported_hooks(regime, model):
    """The hooks ``regime`` overrides that the training loop of ``model`` (an instance or a
    class) does not consult; empty if the two can be combined."""
    return frozenset(regime.overridden_hooks()) - frozenset(getattr(model, "training_hooks", ()))


def check_compatible(regime, model):
    """Raise :class:`TypeError` if ``regime`` overrides hooks the training loop of ``model``
    does not consult."""
    missing = unsupported_hooks(regime, model)
    if missing:
        name = type(model).__name__
        supported = sorted(getattr(model, "training_hooks", ()))
        raise TypeError(
            f"{type(regime).__name__} cannot be used with {name}: it changes "
            f"{sorted(missing)}, which the training loop of {name} does not consult "
            f"(it consults {supported if supported else 'no regime hooks'}). "
            f"The regime would be ignored.")


class TrainingRegime:
    """Base class of training regimes.

    Subclasses override the hooks they need. ``self.model`` is available after
    :meth:`attach`.
    """

    #: Key of the :class:`mowl.catalog.RegimeCard` describing this regime.
    key = ""

    #: ``True`` for a regime that does the model selection itself: the generic loop then skips
    #: its own validation (by the loss of ``eval_gci_name``, which saves a checkpoint).
    replaces_validation = False

    def __init__(self):
        self.model = None

    def overridden_hooks(self):
        """The :data:`HOOKS` this regime overrides (what it changes in a training loop)."""
        return frozenset(h for h in HOOKS
                         if getattr(type(self), h, None) is not getattr(TrainingRegime, h))

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

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        """Post-process negative samples, wherever they came from (this regime, another
        regime of a composition, or the model's own sampler). Typical use: replace or
        drop sampled negatives that are known or entailed to be true.

        :param negatives: Tensor of corrupted rows, or ``None``.
        :return: The negatives to use. Must keep the number of rows per positive that the
            model's loss expects (resample rather than drop if the model requires it).
            The default returns ``negatives`` unchanged.
        """
        return negatives

    def loss(self, gci_name, pos_scores, neg_scores):
        """Combine the scores of positives and negatives of one axiom type into a loss.

        :return: A scalar tensor, or ``NotImplemented`` to use the model's
            ``compute_loss``.
        """
        return NotImplemented

    #: How the negatives handed to :meth:`loss` are scored. ``False`` (default): with the
    #: module's negative-form loss (``module(neg, gci_name, neg=True)``), as the model
    #: itself does. ``True``: with the same score as positives
    #: (``module(neg, gci_name)``), which is what score-contrasting losses (pairwise
    #: ranking, log-sigmoid, BPR, correlation) need.
    score_negatives_like_positives = False

    def scores_negatives_like_positives(self, gci_name):
        """Whether the negatives of ``gci_name`` are scored with the positive-form loss
        (see :attr:`score_negatives_like_positives`) rather than the module's negative
        loss. ``None`` means "this regime does not decide the loss of ``gci_name``" (in a
        :class:`ComposedRegime` the next regime then does). The training loop asks this for
        every axiom type, so that a negative loss is requested from the module only where
        the loss in use needs one: a module has none for most axiom types, and refuses to
        compute it (:class:`~mowl.nn.ELModule`)."""
        return self.score_negatives_like_positives

    def should_stop(self, epoch):
        """Called at the end of every epoch. Return ``True`` to end training (early
        stopping)."""
        return False

    def on_train_end(self):
        """Called once after the last epoch (for example to restore the selected
        checkpoint)."""

    def on_epoch_start(self, epoch, epochs):
        """Called at the beginning of every epoch (schedules, annealing, resampling)."""

    def additional_loss(self, epoch):
        """Extra loss term added once per epoch, or ``None``."""
        return None

    def regularization(self, default):
        """The regularisation term of one optimisation step.

        :param default: The term the model would use (its module's
            ``regularization_loss``; ``0`` if it has none).
        :return: The term to use instead, or ``NotImplemented`` to keep ``default``. See
            :mod:`mowl.training.regularization`.
        """
        return NotImplemented

    def after_step(self):
        """Called after every optimisation step, before :meth:`on_epoch_end` (projections
        of the parameters back into an allowed set)."""

    def on_epoch_end(self, epoch, losses):
        """Called at the end of every epoch, after the optimisation step and before
        validation (structural changes of the model such as re-seeding components,
        updates of loss weights).

        :param losses: Mapping from loss term name to its value in this epoch as a
            float. Always has ``"total"``; models that report their loss per term (for
            example :class:`mowl.models.GroundedEL`: ``"fact"``, ``"axioms"``,
            ``"region"``) add those.
        """

    def optimizer(self):
        """Return the optimiser, or ``NotImplemented`` to use the model's."""
        return NotImplemented


class ComposedRegime(TrainingRegime):
    """Combination of several regimes.

    Dataset transformations are applied in order, per-epoch hooks (start and end) are
    all called in order,
    negative filters are chained in order,
    additional losses are summed, training stops as soon
    as one regime asks for it, and for :meth:`negatives`, :meth:`loss` and
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

    def overridden_hooks(self):
        return frozenset().union(*(r.overridden_hooks() for r in self.regimes))

    @property
    def replaces_validation(self):
        return any(r.replaces_validation for r in self.regimes)

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

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        for r in self.regimes:
            negatives = r.filter_negatives(gci_name, gci_dataset, negatives)
        return negatives

    def loss(self, gci_name, pos_scores, neg_scores):
        return self._first("loss", gci_name, pos_scores, neg_scores)

    def optimizer(self):
        return self._first("optimizer")

    @property
    def score_negatives_like_positives(self):
        """Taken from the regime that decides :meth:`loss` (the first one overriding
        it)."""
        for r in self.regimes:
            if type(r).loss is not TrainingRegime.loss:
                return r.score_negatives_like_positives
        return False

    def scores_negatives_like_positives(self, gci_name):
        for r in self.regimes:
            if type(r).loss is not TrainingRegime.loss:
                decided = r.scores_negatives_like_positives(gci_name)
                if decided is not None:
                    return decided
        return False

    def should_stop(self, epoch):
        # Every regime is asked, so each sees every epoch.
        return any([r.should_stop(epoch) for r in self.regimes])

    def on_train_end(self):
        for r in self.regimes:
            r.on_train_end()

    def on_epoch_start(self, epoch, epochs):
        for r in self.regimes:
            r.on_epoch_start(epoch, epochs)

    def on_epoch_end(self, epoch, losses):
        for r in self.regimes:
            r.on_epoch_end(epoch, losses)

    def regularization(self, default):
        # Chained: every regime sees the term the previous ones left.
        changed = False
        for r in self.regimes:
            out = r.regularization(default)
            if out is not NotImplemented:
                default, changed = out, True
        return default if changed else NotImplemented

    def after_step(self):
        for r in self.regimes:
            r.after_step()

    def additional_loss(self, epoch):
        total = None
        for r in self.regimes:
            extra = r.additional_loss(epoch)
            if extra is not None:
                total = extra if total is None else total + extra
        return total
