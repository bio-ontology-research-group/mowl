"""Training regimes and negative samplers of the canonical mixture.

The trained part of the canonical mixture is the individual model, so its training loop goes
over *individuals*: :class:`IndividualBatches` (the method's own regime) draws a random batch
of individuals, takes their training assertions, samples one negative per assertion and makes
one optimiser step per batch, with the regularisers of the geometry evaluated on the batch.
:class:`AssertionBatches` goes over the assertions instead (minibatches of every form in
turn), as a swap.

Negatives are parts of the regime: :class:`NonEntailedClasses` corrupts the class of a class
assertion with a class that the constructed model does not entail for the individual
(rejection sampling against its own element's classes), :class:`UniformTails` corrupts the
object of a role assertion with a uniformly drawn individual.

Losses and regularisers are evaluated in chunks of bounded size (the support entries of the
individuals, :meth:`MixtureIndividuals.pair_chunks
<mowl.representation.library.mixture_individuals.MixtureIndividuals.pair_chunks>`)
and their gradients accumulated, so that a batch never holds more than a bounded number of
entries.
"""

import os

import torch as th
from tqdm import trange

__all__ = ["NonEntailedClasses", "UniformTails", "IndividualBatches", "AssertionBatches"]

_FORMS = (("class_assertion", 2), ("object_property_assertion", 3))


class NonEntailedClasses:
    """One negative per class assertion ``(a, C)``: ``(a, C')`` with a class :math:`C'` the
    constructed model does not entail for :math:`a`, drawn uniformly by rejection (``tries``
    draws in all), then the next class (cyclically, from a random start) that is not
    entailed. The membership table is that of the geometry's class model."""

    def __init__(self, tries=20):
        self.tries = int(tries)

    def sample(self, geometry, rows):
        table = geometry.class_model.table
        device = rows.device
        if rows.numel() == 0:
            return th.empty((0, 2), dtype=th.long, device=device)
        ind = rows[:, 0]
        n_classes = table.n_classes
        own = geometry.individual_model.individual_element.index_select(0, ind)
        neg = th.randint(0, n_classes, (len(ind),), device=device, dtype=th.long)
        bad = table.contains(own, neg)
        for _ in range(self.tries - 1):
            if not bool(bad.any()):
                break
            neg[bad] = th.randint(0, n_classes, (int(bad.sum()),), device=device,
                                  dtype=th.long)
            bad = table.contains(own, neg)
        for r in th.nonzero(bad, as_tuple=False).reshape(-1).tolist():
            el = int(own[r])
            if el < 0:
                continue
            entailed = set(table.types_of(el).tolist())
            if len(entailed) < n_classes:
                start = int(th.randint(0, n_classes, (1,), device=device, dtype=th.long))
                for step in range(n_classes):
                    cand = (start + step) % n_classes
                    if cand not in entailed:
                        neg[r] = cand
                        break
        return th.stack([ind, neg], dim=1)


class UniformTails:
    """One negative per role assertion ``(a, r, b)``: ``(a, r, b')`` with :math:`b'` a
    uniformly drawn individual (drawn in blocks of the rows of a chunk)."""

    def sample(self, geometry, rows):
        device = rows.device
        if rows.numel() == 0:
            return th.empty((0, 3), dtype=th.long, device=device)
        model = geometry.individual_model
        n, step = len(rows), model.row_chunk(len(rows))
        out = rows.clone()
        for s in range(0, n, step):
            e = min(n, s + step)
            out[s:e, 2] = th.randint(0, len(model), (e - s,), device=device, dtype=th.long)
        return out


def _data(model):
    out = {}
    for name, width in _FORMS:
        ds = model.training_datasets.get(name)
        out[name] = ds[:].to(model.device) if ds is not None and len(ds) else \
            th.empty((0, width), dtype=th.long, device=model.device)
    return out


def _backward(geometry, rows, form, neg=False):
    """The mean loss of ``rows``, its gradient accumulated chunk by chunk; returns its
    value."""
    n = int(rows.shape[0])
    if n == 0:
        return 0.0
    total = 0.0
    for s, e in geometry.individual_model.pair_chunks(rows):
        loss = geometry(rows[s:e], form, neg=neg).mean() * (float(e - s) / float(n))
        if loss.requires_grad:
            loss.backward()
        total += float(loss.detach())
    return total


def _penalties(geometry, batch):
    """The regularisers of the geometry on the individuals ``batch`` (chunk by chunk for
    those that read a batch), their gradients accumulated; returns their value."""
    total = 0.0
    for regularizer in geometry.regularizers:
        if getattr(regularizer, "kind", "penalty") != "penalty":
            continue
        if getattr(regularizer, "batch_penalty", False):
            n = max(int(batch.shape[0]), 1)
            for s, e in geometry.individual_model.chunks(batch):
                term = regularizer.penalty(geometry, batch[s:e]) * (float(e - s) / n)
                if term.requires_grad:
                    term.backward()
                total += float(term.detach())
        else:
            term = regularizer.penalty(geometry)
            if term is not None:
                term.backward()
                total += float(term.detach())
    return total


def _project(geometry):
    for regularizer in geometry.regularizers:
        if getattr(regularizer, "kind", None) == "projection":
            regularizer.project(geometry)
    geometry.project()


class _Regime:
    """What the regimes share: negatives, validation and the best checkpoint."""

    key = ""

    def __init__(self, batch_size=None, class_negatives=None, role_negatives=None):
        self.batch_size = batch_size
        self.class_negatives = class_negatives if class_negatives is not None \
            else NonEntailedClasses()
        self.role_negatives = role_negatives if role_negatives is not None else UniformTails()

    def sampler(self, form):
        return self.class_negatives if form == "class_assertion" else self.role_negatives

    @staticmethod
    def validation_loss(model):
        """Mean positive loss of the validation assertions of ``model.eval_gci_name`` (or
        of the first assertion form that has some), in batches."""
        datasets = model.validation_datasets
        if not datasets:
            return 0.0
        module = model.module
        with th.no_grad():
            module.eval()
            for form in (model.eval_gci_name, "class_assertion", "object_property_assertion"):
                if form in datasets and len(datasets[form]) > 0:
                    rows = datasets[form][:]
                    step = max(int(model.batch_size), 1)
                    total = sum(float(module(rows[s:s + step], form).detach().sum())
                                for s in range(0, len(rows), step))
                    return total / float(len(rows))
        return 0.0

    def run(self, model, epochs, validate_every=1, epoch_callback=None):
        data = _data(model)
        optimizer = model.get_optimizer()
        best, saved = float("inf"), False
        model.loss_history = []
        for epoch in trange(epochs):
            model.module.train()
            loss = self.epoch(model, data, optimizer)
            model.loss_history.append(loss)
            if epoch_callback is not None:
                epoch_callback(epoch, model)
            if validate_every and (epoch + 1) % validate_every == 0:
                value = self.validation_loss(model) if model.dataset.validation is not None \
                    else loss
                if value < best:
                    best = value
                    th.save(model.module.state_dict(), model.model_filepath)
                    saved = True
        if saved and model.model_filepath is not None and os.path.exists(model.model_filepath):
            model.module.load_state_dict(th.load(model.model_filepath,
                                                 map_location=model.device, weights_only=True))


class IndividualBatches(_Regime):
    """The regime of the canonical mixture: per epoch a random order of the individuals, in
    batches of ``batch_size``; per batch the training assertions of its individuals (at most
    ``batch_size`` of every form, drawn at random if there are more) with one negative each,
    the regularisers on the batch, and one optimiser step if anything was supervised. The
    epoch loss is the mean of the losses of the steps. After every ``validate_every`` epochs
    the validation loss (or, without validation data, the epoch loss) selects the best
    parameters, which are saved to ``model.model_filepath`` and restored at the end.

    :param batch_size: Individuals per batch (default: the model's).
    :param class_negatives, role_negatives: The negative samplers.
    """

    key = "individual-batches"

    def epoch(self, model, data, optimizer):
        geometry = model.module
        batch_size = self.batch_size or model.batch_size
        n_ind = len(geometry.individual_model)
        perm = th.randperm(n_ind, device=model.device)
        total, steps = 0.0, 0
        for start in range(0, n_ind, batch_size):
            batch = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss, supervised = 0.0, False
            for form, _ in _FORMS:
                rows = data[form]
                if rows.shape[0] == 0:
                    continue
                mask = th.isin(rows[:, 0], batch)
                if not bool(mask.any()):
                    continue
                pos = rows[mask]
                if pos.shape[0] > batch_size:
                    pos = pos.index_select(0, th.randperm(pos.shape[0],
                                                          device=model.device)[:batch_size])
                neg = self.sampler(form).sample(geometry, pos)
                loss += _backward(geometry, pos, form)
                loss += _backward(geometry, neg, form, neg=True)
                supervised = True
            penalised = any(getattr(r, "kind", "penalty") == "penalty"
                            for r in geometry.regularizers)
            if penalised:
                loss += _penalties(geometry, batch)
            if supervised or penalised:
                optimizer.step()
                _project(geometry)
                total += loss
                steps += 1
        return total / max(steps, 1)


class AssertionBatches(_Regime):
    """A swap for :class:`IndividualBatches`: per epoch the class assertions and then the
    role assertions, each in a random order and in minibatches of ``batch_size``
    assertions, one negative each; one step per minibatch, the regularisers on the
    individuals of the minibatch."""

    key = "assertion-batches"

    def epoch(self, model, data, optimizer):
        geometry = model.module
        batch_size = self.batch_size or model.batch_size
        total, steps = 0.0, 0
        for form, _ in _FORMS:
            rows = data[form]
            perm = th.randperm(rows.shape[0], device=model.device)
            for s in range(0, rows.shape[0], batch_size):
                pos = rows.index_select(0, perm[s:s + batch_size])
                optimizer.zero_grad()
                loss = _backward(geometry, pos, form)
                loss += _backward(geometry, self.sampler(form).sample(geometry, pos), form,
                                  neg=True)
                ids = th.unique(pos[:, 0] if form == "class_assertion"
                                else th.cat([pos[:, 0], pos[:, 2]]))
                loss += _penalties(geometry, ids)
                optimizer.step()
                _project(geometry)
                total += loss
                steps += 1
        return total / max(steps, 1)
