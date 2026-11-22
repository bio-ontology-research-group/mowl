"""The training design of ELK-Em (Singh et al. 2026) as separate regimes.

The published method trains on one mini-batch per normal form and step, with a correlation
term on corrupted subsumptions when the ontology has no assertions, negatives and a weighting
for the assertions, an optional lexical regulariser, a step decay of the learning rate and
the checkpoint with the best median validation rank. Each of these is a regime of its own that
other methods can use (box and ball models with the same needs), and :class:`ELKEmTraining`
names the combination that ELK-Em uses by default:

* :class:`FixedConceptRouting`: axioms of classes that the class model fixes to the empty box
  are dropped or trained as their ``_bot`` normal form;
* :class:`AssertionCorruption`: negatives of class and role assertions (a class or an object
  replaced, observed assertions filtered out);
* :class:`~mowl.training.PerNormalFormMiniBatches` with ``order="permutation"``;
* :class:`~mowl.training.SumLoss` with weights for the assertion forms;
* :class:`PearsonContrastiveTerm`: the correlation term on corruptions of ``gci0`` and ``gci2``;
* :class:`LexicalRegularisation`: external representations tie the individuals;
* :class:`~mowl.training.StepDecay` and :class:`ValidationCheckpoint`.

All sampling draws from the global torch random number generator, in the order of the direct
implementation (``mowl.baselines.models.elkem``), so that the two train identically from the
same seed.
"""

import numpy as np
import torch as th

from mowl.training.regime import ComposedRegime, TrainingRegime

ASSERTION_FORMS = ("class_assertion", "object_property_assertion")


# ------------------------------------------------------------------------------ helpers

def fixed_class_masks(model):
    """Indices of the classes that the class model of ``model`` fixes to the empty box and to
    the full box (empty lists when it fixes none)."""
    classes = getattr(model.module, "class_model", None)
    out = []
    for name in ("bot_mask", "top_mask"):
        mask = getattr(classes, name, None)
        out.append([] if mask is None else th.nonzero(mask).flatten().tolist())
    return out[0], out[1]


def route_fixed_empty(datasets, bot_ids, device="cpu"):
    """Training axioms per normal form with the rows of classes fixed to the empty box
    rerouted: ``gci0`` rows of a fixed-empty subclass are dropped (they hold trivially) and
    ``gci1``/``gci3`` rows with a fixed-empty superclass are trained as ``gci1_bot`` /
    ``gci3_bot``. Returns ``{form: tensor}`` without empty forms."""
    tensors = {k: v[:].to(device) for k, v in datasets.items() if len(v) > 0}
    bot = th.tensor(list(bot_ids), dtype=th.long, device=device)
    if "gci0" in tensors:
        tensors["gci0"] = tensors["gci0"][~th.isin(tensors["gci0"][:, 0], bot)]
    for form in ("gci1", "gci3"):
        if form in tensors:
            data = tensors[form]
            to_bot = th.isin(data[:, 2], bot)
            if to_bot.any():
                bot_form = form + "_bot"
                moved = data[to_bot]
                tensors[bot_form] = (th.cat([tensors[bot_form], moved])
                                     if bot_form in tensors else moved)
                tensors[form] = data[~to_bot]
    return {k: v for k, v in tensors.items() if len(v) > 0}


def class_pool(model, device=None):
    """The named classes of ``model`` except those its class model fixes (``owl:Thing``,
    classes fixed to the empty box): the classes that corruptions may use."""
    bot, top = fixed_class_masks(model)
    excluded = set(top) | set(bot)
    pool = [i for i in model.named_class_index_dict.values() if i not in excluded]
    return th.tensor(pool, dtype=th.long, device=device or model.device)


def resample(pool, n, forbidden, rounds=20):
    """Draw ``n`` entries of ``pool``, redrawing those equal to a forbidden value."""
    sample = pool[th.randint(len(pool), (n,), device=pool.device)]
    for _ in range(rounds):
        clash = th.zeros(n, dtype=th.bool, device=pool.device)
        for values in forbidden:
            clash |= sample == values
        if not clash.any():
            break
        sample[clash] = pool[th.randint(len(pool), (int(clash.sum()),), device=pool.device)]
    return sample


def corrupt_columns(data, num_negatives, columns, pool, forbid=None):
    """``num_negatives`` corruptions per row of ``data`` (rows grouped by positive), each
    replacing one entity by a random entry of ``pool``.

    :param columns: One column: that column is replaced. Two columns ``(a, b)``: ``b`` with
        probability 1/2, else ``a``.
    :param forbid: With one column: the columns whose values the replacement must differ
        from (default: the replaced column only). With two columns the replacement differs
        from the entity it replaces.
    """
    pos = data.repeat_interleave(num_negatives, dim=0)
    neg = pos.clone()
    if len(columns) == 1:
        column = columns[0]
        avoid = (column,) if forbid is None else tuple(forbid)
        neg[:, column] = resample(pool, len(pos), [pos[:, j] for j in avoid])
        return neg
    first, second = columns
    corrupt_second = th.rand(len(pos), device=pos.device) < 0.5
    avoid = th.where(corrupt_second, pos[:, second], pos[:, first])
    new = resample(pool, len(pos), (avoid,))
    neg[:, first] = th.where(corrupt_second, pos[:, first], new)
    neg[:, second] = th.where(corrupt_second, new, pos[:, second])
    return neg


def corrupt_assertions(data, observed, column, pool, num_negatives, rounds=20):
    """``num_negatives`` corruptions per assertion, replacing ``column`` by an entry of
    ``pool``; corruptions that are among the ``observed`` assertions are redrawn."""
    pos = data.repeat_interleave(num_negatives, dim=0)
    neg = pos.clone()
    width = int(max(pos.max().item(), observed.max().item() if observed is not None else 0)) + 1

    def encode(rows):
        code = th.zeros(len(rows), dtype=th.long, device=rows.device)
        for j in range(rows.shape[1]):
            code = code * width + rows[:, j]
        return code

    known = encode(observed) if observed is not None else None
    neg[:, column] = resample(pool, len(neg), (pos[:, column],))
    for _ in range(rounds):
        if known is None:
            break
        clash = th.isin(encode(neg), known)
        if not clash.any():
            break
        neg[clash, column] = pool[th.randint(len(pool), (int(clash.sum()),),
                                             device=pool.device)]
    return neg


def prepare_lexical(features, individual_index, device="cpu"):
    """External representations as ``(individual indices, matrix)`` tensors, from a mapping
    IRI -> vector or an array with one row per individual."""
    if features is None:
        return None, None
    if isinstance(features, dict):
        idx = [individual_index[iri] for iri in features if iri in individual_index]
        feats = [np.asarray(features[iri], dtype=np.float32) for iri in features
                 if iri in individual_index]
        if not idx:
            raise ValueError("No key of lexical_features is an individual of the dataset.")
        feats = np.stack(feats)
    else:
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != len(individual_index):
            raise ValueError("lexical_features must have one row per individual "
                             f"({len(individual_index)}), got shape {feats.shape}.")
        idx = list(range(len(individual_index)))
    return (th.tensor(idx, dtype=th.long, device=device), th.tensor(feats, device=device))


def pearson_loss(pos_scores, neg_scores, eps=1e-8):
    """One minus the Pearson correlation of the scores with the labels (1 for positives, 0
    for corruptions); the two score vectors may differ in length."""
    scores = th.cat([pos_scores, neg_scores])
    labels = th.cat([th.ones_like(pos_scores), th.zeros_like(neg_scores)])
    s = scores - scores.mean()
    t = labels - labels.mean()
    return 1 - (s * t).sum() / (s.norm() * t.norm() + eps)


def subsumption_median_rank(model, chunk_budget=2_000_000):
    """Median rank of the validation subsumptions :math:`C \\sqsubseteq D` (``gci0``) among
    all named classes (unfiltered, ties at their average rank, :math:`\\top` and the query
    excluded), by the margin-free loss of the module; ``inf`` without validation axioms."""
    data = model.validation_datasets.get("gci0")
    if data is None or len(data) == 0:
        return float("inf")
    ranks = _rank_subsumptions(model, data[:], chunk_budget)
    return float(ranks.median()) if len(ranks) else float("inf")


def _rank_subsumptions(model, pairs, chunk_budget=2_000_000):
    """Average-tie ranks of the superclasses of ``pairs`` (rows ``[C, D]``)."""
    _, top_ids = fixed_class_masks(model)
    top = set(top_ids)
    candidates = th.tensor([i for i in model.named_class_index_dict.values()
                            if i not in top], dtype=th.long, device=model.device)
    cand_set = set(candidates.tolist())
    pairs = pairs[[d in cand_set for d in pairs[:, 1].tolist()]]
    n = len(candidates)
    step = max(1, chunk_budget // max(n, 1))
    ranks = []
    was_training = model.module.training
    model.module.eval()
    with th.no_grad():
        for start in range(0, len(pairs), step):
            chunk = pairs[start:start + step].to(model.device)
            q = len(chunk)
            grid = th.stack([chunk[:, 0].repeat_interleave(n), candidates.repeat(q)], dim=1)
            losses = model.module.zero_margin_loss(grid, "gci0").view(q, n)
            target = model.module.zero_margin_loss(chunk, "gci0").unsqueeze(1)
            others = (candidates.unsqueeze(0) != chunk[:, :1]) & \
                (candidates.unsqueeze(0) != chunk[:, 1:2])
            below = ((losses < target) & others).sum(dim=1)
            ties = ((losses == target) & others).sum(dim=1)
            ranks.append(1 + below.double() + ties.double() / 2)
    model.module.train(was_training)
    return th.cat(ranks) if ranks else th.empty(0, dtype=th.double)


# ------------------------------------------------------------------------------ regimes

class FixedConceptRouting(TrainingRegime):
    """Axioms of concepts that the class model fixes to the empty box are not trained as
    they are: ``gci0`` axioms with such a subclass hold trivially and are dropped, ``gci1`` and
    ``gci3`` axioms with such a superclass are trained as ``gci1_bot`` / ``gci3_bot`` (ELK-Em,
    Table 2: "fixed :math:`B_C = [\\infty, -\\infty]`"). The classes are those of the
    ``bot_mask`` of the class model of the module; without such a mask the axioms are only
    moved to the device and empty forms dropped.
    """

    key = "fixed-concept-routing"

    def training_datasets(self, datasets):
        bot, _ = fixed_class_masks(self.model)
        return route_fixed_empty(datasets, bot, self.model.device)


class AssertionCorruption(TrainingRegime):
    """Negatives of the assertions of the ABox, for modules that give them a negative loss:
    ``num_negatives`` corruptions per class assertion :math:`C(a)` (the class is replaced by a
    random class that the class model does not fix) and per role assertion :math:`r(a, b)` (the
    object is replaced by a random individual); corruptions that are observed training
    assertions are redrawn. Every other axiom type gets no negatives.

    :param num_negatives: Corruptions per assertion (``abox_k_neg`` of ELK-Em; 2 in the paper).
    """

    key = "assertion-corruption"

    def __init__(self, num_negatives=2):
        super().__init__()
        if num_negatives < 1:
            raise ValueError("num_negatives must be at least 1.")
        self.num_negatives = int(num_negatives)
        self._observed = {}

    def training_datasets(self, datasets):
        self._observed = {k: v[:] for k, v in datasets.items() if k in ASSERTION_FORMS}
        return datasets

    def negatives(self, gci_name, gci_dataset):
        model = self.model
        if gci_name not in ASSERTION_FORMS:
            return None
        data = gci_dataset[:].to(model.device)
        observed = self._observed.get(gci_name)
        if gci_name == "class_assertion":
            column, pool = 1, class_pool(model)
        else:
            column = 2
            pool = th.tensor(list(model.individual_index_dict.values()), dtype=th.long,
                             device=model.device)
        return corrupt_assertions(data, observed, column, pool, self.num_negatives)


class PearsonContrastiveTerm(TrainingRegime):
    """A correlation term on corrupted subsumptions, added once per step (ELK-Em, Section 4.2,
    Eq. 5). For every form of ``corruptions`` the step's batch is corrupted
    ``num_negatives`` times per axiom and one minus the Pearson correlation of the margin-free
    scores :math:`\\exp(-\\ell^0)` of the positives and of the corruptions with their labels is
    summed over the forms and multiplied by ``weight``. It is used only when the training
    data has no assertions (``active=None``), as in the paper.

    :param weight: ``contrastive_factor`` (3.0).
    :param num_negatives: Corruptions per positive (``k_neg``; 3 or 5 in the paper).
    :param corruptions: ``{form: (columns, forbidden columns)}``; default: ``gci0`` with its
        superclass replaced (never equal to either class) and ``gci2`` with either class
        replaced.
    :param active: ``True``/``False`` to force the term on or off; ``None``: on iff the
        training data has no assertion forms.
    """

    key = "pearson-contrastive-term"

    def __init__(self, weight=3.0, num_negatives=3, corruptions=None, active=None):
        super().__init__()
        if num_negatives < 1:
            raise ValueError("num_negatives must be at least 1.")
        self.weight = weight
        self.num_negatives = int(num_negatives)
        self.corruptions = dict(corruptions) if corruptions is not None else {
            "gci0": ((1,), (0, 1)), "gci2": ((0, 2), None)}
        self.active = active
        self._batches = {}
        self._assertions = False

    def training_datasets(self, datasets):
        self._assertions = any(k in datasets and len(datasets[k]) > 0 for k in ASSERTION_FORMS)
        return datasets

    @property
    def is_active(self):
        """Whether the term is used (see ``active``)."""
        if self.active is not None:
            return bool(self.active)
        return not self._assertions

    def corrupt(self, form, data):
        """The corruptions of ``data`` of ``form`` (a tensor of rows)."""
        columns, forbid = self.corruptions[form]
        return corrupt_columns(data.to(self.model.device), self.num_negatives, columns,
                               class_pool(self.model), forbid)

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        if gci_name in self.corruptions:
            self._batches[gci_name] = gci_dataset[:]
        return negatives

    def additional_loss(self, epoch):
        batches, self._batches = self._batches, {}
        if not self.is_active or not self.weight:
            return None
        module = self.model.module
        loss = None
        for form in self.corruptions:
            pos = batches.get(form)
            if pos is None or len(pos) == 0:
                continue
            neg = self.corrupt(form, pos)
            term = pearson_loss(module.score(pos, form), module.score(neg, form))
            loss = term if loss is None else loss + term
        return None if loss is None else self.weight * loss


class LexicalRegularisation(TrainingRegime):
    """The lexical regulariser of ELK-Em (Eq. 6): on a random batch of ``batch_size``
    individuals with external representations (for example text embeddings), the pairwise
    similarity of the representations is matched by the geometric similarity
    :math:`\\exp(-\\lambda\\lVert e_i - e_j\\rVert^2 / n)` of the points. The module must have
    ``lexical_loss(indices, features, decay)``.

    :param features: Mapping individual IRI -> vector, or an array with one row per
        individual (in the order of the model's ``individual_index_dict``).
    :param weight: Weight of the term.
    :param decay: :math:`\\lambda`.
    :param batch_size: Individuals per step.
    """

    key = "lexical-regularisation"

    def __init__(self, features, weight=1.0, decay=1.0, batch_size=2000):
        super().__init__()
        self.features = features
        self.weight = weight
        self.decay = decay
        self.batch_size = batch_size
        self.indices = self.matrix = None

    def attach(self, model):
        super().attach(model)
        self.indices, self.matrix = prepare_lexical(self.features, model.individual_index_dict,
                                                    model.device)

    def term(self):
        """The unweighted regulariser on a random batch, or ``None`` without features."""
        if self.indices is None:
            return None
        pick = th.randperm(len(self.indices), device=self.model.device)[:self.batch_size]
        return self.model.module.lexical_loss(self.indices[pick], self.matrix[pick], self.decay)

    def additional_loss(self, epoch):
        lex = self.term()
        if lex is None or not self.weight:
            return None
        return self.weight * lex


class ValidationCheckpoint(TrainingRegime):
    """Model selection that saves the best checkpoint to the model's ``model_filepath``:
    every ``every`` epochs the module is evaluated by ``metric(model)`` (lower is better;
    default: the median rank of the validation subsumptions,
    :func:`subsumption_median_rank`) and its parameters are saved when it improves
    (``model.best_validation_median_rank`` records the best value); without validation
    axioms, or if no evaluation improved, the final parameters are saved. The module keeps
    its final parameters: ``model.load_best_model()`` (and ``get_embeddings``) load the
    checkpoint. This replaces the built-in validation of the generic loop (by a loss), which is
    skipped while this regime is attached.

    :param metric: ``f(model) -> float``, or ``None`` for the median rank.
    :param every: Epochs between evaluations (the paper: 100); ``None``: the ``validate_every``
        of the call of ``train``.
    """

    key = "validation-checkpoint"
    replaces_validation = True

    def __init__(self, metric=None, every=None):
        super().__init__()
        self.metric = metric or subsumption_median_rank
        self.every = None if every is None else max(1, int(every))
        self.best = float("inf")
        self.history = []

    def attach(self, model):
        super().attach(model)
        self.best, self.history = float("inf"), []

    def _has_validation(self):
        dataset = self.model.dataset
        if dataset.validation is None:
            return False
        data = self.model.validation_datasets.get("gci0")
        return data is not None and len(data) > 0

    def training_datasets(self, datasets):
        self.best, self.history = float("inf"), []
        return datasets

    def on_epoch_end(self, epoch, losses):
        every = self.every or max(1, int(getattr(self.model, "_train_validate_every", 1)))
        if not self._has_validation() or (epoch + 1) % every:
            return
        value = float(self.metric(self.model))
        self.history.append((epoch + 1, value))
        if value < self.best:
            self.best = value
            self.model.best_validation_median_rank = value
            th.save(self.model.module.state_dict(), self.model.model_filepath)

    def on_train_end(self):
        if not self._has_validation() or self.best == float("inf"):
            th.save(self.model.module.state_dict(), self.model.model_filepath)


class ELKEmTraining(ComposedRegime):
    """How ELK-Em trains (Singh et al. 2026, Sections 4 and 5): the combination of
    :class:`FixedConceptRouting`, :class:`AssertionCorruption`,
    :class:`~mowl.training.PerNormalFormMiniBatches` (a random subset of ``batch_size`` axioms
    per normal form and step), :class:`~mowl.training.SumLoss` (weights of the assertion
    losses), :class:`PearsonContrastiveTerm`, :class:`LexicalRegularisation` (given features),
    :class:`~mowl.training.StepDecay` (given ``lr_decay_epoch``) and
    :class:`ValidationCheckpoint`. Every option is an option of
    :class:`~mowl.models.ELKEm` and has its value there. Use the parts one by one to change a
    single piece, or :class:`~mowl.training.ELKEmCodeTraining` for the design of the authors'
    code.

    :param batch_size: Axioms per normal form and step; ``None``: the model's.
    :param k_neg: Corruptions per positive of the correlation term.
    :param contrastive_factor: Weight of the correlation term.
    :param use_contrastive: Force the correlation term on or off (default: on iff no
        assertions).
    :param abox_k_neg: Corruptions per assertion.
    :param abox_pos_weight, abox_neg_weight: Weights of the positive and corrupted assertion
        losses.
    :param lexical_features, lexical_weight, lexical_decay, lexical_batch_size: See
        :class:`LexicalRegularisation`; no features: no regulariser.
    :param lr_decay_epoch, lr_decay_factor: Epoch of the step decay (``None``: none) and its
        factor.
    :param validate_every: Epochs between evaluations of the model selection (``None``: the
        ``validate_every`` of the call of ``train``).
    :param select: ``False``: no model selection (the final state is not saved either).
    """

    key = "elkem-training"

    def __init__(self, batch_size=None, k_neg=3, contrastive_factor=3.0, use_contrastive=None,
                 abox_k_neg=2, abox_pos_weight=1.0, abox_neg_weight=1.0,
                 lexical_features=None, lexical_weight=1.0, lexical_decay=1.0,
                 lexical_batch_size=2000, lr_decay_epoch=5000, lr_decay_factor=0.1,
                 validate_every=None, select=True):
        from mowl.training.batching import PerNormalFormMiniBatches
        from mowl.training.losses import SumLoss
        from mowl.training.schedules import StepDecay
        parts = [FixedConceptRouting(), AssertionCorruption(abox_k_neg),
                 PerNormalFormMiniBatches(batch_size=batch_size, order="permutation"),
                 SumLoss(neg_weight=abox_neg_weight, pos_weight=abox_pos_weight,
                         normal_forms=ASSERTION_FORMS),
                 PearsonContrastiveTerm(contrastive_factor, k_neg, active=use_contrastive)]
        if lexical_features is not None:
            parts.append(LexicalRegularisation(lexical_features, lexical_weight, lexical_decay,
                                               lexical_batch_size))
        if lr_decay_epoch is not None:
            parts.append(StepDecay([lr_decay_epoch], gamma=lr_decay_factor))
        if select:
            parts.append(ValidationCheckpoint(every=validate_every))
        super().__init__(*parts)
        self.key = type(self).key
