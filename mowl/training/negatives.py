"""Negative sampling regimes for :math:`\\mathcal{EL}^{++}` models.

Every geometric :math:`\\mathcal{EL}^{++}` method trains with corrupted axioms ("negatives"),
but the papers differ in *which* column of *which* normal form is corrupted, *from which
pool* the replacement is drawn, *how many* negatives a positive gets, and *which* sampled
negatives are rejected. These choices are part of the training regime, not of the
representation, and they change published numbers (see
``docs/source/methods/training_and_evaluation.rst``). This module implements them as
composable :class:`~mowl.training.TrainingRegime` objects:

* generators (the ``negatives`` hook): :class:`FixedColumnCorruption`,
  :class:`EitherSideCorruption`, :class:`PerColumnCorruption` (all instances of
  :class:`CorruptionNegatives`), :class:`TypeConstrainedCorruption`,
  :class:`SiblingHardNegatives`, :class:`ContradictionNegatives`;
* filters (the ``filter_negatives`` hook): :class:`KnownPositiveFilter`.

Layout convention. All generators return ``K * n`` rows for ``n`` positives, as ``K``
consecutive blocks of ``n`` rows; row ``i`` is a corruption of positive ``i mod n``. This
is the layout of :meth:`~mowl.base_models.elmodel.EmbeddingELModel.generate_negatives`,
and the filters of this module and of :mod:`mowl.training.dele` rely on it.

Generators return negatives only for the normal forms they are configured for and
``NotImplemented`` otherwise (the model's own sampler is then used). By default they refuse
normal forms whose module has no negative loss (``module.neg_capable_gcis``), because the
loop scores negatives with ``module(..., neg=True)``, which for such a module is the
*positive* loss. Pass ``require_negative_loss=False`` when a score-based loss regime of
:mod:`mowl.training.losses` recomputes the scores of the negatives itself.
"""

import numpy as np
import torch as th

from mowl.training.regime import TrainingRegime

#: Columns holding object property indices, per axiom type (mOWL column layout).
ROLE_COLUMNS = {"gci2": (1,), "gci3": (0,), "gci3_bot": (0,),
                "object_property_assertion": (1,),
                "role_inclusion": (0, 1), "role_chain": (0, 1, 2)}

#: Columns holding individual indices, per axiom type.
INDIVIDUAL_COLUMNS = {"class_assertion": (0,), "object_property_assertion": (0, 2)}

#: Column corrupted by the default sampler of mOWL's EL models
#: (:attr:`~mowl.base_models.elmodel.EmbeddingELModel._DEFAULT_NEG_SAMPLING_CONFIG`): the
#: right-hand side of a subsumption, the class of a class assertion, the object of a role
#: assertion.
FIXED_COLUMNS = {"gci0": (1,), "gci1": (2,), "gci2": (2,), "gci3": (2,),
                 "class_assertion": (1,), "object_property_assertion": (2,)}

#: The two "sides" of every axiom type: the concepts (or individuals) on the left- and on
#: the right-hand side. For ``gci1`` (:math:`C \\sqcap D \\sqsubseteq E`) both conjuncts
#: count as left-hand side.
SIDE_COLUMNS = {"gci0": (0, 1), "gci1": (0, 1, 2), "gci2": (0, 2), "gci3": (1, 2),
                "class_assertion": (0, 1), "object_property_assertion": (0, 2)}

MODES = ("fixed", "either", "each", "all")


def column_kind(gci_name, column):
    """``"roles"``, ``"individuals"`` or ``"classes"``: the kind of entity in a column."""
    if column in ROLE_COLUMNS.get(gci_name, ()):
        return "roles"
    if column in INDIVIDUAL_COLUMNS.get(gci_name, ()):
        return "individuals"
    return "classes"


def default_pool(model, gci_name, column):
    """The indices a corrupted entity of ``column`` is drawn from by default: the named
    classes of the model (never the auxiliary classes of the normal forms), its
    individuals, or its object properties."""
    kind = column_kind(gci_name, column)
    if kind == "roles":
        values = model.object_property_index_dict.values()
    elif kind == "individuals":
        values = model.individual_index_dict.values()
    else:
        values = model.named_class_index_dict.values()
    return np.fromiter(values, dtype=np.int64)


def _draw(pool, size, rng):
    """``size`` uniform draws from ``pool``. With ``rng=None`` NumPy's global generator is
    used exactly as the model's own sampler uses it (``np.random.choice``), so that a
    regime with ``seed=None`` reproduces the model's negatives under ``np.random.seed``."""
    if len(pool) == 0:
        raise ValueError("Cannot sample negatives from an empty pool.")
    if rng is None:
        return np.asarray(np.random.choice(pool, size=size, replace=True), dtype=np.int64)
    return pool[rng.integers(0, len(pool), size=size)]


def _check_alignment(positives, negatives):
    n = len(positives)
    return n > 0 and len(negatives) % n == 0


def corrupted_columns(positives, negatives):
    """For every negative row, the list of columns in which it differs from the positive
    it was derived from (positive ``i mod n``), or ``None`` if the negatives are not
    aligned with the positives."""
    if not _check_alignment(positives, negatives):
        return None
    n = len(positives)
    reps = positives.repeat(len(negatives) // n, 1).to(negatives.device)
    diff = negatives != reps
    return [th.nonzero(d).flatten().tolist() for d in diff]


class CorruptionNegatives(TrainingRegime):
    """Uniform corruption of one or more columns of the training axioms.

    For every positive, ``num_negatives`` negatives are produced per *block*; what a block
    is depends on ``mode``:

    * ``"fixed"``: one block per negative; the first configured column is corrupted
      (:class:`FixedColumnCorruption`).
    * ``"either"``: one block per negative; every row corrupts one of the configured
      columns, chosen uniformly at random per row (:class:`EitherSideCorruption`).
    * ``"each"``: one block per configured column and negative; every block corrupts its
      own column (:class:`PerColumnCorruption`), so a positive gets
      ``len(columns) * num_negatives`` negatives.
    * ``"all"``: one block per negative; all configured columns are corrupted at once.

    :param columns: ``{axiom type: column or tuple of columns}`` to corrupt. Axiom types
        that are not keys get no negatives from this regime (``NotImplemented``: the
        model's sampler is used). Defaults depend on the subclass; the defaults are
        restricted to the axiom types the module has a negative loss for
        (``neg_capable_gcis``), as the model's own sampler does.
    :param mode: One of ``"fixed"``, ``"either"``, ``"each"``, ``"all"``.
    :param num_negatives: Negatives per positive (per column for ``mode="each"``).
    :param pools: Optional ``{axiom type: {column: iterable of indices}}`` restricting
        the replacement entities of a column (for example to proteins). Other columns
        use :func:`default_pool`.
    :param different: If ``True``, a replacement equal to the original entity is
        re-drawn (up to ``max_redraw`` times), so that a negative differs from its
        positive ("a randomly selected *different* concept", Box\\ :sup:`2`\\ EL,
        Section 3.2).
    :param require_negative_loss: Refuse axiom types whose module has no negative loss.
    :param seed: Seed of a private generator. ``None`` (default) draws from NumPy's global
        generator in the same way as the model's own sampler, so that under the same
        ``np.random.seed`` :class:`FixedColumnCorruption` and
        :class:`PerColumnCorruption` (one negative per column) reproduce the model's
        default negatives exactly.
    """

    key = ""
    _default_columns = FIXED_COLUMNS
    _default_mode = "fixed"

    def __init__(self, columns=None, mode=None, num_negatives=1, pools=None,
                 different=False, max_redraw=10, require_negative_loss=True, seed=None):
        super().__init__()
        mode = self._default_mode if mode is None else mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
        if int(num_negatives) < 1:
            raise ValueError("num_negatives must be at least 1.")
        self._explicit_columns = columns is not None
        cols = self._default_columns if columns is None else columns
        self.columns = {k: (tuple(v) if isinstance(v, (list, tuple)) else (int(v),))
                        for k, v in cols.items()}
        for k, v in self.columns.items():
            if not v:
                raise ValueError(f"No column given for {k!r}.")
        self.mode = mode
        self.num_negatives = int(num_negatives)
        self.pools = {g: {int(c): np.asarray(list(p), dtype=np.int64)
                          for c, p in cols_.items()}
                      for g, cols_ in (pools or {}).items()}
        self.different = bool(different)
        self.max_redraw = int(max_redraw)
        self.require_negative_loss = bool(require_negative_loss)
        self.rng = None if seed is None else np.random.default_rng(seed)
        self._pool_cache = {}

    def attach(self, model):
        super().attach(model)
        self._pool_cache = {}
        if self.require_negative_loss:
            capable = getattr(getattr(model, "module", None), "neg_capable_gcis", None)
            if capable is not None and not self._explicit_columns:
                # Like the model's own sampler: by default only the axiom types that
                # the module has a negative loss for.
                self.columns = {k: v for k, v in type(self)._default_columns.items()
                                if k in capable}
            elif capable is not None:
                absent = set(self._loaded_but_absent(model))
                missing = sorted(set(self.columns) - set(capable) - absent)
                if missing:
                    raise TypeError(
                        f"{type(self).__name__} was asked for negatives of {missing}, but "
                        f"{type(model.module).__name__} has no negative loss for them "
                        f"(neg_capable_gcis={sorted(capable)}). Restrict 'columns', or "
                        f"pass require_negative_loss=False together with a score-based "
                        f"loss regime of mowl.training.losses.")

    @staticmethod
    def _loaded_but_absent(model):
        # Axiom types the model does not train on at all do not need a negative loss.
        try:
            present = set(model.training_datasets)
        except Exception:
            return ()
        return [g for g in ("class_assertion", "object_property_assertion",
                            "role_inclusion", "role_chain") if g not in present]

    def pool(self, gci_name, column):
        """Indices a replacement for ``column`` of ``gci_name`` is drawn from."""
        key = (gci_name, column)
        if key not in self._pool_cache:
            custom = self.pools.get(gci_name, {}).get(column)
            self._pool_cache[key] = (custom if custom is not None
                                     else default_pool(self.model, gci_name, column))
        return self._pool_cache[key]

    def _replacement(self, gci_name, column, original):
        """Replacement entities for the entities ``original`` (a 1-d LongTensor) of
        ``column``. Subclasses change the pool here."""
        pool = self.pool(gci_name, column)
        new = _draw(pool, len(original), self.rng)
        if self.different and len(pool) > 1:
            orig = original.cpu().numpy()
            for _ in range(self.max_redraw):
                same = new == orig
                if not same.any():
                    break
                new[same] = _draw(pool, int(same.sum()), self.rng)
        return th.as_tensor(new, dtype=th.long)

    def _corrupt(self, gci_name, data, column_choice):
        """Corrupt ``data``; ``column_choice`` is a list of columns (all corrupted) or a
        LongTensor with one column per row."""
        out = data.clone()
        if isinstance(column_choice, th.Tensor):
            for col in set(column_choice.tolist()):
                rows = th.nonzero(column_choice == col).flatten()
                rep = self._replacement(gci_name, col, data[rows, col])
                out[rows, col] = rep.to(out.device)
        else:
            for col in column_choice:
                out[:, col] = self._replacement(gci_name, col, data[:, col]).to(out.device)
        return out

    def negatives(self, gci_name, gci_dataset):
        if gci_name not in self.columns:
            return NotImplemented
        data = gci_dataset[:]
        if len(data) == 0:
            return None
        cols = self.columns[gci_name]
        if max(cols) >= data.shape[1]:
            raise ValueError(f"Column {max(cols)} is out of range for {gci_name!r} "
                             f"({data.shape[1]} columns).")
        blocks = []
        if self.mode == "each":
            for col in cols:
                for _ in range(self.num_negatives):
                    blocks.append(self._corrupt(gci_name, data, [col]))
        else:
            for _ in range(self.num_negatives):
                if self.mode == "fixed":
                    blocks.append(self._corrupt(gci_name, data, [cols[0]]))
                elif self.mode == "all":
                    blocks.append(self._corrupt(gci_name, data, list(cols)))
                else:
                    which = (np.random.randint(0, len(cols), size=len(data))
                             if self.rng is None
                             else self.rng.integers(0, len(cols), size=len(data)))
                    choice = th.as_tensor(np.asarray(cols)[which], dtype=th.long)
                    blocks.append(self._corrupt(gci_name, data, choice))
        return th.cat(blocks, dim=0)


class FixedColumnCorruption(CorruptionNegatives):
    """Corrupt one fixed column of every axiom with a uniformly drawn entity.

    The default sampler of mOWL's EL models, and the scheme of CatE (Zhapa-Camacho and
    Hoehndorf, Section 3.2, Eq. 7: "replacing :math:`B` ... by random sampling in a
    uniform distribution"), of ELK-Em for :math:`C \\sqsubseteq D` (Singh et al.,
    Section 4.2: "replacing :math:`D` with a uniformly sampled concept name, without
    filtering") and of DELE (Mashkova et al., Section 4, one column per normal form).
    Default columns: :data:`FIXED_COLUMNS` (the right-hand side).

    With the default arguments, attaching this regime to a model reproduces the model's
    own default sampler (same columns, same pools, one negative per positive).
    """

    key = "fixed-column-corruption"
    _default_columns = FIXED_COLUMNS
    _default_mode = "fixed"


class EitherSideCorruption(CorruptionNegatives):
    """Every negative corrupts one side of the axiom, chosen uniformly at random.

    The scheme stated for :math:`C \\sqsubseteq \\exists R.D` by ELEmbeddings (Kulmanov
    et al. 2019, Section 3.4: "replacing either :math:`C` or :math:`D`"), EmEL++ (Mondal
    et al., Section 3.3), BoxEL (Xiong et al., Section 4.5: "randomly replacing one of the
    head or tail entity"; Section 5.2 for :math:`C \\sqsubseteq D`), Box\\ :sup:`2`\\ EL
    (Section 3.2, *as stated in the paper*; its reference code corrupts both sides, see
    :class:`PerColumnCorruption`), ELK-Em for :math:`C \\sqsubseteq \\exists r.D`
    (Section 4.2: ":math:`C` or :math:`D` with equal probability") and binary-code boxes
    (Zhang et al. 2022, Section 4.4: "randomly corrupting either the source or target
    node"). Default columns: :data:`SIDE_COLUMNS`.
    """

    key = "either-side-corruption"
    _default_columns = SIDE_COLUMNS
    _default_mode = "either"


class PerColumnCorruption(CorruptionNegatives):
    """Every positive gets one block of negatives per side: one with the left-hand side
    corrupted and one with the right-hand side corrupted.

    This is what the reference implementation of Box\\ :sup:`2`\\ EL does for
    :math:`C \\sqsubseteq \\exists r.D` (``train.py``, ``sample_negatives``: a
    tail-corrupted and a head-corrupted copy of every axiom), and what TransBox (Yang et al.
    2025, Section 4.3: "replacing :math:`A` with :math:`A'` and :math:`B` with
    :math:`B'`") and mOWL's :class:`~mowl.models.TransBox` do. Default columns:
    :data:`SIDE_COLUMNS`. :class:`~mowl.models.TransBox` has this built in for ``gci2``;
    attaching ``PerColumnCorruption(columns={"gci2": (0, 2)})`` to a plain model
    reproduces it.
    """

    key = "per-column-corruption"
    _default_columns = SIDE_COLUMNS
    _default_mode = "each"


class TypeConstrainedCorruption(CorruptionNegatives):
    """Corrupt an entity only with an entity of the *same type*.

    ``types`` partitions the entities into types; a replacement for an entity is drawn
    from the type of that entity (entities without a type are replaced from the column's
    default pool). This is the sampler of the reference implementation of the NeSy 2024
    paper and of DELE (``dataloader.py``: GO classes are replaced by GO classes, protein
    classes by protein classes), and the "type-constrained negative sampling" that
    Mashkova and Hoehndorf (2026, Section 6.3) build on (Krompaß et al. 2015). mOWL's
    PPI example models restrict the replacement of ``gci2`` to proteins, which is the
    one-type case (use ``pools=`` of :class:`FixedColumnCorruption` for that).

    :param types: Iterable of iterables of entity indices (one per type), or a mapping
        ``{index: type label}``.
    :param columns, mode, num_negatives, different, seed: As for
        :class:`CorruptionNegatives`; default columns :data:`FIXED_COLUMNS`, mode
        ``"fixed"``.
    """

    key = "type-constrained-corruption"
    _default_columns = FIXED_COLUMNS
    _default_mode = "fixed"

    def __init__(self, types, columns=None, mode=None, num_negatives=1, different=False,
                 max_redraw=10, require_negative_loss=True, seed=None):
        super().__init__(columns=columns, mode=mode, num_negatives=num_negatives,
                         different=different, max_redraw=max_redraw,
                         require_negative_loss=require_negative_loss, seed=seed)
        if isinstance(types, dict):
            groups = {}
            for idx, label in types.items():
                groups.setdefault(label, []).append(int(idx))
            types = list(groups.values())
        self.type_pools = [np.asarray(sorted(set(int(i) for i in t)), dtype=np.int64)
                           for t in types]
        self.type_of = {}
        for t, members in enumerate(self.type_pools):
            for i in members.tolist():
                if i in self.type_of:
                    raise ValueError(f"Entity {i} has more than one type.")
                self.type_of[i] = t

    def _replacement(self, gci_name, column, original):
        orig = original.cpu().numpy()
        new = np.empty(len(orig), dtype=np.int64)
        labels = np.asarray([self.type_of.get(int(i), -1) for i in orig])
        for t in set(labels.tolist()):
            rows = np.nonzero(labels == t)[0]
            pool = self.type_pools[t] if t >= 0 else self.pool(gci_name, column)
            new[rows] = _draw(pool, len(rows), self.rng)
            if self.different and len(pool) > 1:
                for _ in range(self.max_redraw):
                    same = rows[new[rows] == orig[rows]]
                    if len(same) == 0:
                        break
                    new[same] = _draw(pool, len(same), self.rng)
        return th.as_tensor(new, dtype=th.long)


class KnownPositiveFilter(TrainingRegime):
    """Never use a known positive as a negative: a sampled negative that is itself an
    axiom of the training data is re-sampled in its corrupted column.

    ELEmbeddings (Kulmanov et al. 2019, Section 3.4) and EmEL++ (Mondal et al.,
    Section 3.3) sample negatives "such that neither :math:`C' \\sqsubseteq \\exists R.D`
    nor :math:`C \\sqsubseteq \\exists R.D'` are asserted axioms"; ELK-Em (Singh et al.,
    Section 4.2) filters "observed training positives" from corrupted class assertions;
    HiT (He et al. 2024, Section 2.3) requires negatives outside the asserted *and*
    transitively implied edges (use :class:`SiblingHardNegatives` or DELE's closure filter
    for the latter). Unlike the deductive-closure filter of DELE
    (:class:`mowl.training.dele.ClosureFilteredNegatives`), only *asserted* axioms are
    removed; entailed ones can still be sampled.

    A pure ``filter_negatives`` regime: it works with any sampler whose output follows the
    layout of this module (row ``i`` derived from positive ``i mod n``). The corrupted
    column of a row is the single column in which it differs from its positive (the column
    of :data:`FIXED_COLUMNS` if it equals its positive); rows that are still positives
    after ``max_resample`` rounds, or whose corrupted column cannot be identified, are
    dropped.

    :param pools: Optional ``{axiom type: {column: indices}}`` for re-sampling (default
        :func:`default_pool`); give the pools of the generator when it is restricted.
    :param max_resample: Re-sampling rounds.
    :param seed: Seed.
    """

    key = "known-positive-filtering"

    def __init__(self, pools=None, max_resample=10, seed=None):
        super().__init__()
        self.pools = {g: {int(c): np.asarray(list(p), dtype=np.int64)
                          for c, p in cols.items()} for g, cols in (pools or {}).items()}
        self.max_resample = int(max_resample)
        self.rng = np.random.default_rng(seed)
        self.num_filtered = 0
        self.num_dropped = 0
        self._known = {}

    def _known_set(self, gci_name, positives):
        if gci_name not in self._known:
            datasets = self.model.training_datasets
            rows = datasets[gci_name][:].tolist() if gci_name in datasets else []
            self._known[gci_name] = set(map(tuple, rows))
        # The positives of this step may include axioms added by another regime.
        base, extra = self._known[gci_name], set(map(tuple, positives.tolist()))
        return lambda row: row in base or row in extra

    def filter_negatives(self, gci_name, gci_dataset, negatives):
        if negatives is None or len(negatives) == 0:
            return negatives
        positives = gci_dataset[:]
        known = self._known_set(gci_name, positives)
        rows = negatives.tolist()
        bad = [i for i, r in enumerate(rows) if known(tuple(r))]
        if not bad:
            return negatives
        self.num_filtered += len(bad)
        cols = corrupted_columns(positives, negatives)
        negatives = negatives.clone()
        keep = th.ones(len(negatives), dtype=th.bool)
        pending = []
        for i in bad:
            diff = cols[i] if cols is not None else []
            if len(diff) == 1:
                pending.append((i, diff[0]))
            elif cols is not None and len(diff) == 0 and gci_name in FIXED_COLUMNS:
                # the corruption drew the original entity: use the default column
                pending.append((i, FIXED_COLUMNS[gci_name][0]))
            else:
                keep[i] = False
        for _ in range(self.max_resample):
            still = []
            for i, col in pending:
                pool = self.pools.get(gci_name, {}).get(col)
                if pool is None:
                    pool = default_pool(self.model, gci_name, col)
                negatives[i, col] = int(_draw(pool, 1, self.rng)[0])
                if known(tuple(negatives[i].tolist())):
                    still.append((i, col))
            pending = still
            if not pending:
                break
        for i, _ in pending:
            keep[i] = False
        self.num_dropped += int((~keep).sum())
        out = negatives[keep.to(negatives.device)]
        return out if len(out) > 0 else None


def _adjacency(rows):
    up, down = {}, {}
    for c, d in rows:
        if c == d:
            continue
        up.setdefault(c, set()).add(d)
        down.setdefault(d, set()).add(c)
    return up, down


def _reachable(start, edges, cache):
    if start in cache:
        return cache[start]
    seen, stack = set(), [start]
    while stack:
        x = stack.pop()
        for y in edges.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    cache[start] = seen
    return seen


class SiblingHardNegatives(TrainingRegime):
    """Hard negatives for :math:`C \\sqsubseteq D` from the siblings of :math:`C`.

    HiT (He et al., NeurIPS 2024, Section 2.3 and Section 4 "Multi-hop Inference"):
    a pair :math:`(e_1, e_2)` is a negative if it is neither an asserted nor a
    transitively implied subsumption, and a *hard* negative if, in addition, :math:`e_1`
    and :math:`e_2` are siblings (share a parent). For every positive
    :math:`C \\sqsubseteq D` of ``gci0``, ``num_negatives`` negatives
    :math:`C \\sqsubseteq S` are drawn with :math:`S` a sibling of :math:`C` in the
    asserted ``gci0`` hierarchy and :math:`C \\sqsubseteq S` not in its transitive closure;
    when :math:`C` has no such sibling, a uniformly drawn named class that is not a
    (transitive) superclass of :math:`C` is used, as HiT supplements missing siblings with
    random negatives. HiT trains with 10 negatives per positive.

    The hierarchy is the transitive closure of the *asserted* ``gci0`` axioms (HiT's
    :math:`E \\cup T`); entailments that need other axioms are not excluded.

    :param num_negatives: Negatives per positive.
    :param require_negative_loss: As for :class:`CorruptionNegatives`.
    :param seed: Seed.
    """

    key = "sibling-hard-negatives"

    def __init__(self, num_negatives=1, require_negative_loss=True, seed=None):
        super().__init__()
        if int(num_negatives) < 1:
            raise ValueError("num_negatives must be at least 1.")
        self.num_negatives = int(num_negatives)
        self.require_negative_loss = bool(require_negative_loss)
        self.rng = np.random.default_rng(seed)
        self._graph = None
        self.num_random_fallback = 0

    def attach(self, model):
        super().attach(model)
        self._graph = None
        capable = getattr(getattr(model, "module", None), "neg_capable_gcis", None)
        if self.require_negative_loss and capable is not None and "gci0" not in capable:
            raise TypeError(f"{type(model.module).__name__} has no negative loss for "
                            f"gci0; pass require_negative_loss=False together with a "
                            f"score-based loss regime.")

    def _build(self):
        rows = self.model.training_datasets["gci0"][:].tolist()
        named = set(self.model.named_class_index_dict.values())
        up, down = _adjacency([(c, d) for c, d in rows if c in named and d in named])
        self._graph = (up, down, {}, np.fromiter(named, dtype=np.int64))

    def negatives(self, gci_name, gci_dataset):
        if gci_name != "gci0":
            return NotImplemented
        if self._graph is None:
            self._build()
        up, down, closure_cache, named = self._graph
        data = gci_dataset[:]
        if len(data) == 0:
            return None
        blocks = []
        heads = data[:, 0].tolist()
        candidates = {}
        for c in set(heads):
            sups = _reachable(c, up, closure_cache)
            sibs = set()
            for p in up.get(c, ()):
                sibs |= down.get(p, set())
            sibs.discard(c)
            candidates[c] = (np.asarray(sorted(sibs - sups), dtype=np.int64), sups)
        for _ in range(self.num_negatives):
            new = np.empty(len(heads), dtype=np.int64)
            for i, c in enumerate(heads):
                sibs, sups = candidates[c]
                if len(sibs):
                    new[i] = sibs[self.rng.integers(0, len(sibs))]
                    continue
                self.num_random_fallback += 1
                for _ in range(20):
                    x = int(named[self.rng.integers(0, len(named))])
                    if x != c and x not in sups:
                        break
                new[i] = x
            block = data.clone()
            block[:, 1] = th.as_tensor(new, dtype=th.long, device=data.device)
            blocks.append(block)
        return th.cat(blocks, dim=0)


class ContradictionNegatives(TrainingRegime):
    """Negatives that are provably false: corruptions that contradict an asserted
    disjointness axiom.

    Mashkova and Hoehndorf (2026, Section 6.3, "type-aware negative sampling") draw a
    fraction :math:`p_\\bot` of the negatives from the contradictory axioms
    :math:`S_\\bot` "via cheap reasoning shortcuts (asserted disjointness, domain/range
    violations)" and the rest by uniform corruption. Here the shortcut is asserted
    disjointness: for a positive :math:`C \\sqsubseteq D` (``gci0``) or :math:`C(a)`
    (``class_assertion``, columns ``[a, C]``), the corrupted class :math:`D'` is chosen
    among the classes that are disjoint with a told superclass of :math:`C` (resp. of a
    told type of :math:`a`): if :math:`A \\sqcap B \\sqsubseteq \\bot` is asserted, the
    hierarchy entails :math:`C \\sqsubseteq A` and :math:`D' \\sqsubseteq B`, then
    :math:`C \\sqsubseteq D'` entails :math:`C \\sqsubseteq \\bot`. Told sub- and
    superclasses are the transitive closure of the asserted ``gci0`` axioms. If no
    contradictory replacement exists, or with probability :math:`1 - p_\\bot`, the
    corrupted class is drawn uniformly from the named classes.

    By construction every negative drawn from :math:`S_\\bot` is contradictory with the
    ontology unless :math:`C` (resp. :math:`a`) is unsatisfiable. Domain and range
    violations, and the other normal forms evaluated in the paper (GCI2, GCI1-bottom,
    GCI3), are not implemented. Counters :attr:`num_contradictory` and
    :attr:`num_uniform` record how many negatives came from each source.

    :param p_bot: Fraction :math:`p_\\bot` of contradictory negatives (the paper sweeps
        it from 0 to 1; its endpoint contrast is between 1 and 0).
    :param num_negatives: Negatives per positive.
    :param normal_forms: Subset of ``("gci0", "class_assertion")``.
    :param require_negative_loss: As for :class:`CorruptionNegatives`.
    :param seed: Seed.
    """

    key = "contradiction-negatives"
    _COLUMN = {"gci0": (0, 1), "class_assertion": (0, 1)}
    BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"

    def __init__(self, p_bot=1.0, num_negatives=1, normal_forms=("gci0",),
                 require_negative_loss=True, seed=None):
        super().__init__()
        if not 0.0 <= float(p_bot) <= 1.0:
            raise ValueError("p_bot must be in [0, 1].")
        for nf in normal_forms:
            if nf not in self._COLUMN:
                raise ValueError(f"Unsupported normal form {nf!r}.")
        self.p_bot = float(p_bot)
        self.num_negatives = int(num_negatives)
        self.normal_forms = tuple(normal_forms)
        self.require_negative_loss = bool(require_negative_loss)
        self.rng = np.random.default_rng(seed)
        self._index = None
        self.num_contradictory = 0
        self.num_uniform = 0

    def attach(self, model):
        super().attach(model)
        self._index = None
        capable = getattr(getattr(model, "module", None), "neg_capable_gcis", None)
        if self.require_negative_loss and capable is not None:
            missing = [nf for nf in self.normal_forms if nf not in capable]
            if missing:
                raise TypeError(f"{type(model.module).__name__} has no negative loss for "
                                f"{missing}; pass require_negative_loss=False together "
                                f"with a score-based loss regime.")

    def _disjoint_pairs(self):
        ds = self.model.training_datasets
        bot = self.model.class_index_dict.get(self.BOTTOM)
        pairs = []
        for name in ("gci1_bot", "gci1"):
            if name in ds and len(ds[name]):
                rows = ds[name][:]
                if name == "gci1" and bot is not None:
                    rows = rows[rows[:, 2] == bot]
                elif name == "gci1":
                    continue
                pairs.extend((int(a), int(b)) for a, b in rows[:, :2].tolist())
        return pairs

    def _build(self):
        ds = self.model.training_datasets
        rows = ds["gci0"][:].tolist() if "gci0" in ds else []
        bot = self.model.class_index_dict.get(self.BOTTOM)
        up, down = _adjacency([(c, d) for c, d in rows if d != bot])
        disjoint = {}
        for a, b in self._disjoint_pairs():
            disjoint.setdefault(a, set()).add(b)
            disjoint.setdefault(b, set()).add(a)
        named = set(self.model.named_class_index_dict.values())
        types = {}
        if "class_assertion" in ds and len(ds["class_assertion"]):
            for a, c in ds["class_assertion"][:].tolist():
                types.setdefault(a, set()).add(c)
        self._index = {"up": up, "down": down, "disjoint": disjoint, "named": named,
                       "named_arr": np.fromiter(named, dtype=np.int64), "types": types,
                       "up_cache": {}, "down_cache": {}, "contra_cache": {}}

    def contradictory_classes(self, classes):
        """Named classes :math:`D'` with :math:`C \\sqcap D' \\sqsubseteq \\bot` entailed
        through asserted disjointness, for :math:`C` the conjunction of ``classes``."""
        if self._index is None:
            self._build()
        ix = self._index
        key = tuple(sorted(classes))
        if key in ix["contra_cache"]:
            return ix["contra_cache"][key]
        sups = set(classes)
        for c in classes:
            sups |= _reachable(c, ix["up"], ix["up_cache"])
        out = set()
        for a in sups:
            for b in ix["disjoint"].get(a, ()):
                out.add(b)
                out |= _reachable(b, ix["down"], ix["down_cache"])
        arr = np.asarray(sorted(out & ix["named"]), dtype=np.int64)
        ix["contra_cache"][key] = arr
        return arr

    def negatives(self, gci_name, gci_dataset):
        if gci_name not in self.normal_forms:
            return NotImplemented
        if self._index is None:
            self._build()
        data = gci_dataset[:]
        if len(data) == 0:
            return None
        head_col, tail_col = self._COLUMN[gci_name]
        named = self._index["named_arr"]
        blocks = []
        for _ in range(self.num_negatives):
            new = np.empty(len(data), dtype=np.int64)
            for i, h in enumerate(data[:, head_col].tolist()):
                if gci_name == "gci0":
                    base = (h,)
                else:
                    base = tuple(self._index["types"].get(h, ()))
                pool = self.contradictory_classes(base) if base else named[:0]
                if len(pool) and self.rng.random() < self.p_bot:
                    new[i] = pool[self.rng.integers(0, len(pool))]
                    self.num_contradictory += 1
                else:
                    new[i] = named[self.rng.integers(0, len(named))]
                    self.num_uniform += 1
            block = data.clone()
            block[:, tail_col] = th.as_tensor(new, dtype=th.long, device=data.device)
            blocks.append(block)
        return th.cat(blocks, dim=0)
