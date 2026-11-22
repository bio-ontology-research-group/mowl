import logging

import numpy as np
import torch as th
from tqdm import trange

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.baselines.nn.el.elkem.losses import pearson_contrastive_loss
from mowl.baselines.nn.el.elkem.module import ELKEmModule

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

TOP_IRI = "http://www.w3.org/2002/07/owl#Thing"
BOT_IRI = "http://www.w3.org/2002/07/owl#Nothing"

ASSERTION_FORMS = ("class_assertion", "object_property_assertion")


class ELKEm(EmbeddingELModel):
    r"""ELK-Em: closure-aware box embeddings for :math:`\mathcal{EL}^+_\bot` ontologies
    [singh2026]_.

    Concepts are axis-aligned boxes, individuals are points, and a role :math:`r` is a
    triple :math:`\langle R_r, E_r, T_r\rangle` (range box, error box, diagonal affine map
    with positive slope) with :math:`r^{\mathcal{E}} = \{(x, y) \mid x \in T_r(y) \oplus E_r,
    y \in R_r\}`, so that :math:`(\exists r.C)^{\mathcal{E}} = T_r(C^{\mathcal{E}} \cap R_r)
    \oplus E_r` is again a box. See :class:`~mowl.nn.el.elkem.module.ELKEmModule` for the
    per-axiom losses. If every axiom has zero loss, the embedding is a model of the
    ontology (Theorem 6 of the paper) and satisfies everything ELK's inference rules derive
    from it (Corollary 7).

    .. rubric:: Training objective

    Each epoch draws one mini-batch of at most ``batch_size`` axioms from every normal form
    (as in the authors' code, one optimisation step per epoch) and minimises

    * the mean per-axiom loss of every normal form (Table 2 of the paper);
    * when training on the TBox alone, ``contrastive_factor`` times the Pearson correlation
      loss (Eq. 5) between the margin-free scores :math:`\exp(-\ell^0)` of the positives
      and of ``k_neg`` corruptions per positive, computed separately for ``gci0``
      (corrupting :math:`D`) and ``gci2`` (corrupting :math:`C` or :math:`D` with equal
      probability) and summed. Corruptions are not filtered against entailed axioms;
    * with assertions: ``abox_pos_weight`` times the assertion losses and
      ``abox_neg_weight`` times :math:`\mathrm{ReLU}(-\log\tau - \ell)` of ``abox_k_neg``
      corruptions per assertion (the class, resp. the object, is replaced; observed
      training assertions are filtered out);
    * optionally, ``lexical_weight`` times the lexical regulariser (Eq. 6) on a random batch
      of ``lexical_batch_size`` individuals with user-supplied external representations;
    * ``reg_factor`` times a box-size penalty taken from the authors' code.

    Adam is used with the learning rate multiplied by ``lr_decay_factor`` after
    ``lr_decay_epoch`` epochs. When the dataset has a validation ontology, the model is
    validated every ``validate_every`` epochs by the median rank of the validation
    subsumptions :math:`C \sqsubseteq D` (``gci0``) among all named classes (unfiltered,
    ties at their average rank, the query and :math:`\top` excluded), and the checkpoint
    with the lowest median rank is saved to ``model_filepath``. Without a validation
    ontology the final state is saved.

    .. rubric:: :math:`\top`, :math:`\bot` and unsatisfiable classes

    ``owl:Thing`` is fixed to the full box and ``owl:Nothing`` to the empty box. Classes
    :math:`A` with :math:`A \sqsubseteq \bot` (``gci0_bot``) are fixed to the empty box
    (Table 2: "fixed :math:`B_C = [\infty, -\infty]`"), and this is propagated backwards
    through ``gci0`` (:math:`A \sqsubseteq B`) and ``gci2`` (:math:`A \sqsubseteq \exists
    r.B`) axioms whose right-hand concept is fixed empty, as the authors' code does for
    ``gci0``. ``gci1``/``gci3`` axioms whose superclass is fixed empty are trained as
    ``gci1_bot``/``gci3_bot``, and ``gci0`` axioms with a fixed-empty subclass are dropped
    (they hold trivially).

    .. rubric:: Published configuration

    For the subsumption-ranking benchmark (Table 3) the paper uses ``embed_dim=50``, Adam,
    ``learning_rate=0.01`` decayed by 0.1 after 5,000 epochs, ``margin=0.1``,
    ``batch_size=1024``, 10,000 epochs, ``k_neg=5`` for GO and ``k_neg=3`` for GALEN and
    Anatomy, and checkpoint selection by validation median rank every 100 epochs
    (``validate_every=100``). ``contrastive_factor=3.0`` and ``reg_factor=1.0`` are the
    defaults of the authors' training script; the paper does not state them. For protein
    function prediction (Table 4) it uses ``embed_dim=100``, 15,000 epochs, ``margin=0.05``,
    ``abox_k_neg=2``, ``abox_threshold=0.4``, ``lexical_batch_size=2000``,
    ``lexical_decay=1.0`` and weights 2.0 / 0.5 / 10.0 for the positive, negative and
    lexical terms, with a cosine learning-rate decay that is not implemented here.

    .. rubric:: Differences from the paper and the authors' code

    The built-in loop below follows the paper. The design of the authors' code is available
    as ``variant="reference"`` (geometry) plus the training regime
    ``elkem-code-training`` (:class:`~mowl.training.reference.ELKEmCodeTraining`).

    * The paper sums the per-axiom losses; the authors' code squares them before averaging.
      This implementation follows the paper (mean of the unsquared losses per normal form).
    * The authors' code adds an overlap term :math:`\ell_{\cap+}(R_r, B_D)` to the ``gci2``
      loss; Table 2 of the paper does not, and neither does this implementation.
    * The box of :math:`\exists r.C` is kept empty when :math:`C \cap R_r` is empty (see
      :class:`~mowl.nn.el.elkem.module.ELKEmModule`); this keeps the ``gci2`` loss sound
      without the code's overlap term.
    * The box-size penalty (``reg_factor``) is not described in the paper.
    * The Pearson term is computed per normal form and summed, as in the authors' code (the
      paper's Eq. 5 does not say whether positives of different forms share one
      correlation).

    :param dataset: The dataset.
    :type dataset: :class:`mowl.datasets.Dataset`
    :param embed_dim: Embedding dimension :math:`n`. Defaults to 50.
    :param margin: Margin :math:`\gamma`. Defaults to 0.1.
    :param learning_rate: Adam learning rate. Defaults to 0.01.
    :param batch_size: Axioms per normal form and epoch. Defaults to 1024.
    :param k_neg: Corruptions per positive ``gci0``/``gci2`` axiom for the contrastive
        term. Defaults to 3.
    :param contrastive_factor: Weight of the Pearson contrastive term. Defaults to 3.0.
    :param use_contrastive: Whether to use the contrastive term. ``None`` (default) uses it
        iff the training ontology has no assertions, as in the paper.
    :param reg_factor: Weight of the box-size penalty. Defaults to 1.0.
    :param lr_decay_epoch: Epoch after which the learning rate is decayed, or ``None``.
        Defaults to 5000.
    :param lr_decay_factor: Learning-rate decay factor. Defaults to 0.1.
    :param abox_threshold: Target-score threshold :math:`\tau` for corrupted assertions.
        Defaults to 0.4.
    :param abox_k_neg: Corruptions per assertion. Defaults to 2.
    :param abox_pos_weight: Weight of the positive assertion losses. Defaults to 1.0.
    :param abox_neg_weight: Weight of the corrupted assertion losses. Defaults to 1.0.
    :param lexical_features: External representations of individuals, either a mapping
        from individual IRI to vector or an array of shape ``(num_individuals, d)`` in the
        order of :attr:`individual_index_dict`. ``None`` disables lexical regularisation.
    :param lexical_weight: Weight of the lexical regulariser. Defaults to 1.0.
    :param lexical_decay: Decay rate :math:`\lambda` of the geometric similarity.
        Defaults to 1.0.
    :param lexical_batch_size: Individuals per lexical batch. Defaults to 2000.
    :param model_filepath: Where the selected checkpoint is saved.
    :param device: Torch device. Defaults to ``"cpu"``.
    :param load_role_axioms: Train on role inclusions and binary role chains. Defaults to
        ``False``.
    :param variant: Geometry of the module: ``"paper"`` (default; Table 2 of the paper) or
        ``"reference"`` (the authors' ranking code: interval-arithmetic existential boxes and
        an overlap term in the ``gci2`` loss); see
        :class:`~mowl.nn.el.elkem.module.ELKEmModule`.
    """

    def __init__(self, dataset, embed_dim=50, margin=0.1, learning_rate=0.01,
                 batch_size=1024, k_neg=3, contrastive_factor=3.0, use_contrastive=None,
                 reg_factor=1.0, lr_decay_epoch=5000, lr_decay_factor=0.1,
                 abox_threshold=0.4, abox_k_neg=2, abox_pos_weight=1.0, abox_neg_weight=1.0,
                 lexical_features=None, lexical_weight=1.0, lexical_decay=1.0,
                 lexical_batch_size=2000, model_filepath=None, device="cpu",
                 load_role_axioms=False, variant="paper"):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, load_role_axioms=load_role_axioms)
        if k_neg < 1:
            raise ValueError("k_neg must be at least 1.")
        if not 0 < abox_threshold < 1:
            raise ValueError("abox_threshold must lie in (0, 1).")
        self.margin = margin
        self.k_neg = k_neg
        self.contrastive_factor = contrastive_factor
        self.use_contrastive = use_contrastive
        self.reg_factor = reg_factor
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor
        self.abox_threshold = abox_threshold
        self.abox_k_neg = abox_k_neg
        self.abox_pos_weight = abox_pos_weight
        self.abox_neg_weight = abox_neg_weight
        self.lexical_weight = lexical_weight
        self.lexical_decay = lexical_decay
        self.lexical_batch_size = lexical_batch_size
        self.best_validation_median_rank = None
        self.variant = variant

        self._bot_ids, self._top_ids = self._fixed_classes()
        self._tensors = None
        self._lexical_idx, self._lexical_feats = self._prepare_lexical(lexical_features)
        self.init_module()

    def init_module(self):
        self.module = ELKEmModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            margin=self.margin,
            bot_ids=self._bot_ids,
            top_ids=self._top_ids,
            abox_threshold=self.abox_threshold,
            variant=self.variant,
        ).to(self.device)

    # ------------------------------------------------------------------ data

    def _fixed_classes(self):
        """Indices of the classes fixed to the empty box and to the full box."""
        cid = self.class_index_dict
        top = [cid[TOP_IRI]] if TOP_IRI in cid else []
        bot = {cid[BOT_IRI]} if BOT_IRI in cid else set()
        datasets = self.training_datasets

        def rows(name):
            if name not in datasets or len(datasets[name]) == 0:
                return []
            return datasets[name][:].tolist()

        bot |= {row[0] for row in rows("gci0_bot")}
        # sub-concepts of a concept fixed empty: A ⊑ B and A ⊑ ∃r.B with B empty
        subs = {}
        for a, b in rows("gci0"):
            subs.setdefault(b, []).append(a)
        for a, _, b in rows("gci2"):
            subs.setdefault(b, []).append(a)
        stack = list(bot)
        while stack:
            for a in subs.get(stack.pop(), ()):
                if a not in bot:
                    bot.add(a)
                    stack.append(a)
        if set(top) & bot:
            raise ValueError("The ontology entails owl:Thing ⊑ owl:Nothing (inconsistent).")
        return sorted(bot), top

    def _training_tensors(self):
        """Training axioms per normal form, with the rows involving fixed-empty concepts
        rerouted (see the class docstring)."""
        if self._tensors is not None:
            return self._tensors
        tensors = {k: v[:].to(self.device) for k, v in self.training_datasets.items()
                   if len(v) > 0}
        bot = th.tensor(self._bot_ids, dtype=th.long, device=self.device)
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
        self._tensors = {k: v for k, v in tensors.items() if len(v) > 0}
        return self._tensors

    def _prepare_lexical(self, features):
        if features is None:
            return None, None
        ind_dict = self.individual_index_dict
        if isinstance(features, dict):
            idx = [ind_dict[iri] for iri in features if iri in ind_dict]
            feats = [np.asarray(features[iri], dtype=np.float32) for iri in features
                     if iri in ind_dict]
            if not idx:
                raise ValueError("No key of lexical_features is an individual of the dataset.")
            feats = np.stack(feats)
        else:
            feats = np.asarray(features, dtype=np.float32)
            if feats.ndim != 2 or feats.shape[0] != len(ind_dict):
                raise ValueError("lexical_features must have one row per individual "
                                 f"({len(ind_dict)}), got shape {feats.shape}.")
            idx = list(range(len(ind_dict)))
        return (th.tensor(idx, dtype=th.long, device=self.device),
                th.tensor(feats, device=self.device))

    # ------------------------------------------------------------------ negatives

    def _class_pool(self):
        excluded = set(self._top_ids) | set(self._bot_ids)
        pool = [i for i in self.named_class_index_dict.values() if i not in excluded]
        return th.tensor(pool, dtype=th.long, device=self.device)

    @staticmethod
    def _resample(pool, n, forbidden, rounds=20):
        """Draw ``n`` entries of ``pool``, redrawing those equal to a forbidden value."""
        sample = pool[th.randint(len(pool), (n,), device=pool.device)]
        for _ in range(rounds):
            clash = th.zeros(n, dtype=th.bool, device=pool.device)
            for values in forbidden:
                clash |= sample == values
            if not clash.any():
                break
            sample[clash] = pool[th.randint(len(pool), (int(clash.sum()),),
                                            device=pool.device)]
        return sample

    def generate_negatives(self, gci_name, gci_dataset):
        r"""Corrupted axioms for ``gci_name``.

        * ``gci0`` :math:`C \sqsubseteq D`: ``k_neg`` corruptions per row, :math:`D`
          replaced by a random named class (not :math:`C`, :math:`D`, :math:`\top`,
          :math:`\bot`).
        * ``gci2`` :math:`C \sqsubseteq \exists r.D`: ``k_neg`` corruptions per row, each
          replacing :math:`C` or :math:`D` with equal probability.
        * ``class_assertion``: ``abox_k_neg`` corruptions per row, replacing the class;
          observed training assertions are filtered out.
        * ``object_property_assertion``: ``abox_k_neg`` corruptions per row, replacing the
          object individual; observed training assertions are filtered out.

        Rows of the result are grouped by positive (``repeat_interleave`` order).
        """
        data = gci_dataset[:].to(self.device)
        if gci_name == "gci0":
            pos = data.repeat_interleave(self.k_neg, dim=0)
            neg = pos.clone()
            neg[:, 1] = self._resample(self._class_pool(), len(pos), (pos[:, 0], pos[:, 1]))
            return neg
        if gci_name == "gci2":
            pos = data.repeat_interleave(self.k_neg, dim=0)
            corrupt_d = th.rand(len(pos), device=self.device) < 0.5
            avoid = th.where(corrupt_d, pos[:, 2], pos[:, 0])
            new = self._resample(self._class_pool(), len(pos), (avoid,))
            neg = pos.clone()
            neg[:, 0] = th.where(corrupt_d, pos[:, 0], new)
            neg[:, 2] = th.where(corrupt_d, new, pos[:, 2])
            return neg
        if gci_name == "class_assertion":
            return self._corrupt_assertions(data, column=1, pool=self._class_pool(),
                                            gci_name=gci_name)
        if gci_name == "object_property_assertion":
            pool = th.tensor(list(self.individual_index_dict.values()), dtype=th.long,
                             device=self.device)
            return self._corrupt_assertions(data, column=2, pool=pool, gci_name=gci_name)
        return super().generate_negatives(gci_name, gci_dataset)

    def _corrupt_assertions(self, data, column, pool, gci_name, rounds=20):
        pos = data.repeat_interleave(self.abox_k_neg, dim=0)
        neg = pos.clone()
        observed = self._training_tensors().get(gci_name)
        width = int(max(pos.max().item(), observed.max().item()
                        if observed is not None else 0)) + 1

        def encode(rows):
            code = th.zeros(len(rows), dtype=th.long, device=rows.device)
            for j in range(rows.shape[1]):
                code = code * width + rows[:, j]
            return code

        known = encode(observed) if observed is not None else None
        neg[:, column] = self._resample(pool, len(neg), (pos[:, column],))
        for _ in range(rounds):
            if known is None:
                break
            clash = th.isin(encode(neg), known)
            if not clash.any():
                break
            neg[clash, column] = pool[th.randint(len(pool), (int(clash.sum()),),
                                                 device=self.device)]
        return neg

    # ------------------------------------------------------------------ hooks

    def get_optimizer(self):
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def get_scheduler(self, optimizer):
        """Step decay of the learning rate after ``lr_decay_epoch`` epochs."""
        milestones = [] if self.lr_decay_epoch is None else [self.lr_decay_epoch]
        return th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                 gamma=self.lr_decay_factor)

    def get_regularization_loss(self):
        if not self.reg_factor:
            return 0
        return self.reg_factor * self.module.size_regularization_loss()

    def lexical_regularization_loss(self):
        """Lexical regulariser (Eq. 6) on a random batch of individuals with external
        representations, or ``None`` if no representations were given."""
        if self._lexical_idx is None:
            return None
        n = len(self._lexical_idx)
        pick = th.randperm(n, device=self.device)[:self.lexical_batch_size]
        return self.module.lexical_loss(self._lexical_idx[pick], self._lexical_feats[pick],
                                        self.lexical_decay)

    def additional_training_loss(self, epoch):
        lex = self.lexical_regularization_loss()
        if lex is None or not self.lexical_weight:
            return None
        return self.lexical_weight * lex

    def contrastive_active(self):
        """Whether the Pearson contrastive term is used (see ``use_contrastive``)."""
        if self.use_contrastive is not None:
            return bool(self.use_contrastive)
        return not any(form in self._training_tensors() for form in ASSERTION_FORMS)

    def contrastive_loss(self, batches):
        """Sum over ``gci0`` and ``gci2`` of the Pearson loss (Eq. 5) of the batch."""
        loss = None
        for form in ("gci0", "gci2"):
            pos = batches.get(form)
            if pos is None or len(pos) == 0:
                continue
            neg = self.generate_negatives(form, pos)
            term = pearson_contrastive_loss(self.module.score(pos, form),
                                            self.module.score(neg, form))
            loss = term if loss is None else loss + term
        return loss

    def _sample_batches(self):
        batches = {}
        for name, data in self._training_tensors().items():
            if len(data) > self.batch_size:
                data = data[th.randperm(len(data), device=self.device)[:self.batch_size]]
            batches[name] = data
        return batches

    def training_loss(self, batches=None):
        """Total loss of one mini-batch per normal form (without the regulariser)."""
        if batches is None:
            batches = self._sample_batches()
        loss = th.tensor(0.0, device=self.device)
        for name, data in batches.items():
            pos = self.module(data, name)
            if name in ASSERTION_FORMS:
                neg = self.module(self.generate_negatives(name, data), name, neg=True)
                loss = loss + self.abox_pos_weight * pos.mean() + \
                    self.abox_neg_weight * neg.mean()
            else:
                loss = loss + pos.mean()
        if self.contrastive_active() and self.contrastive_factor:
            contrast = self.contrastive_loss(batches)
            if contrast is not None:
                loss = loss + self.contrastive_factor * contrast
        return loss

    # ------------------------------------------------------------------ training

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train the model (see the class docstring for the objective).

        With a :attr:`~mowl.base_models.elmodel.EmbeddingELModel.training_regime` attached
        (for example the regime 'elkem-code-training' of
        :class:`mowl.training.reference.ELKEmCodeTraining`), the *generic* training loop of
        :meth:`mowl.base_models.elmodel.EmbeddingELModel.train` runs instead of the
        built-in one, and every hook (batches, negatives, loss, schedule, selection) is
        the regime's. The built-in loop below is the model's own design and is used
        without a regime.

        :param epochs: Number of epochs (one mini-batch per normal form and epoch).
        :param validate_every: Validate every N epochs. Defaults to 1; the paper uses 100.
        :param epoch_callback: Optional ``epoch_callback(epoch, model)`` called after each
            epoch.
        """
        if self.training_regime is not None:
            return super().train(epochs, validate_every=validate_every,
                                 epoch_callback=epoch_callback)
        tensors = self._training_tensors()
        logger.info("Training axioms: " + ", ".join(f"{k}: {len(v)}"
                                                    for k, v in tensors.items()))
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        has_validation = self._has_validation_subsumptions()
        best = float("inf")

        for epoch in trange(epochs):
            self.module.train()
            self.on_epoch_start(epoch, epochs)
            loss = self.training_loss()
            extra = self.additional_training_loss(epoch)
            if extra is not None:
                loss = loss + extra
            loss = loss + self.step_regularization_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.after_optimizer_step()
            scheduler.step()

            if epoch_callback is not None:
                epoch_callback(epoch, self)

            if has_validation and (epoch + 1) % validate_every == 0:
                med = self.validation_median_rank()
                if med < best:
                    best = med
                    self.best_validation_median_rank = med
                    th.save(self.module.state_dict(), self.model_filepath)
                logger.debug(f"Epoch {epoch + 1}: train loss {loss.item():.4f}, "
                             f"validation median rank {med}")

        if not has_validation or best == float("inf"):
            th.save(self.module.state_dict(), self.model_filepath)

    def _has_validation_subsumptions(self):
        if self.dataset.validation is None:
            return False
        data = self.validation_datasets.get("gci0")
        return data is not None and len(data) > 0

    def _rank_subsumptions(self, pairs, chunk_budget=2_000_000):
        """Average-tie ranks of the superclasses of ``pairs`` (rows ``[C, D]``) among all
        named classes except :math:`\\top` and the query, by ascending :math:`\\ell^0`."""
        top = set(self._top_ids)
        candidates = th.tensor([i for i in self.named_class_index_dict.values()
                                if i not in top], dtype=th.long, device=self.device)
        cand_set = set(candidates.tolist())
        pairs = pairs[[d in cand_set for d in pairs[:, 1].tolist()]]
        n = len(candidates)
        step = max(1, chunk_budget // max(n, 1))
        ranks = []
        was_training = self.module.training
        self.module.eval()
        with th.no_grad():
            for start in range(0, len(pairs), step):
                chunk = pairs[start:start + step].to(self.device)
                q = len(chunk)
                grid = th.stack([chunk[:, 0].repeat_interleave(n), candidates.repeat(q)], dim=1)
                losses = self.module.zero_margin_loss(grid, "gci0").view(q, n)
                target = self.module.zero_margin_loss(chunk, "gci0").unsqueeze(1)
                others = (candidates.unsqueeze(0) != chunk[:, :1]) & \
                    (candidates.unsqueeze(0) != chunk[:, 1:2])
                below = ((losses < target) & others).sum(dim=1)
                ties = ((losses == target) & others).sum(dim=1)
                ranks.append(1 + below.double() + ties.double() / 2)
        self.module.train(was_training)
        return th.cat(ranks) if ranks else th.empty(0, dtype=th.double)

    def validation_median_rank(self):
        """Median rank of the validation subsumptions (unfiltered, average-tie ranks)."""
        ranks = self._rank_subsumptions(self.validation_datasets["gci0"][:])
        return float(ranks.median()) if len(ranks) else float("inf")

    # ------------------------------------------------------------------ scoring

    def score(self, axiom):
        r"""Score :math:`s(\alpha) = \exp(-\ell^0(\alpha)) \in (0, 1]` (Eq. 4) of an
        :math:`\mathcal{EL}` axiom in normal form. Higher is better."""
        was_training = self.module.training
        self.module.eval()
        with th.no_grad():
            loss = super().score(axiom)
        self.module.train(was_training)
        return th.exp(-loss)

    @property
    def class_embeddings(self):
        """Class boxes as ``{iri: concat(lower, upper)}``."""
        idx = th.arange(len(self.class_index_dict), device=self.device)
        with th.no_grad():
            lower, upper = self.module.class_box(idx)
        boxes = th.cat([lower, upper], dim=1).cpu().numpy()
        return dict(zip(self.class_index_dict.keys(), boxes))

    @property
    def object_property_embeddings(self):
        """Roles as ``{iri: concat(range_l, range_u, error_l, error_u, slope, shift)}``."""
        idx = th.arange(len(self.object_property_index_dict), device=self.device)
        with th.no_grad():
            parts = th.cat(self.module.role(idx), dim=1).cpu().numpy()
        return dict(zip(self.object_property_index_dict.keys(), parts))

    @property
    def individual_embeddings(self):
        if self.module.ind_embedding is None:
            return dict()
        points = self.module.ind_embedding.weight.detach().cpu().numpy()
        return dict(zip(self.individual_index_dict.keys(), points))

    def get_embeddings(self):
        """Embeddings of the selected checkpoint: ``(classes, roles, individuals)``."""
        self.load_best_model()
        return self.class_embeddings, self.object_property_embeddings, \
            self.individual_embeddings or None
