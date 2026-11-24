import logging

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.elkem.module import ELKEmModule
from mowl.evaluation.region_access import concatenated_rows
from mowl.evaluation.scoring_rules import exp_negative_score
from mowl.training import elkem as T

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

TOP_IRI = "http://www.w3.org/2002/07/owl#Thing"
BOT_IRI = "http://www.w3.org/2002/07/owl#Nothing"

ASSERTION_FORMS = T.ASSERTION_FORMS


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

    The objective below is the default training regime of the method
    (:class:`~mowl.training.elkem.ELKEmTraining`, the regimes that ``model.training_regime``
    holds: mini-batches, negatives of the assertions, the correlation term, the lexical
    regulariser, the learning-rate decay and the model selection); any of them can be replaced
    by assigning another regime. Each epoch draws one mini-batch of at most ``batch_size`` axioms from every normal form
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

    The default regimes follow the paper. The design of the authors' code is available
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

    _supports_axiom_updates = False

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
        self._lexical_features = lexical_features
        # validates the features now
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
            size_weight=self.reg_factor,
        ).to(self.device)
        # the training regimes of the method (the options are the constructor's)
        self.module.default_regimes = (self._default_training_regime,)
        self._attach_default_regimes()

    def _training_options(self):
        """The options of the default regime :class:`~mowl.training.elkem.ELKEmTraining`."""
        return dict(batch_size=self.batch_size, k_neg=self.k_neg,
                    contrastive_factor=self.contrastive_factor,
                    use_contrastive=self.use_contrastive, abox_k_neg=self.abox_k_neg,
                    abox_pos_weight=self.abox_pos_weight, abox_neg_weight=self.abox_neg_weight,
                    lexical_features=self._lexical_features, lexical_weight=self.lexical_weight,
                    lexical_decay=self.lexical_decay,
                    lexical_batch_size=self.lexical_batch_size,
                    lr_decay_epoch=self.lr_decay_epoch, lr_decay_factor=self.lr_decay_factor)

    def _default_training_regime(self):
        return T.ELKEmTraining(**self._training_options())

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
        rerouted (:class:`~mowl.training.elkem.FixedConceptRouting`)."""
        if self._tensors is None:
            self._tensors = T.route_fixed_empty(self.training_datasets, self._bot_ids,
                                                self.device)
        return self._tensors

    def _prepare_lexical(self, features):
        return T.prepare_lexical(features, self.individual_index_dict, self.device)

    # ------------------------------------------------------------------ negatives

    def _class_pool(self):
        excluded = set(self._top_ids) | set(self._bot_ids)
        pool = [i for i in self.named_class_index_dict.values() if i not in excluded]
        return th.tensor(pool, dtype=th.long, device=self.device)

    def generate_negatives(self, gci_name, gci_dataset):
        r"""Corrupted axioms for ``gci_name`` (the sampling of the regimes
        :class:`~mowl.training.elkem.PearsonContrastiveTerm` and
        :class:`~mowl.training.elkem.AssertionCorruption`).

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
            return T.corrupt_columns(data, self.k_neg, (1,), self._class_pool(), (0, 1))
        if gci_name == "gci2":
            return T.corrupt_columns(data, self.k_neg, (0, 2), self._class_pool())
        if gci_name == "class_assertion":
            return self._corrupt_assertions(data, column=1, pool=self._class_pool(),
                                            gci_name=gci_name)
        if gci_name == "object_property_assertion":
            pool = th.tensor(list(self.individual_index_dict.values()), dtype=th.long,
                             device=self.device)
            return self._corrupt_assertions(data, column=2, pool=pool, gci_name=gci_name)
        return super().generate_negatives(gci_name, gci_dataset)

    def _corrupt_assertions(self, data, column, pool, gci_name, rounds=20):
        return T.corrupt_assertions(data, self._training_tensors().get(gci_name), column, pool,
                                    self.abox_k_neg, rounds)

    # ------------------------------------------------------------------ training

    def get_optimizer(self):
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def lexical_regularization_loss(self):
        """Lexical regulariser (Eq. 6) on a random batch of individuals with external
        representations, or ``None`` if no representations were given
        (:class:`~mowl.training.elkem.LexicalRegularisation`)."""
        regime = T.LexicalRegularisation(self._lexical_features, self.lexical_weight,
                                         self.lexical_decay, self.lexical_batch_size)
        regime.model = self
        regime.indices, regime.matrix = self._lexical_idx, self._lexical_feats
        return regime.term()

    def contrastive_active(self):
        """Whether the Pearson contrastive term is used (see ``use_contrastive``)."""
        if self.use_contrastive is not None:
            return bool(self.use_contrastive)
        return not any(form in self._training_tensors() for form in ASSERTION_FORMS)

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train the model with the generic loop and the regimes of the method (see the class
        docstring); a model whose regime was removed (``training_regime = None``) gets the
        default regimes again.

        :param epochs: Number of epochs (one mini-batch per normal form and epoch).
        :param validate_every: Validate every N epochs. Defaults to 1; the paper uses 100.
        :param epoch_callback: Optional ``epoch_callback(epoch, model)`` called after each
            epoch.
        """
        if self.training_regime is None:
            self._attach_default_regimes()
        return super().train(epochs, validate_every=validate_every,
                             epoch_callback=epoch_callback)

    def _has_validation_subsumptions(self):
        if self.dataset.validation is None:
            return False
        data = self.validation_datasets.get("gci0")
        return data is not None and len(data) > 0

    def _rank_subsumptions(self, pairs, chunk_budget=2_000_000):
        """Average-tie ranks of the superclasses of ``pairs`` (rows ``[C, D]``) among all
        named classes except :math:`\\top` and the query, by ascending :math:`\\ell^0`."""
        return T._rank_subsumptions(self, pairs, chunk_budget)

    def validation_median_rank(self):
        """Median rank of the validation subsumptions (unfiltered, average-tie ranks)."""
        return T.subsumption_median_rank(self)

    # ------------------------------------------------------------------ scoring

    def score(self, axiom):
        r"""Score :math:`s(\alpha) = \exp(-\ell^0(\alpha)) \in (0, 1]` (Eq. 4) of an
        :math:`\mathcal{EL}` axiom in normal form. Higher is better."""
        was_training = self.module.training
        self.module.eval()
        with th.no_grad():
            loss = super().score(axiom)
        self.module.train(was_training)
        return exp_negative_score(loss)

    @property
    def class_embeddings(self):
        """Class boxes as ``{iri: concat(lower, upper)}``."""
        idx = th.arange(len(self.class_index_dict), device=self.device)
        with th.no_grad():
            lower, upper = self.module.class_box(idx)
        boxes = concatenated_rows([lower, upper])
        return dict(zip(self.class_index_dict.keys(), boxes))

    @property
    def object_property_embeddings(self):
        """Roles as ``{iri: concat(range_l, range_u, error_l, error_u, slope, shift)}``."""
        idx = th.arange(len(self.object_property_index_dict), device=self.device)
        with th.no_grad():
            parts = concatenated_rows(self.module.role(idx))
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
