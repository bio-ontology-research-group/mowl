import torch as th
from torch.utils.data import DataLoader

from mowl.base_models.model import Model
from mowl.datasets.alc import ALCDataset
from mowl.representation.library.logic_alc import (ABOX_EC, ABOX_EE, TBOX,
                                                   AnonymousElements, AxiomKindWeighting)
from mowl.training.loop import RegimeTrained
from mowl.training.regime import HOOKS

__all__ = ["EmbeddingALCModel", "TBOX", "ABOX_EC", "ABOX_EE"]

#: The hooks of a training regime that the loop of an ALC model consults: the course of
#: training, the regularisation and projections. The data, the negatives and the loss of an
#: axiom pattern are the objective's (:class:`AxiomKindWeighting` and the reading of the
#: geometry); a regime that changes them is refused when it is assigned.
ALC_TRAINING_HOOKS = frozenset(HOOKS) - {"training_datasets", "negatives", "filter_negatives",
                                         "loss"}


class _PatternBatches:
    """The rows of one axiom pattern as the training loop sees them: ``len`` is the number of
    rows, ``[:]`` the batches (a :class:`~torch.utils.data.DataLoader`, which draws its order
    when it is iterated)."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader.dataset)

    def __getitem__(self, idx):
        return self.loader

    @property
    def batches(self):
        return len(self.loader)


class EmbeddingALCModel(RegimeTrained, Model):
    """Abstract class that provides basic functionalities for methods that aim to embed the \
    :math:`\\mathcal{ALC}` language.

    Subclasses must implement :meth:`init_module`, which sets ``self.module`` to an
    instance of :class:`ALCModule <mowl.nn.alc.module.ALCModule>` (or a geometry that reads ALC
    axioms, :class:`~mowl.representation.library.logic_alc.ExpressionGeometry`).

    Training is the shared loop of :mod:`mowl.training.loop` (one optimisation step per epoch
    over all batches of all axiom patterns, validation and checkpointing every
    ``validate_every`` epochs) with these parts:

    * the loss of a step: the objective part
      :class:`~mowl.representation.library.logic_alc.AxiomKindWeighting` of the module's
      objective (``module.objective.weighting``), or, if it has none, one made of ``alpha``
      and ``beta``;
    * the anonymous entities of every epoch:
      :class:`~mowl.representation.library.logic_alc.AnonymousElements` (``anon_e``);
    * a training regime (:attr:`training_regime`): the regimes of the module
      (``module.default_regimes``, such as the annealing of Al-cone) unless another is
      assigned; the loop consults the hooks in :data:`ALC_TRAINING_HOOKS`.

    :param dataset: mOWL dataset to use for training.
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param embed_dim: Dimension of the embeddings. Defaults to ``50``.
    :type embed_dim: int, optional
    :param batch_size: Batch size for training. Defaults to ``256``.
    :type batch_size: int, optional
    :param alpha: Weight of the TBox loss term. Defaults to ``0.3``.
    :type alpha: float, optional
    :param beta: Weight of the concept-assertion (ABox-EC) loss term. The relation-assertion \
    (ABox-EE) term is weighted ``1 - alpha - beta``. Defaults to ``0.3``.
    :type beta: float, optional
    :param anon_e: Number of anonymous entities sampled each epoch. Defaults to ``4``.
    :type anon_e: int, optional
    :param learning_rate: Learning rate. Defaults to ``0.001``.
    :type learning_rate: float, optional
    """

    axiom_scoring = "method"
    scored_forms = ("gci0", "gci1", "gci2", "gci3")

    training_hooks = ALC_TRAINING_HOOKS
    #: One optimisation step per epoch, a progress bar, validation by the model itself.
    loop_style = "epochs"

    def __init__(self, dataset, embed_dim=50, batch_size=256, alpha=0.3, beta=0.3,
                 anon_e=4, learning_rate=0.001, model_filepath=None, device="cpu"):
        super().__init__(dataset, model_filepath=model_filepath)

        if not isinstance(batch_size, int):
            raise TypeError("Parameter batch_size must be of type int.")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str.")

        #: The weighting of the loss terms when the module's objective has none.
        self.own_weighting = AxiomKindWeighting(alpha, beta)

        self._datasets_loaded = False
        self._dataloaders_loaded = False
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.anon_e = anon_e
        self.learning_rate = learning_rate
        self.device = device

        #: The neural module. Set by :meth:`init_module` in subclasses.
        self.module = None

        self._training_dataset = None
        self._validation_dataset = None
        self._testing_dataset = None

        self._training_datasets = None
        self._validation_datasets = None
        self._testing_datasets = None

    def init_module(self):
        """Sets ``self.module`` to the :class:`ALCModule <mowl.nn.alc.module.ALCModule>` \
        used by this model. Must be implemented by subclasses."""
        raise NotImplementedError()

    @property
    def evaluation_model(self):
        """The module used for evaluation. Passed to the evaluator by \
        :meth:`Model.evaluate <mowl.base_models.model.Model.evaluate>`."""
        if self.module is None:
            self.init_module()
        return self.module

    def _embeddings_dict(self, embedding, index_dict):
        if self.module is None:
            self.init_module()
        weight = embedding.weight.detach().cpu().numpy()
        return {name: weight[idx] for name, idx in index_dict.items()}

    @property
    def class_embeddings(self):
        """Returns a dictionary mapping class names to their embeddings."""
        return self._embeddings_dict(self.module.c_embedding, self.class_index_dict)

    @property
    def object_property_embeddings(self):
        """Returns a dictionary mapping object property names to their embeddings."""
        return self._embeddings_dict(self.module.r_embedding,
                                     self.object_property_index_dict)

    @property
    def individual_embeddings(self):
        """Returns a dictionary mapping individual names to their embeddings."""
        return self._embeddings_dict(self.module.e_embedding, self.individual_index_dict)

    # -- axiom scores for the rank-based evaluation regimes ------------------------------
    _PATTERN_NS = "http://mowl.borg/pattern#"

    def _axiom_pattern(self, gci_name):
        """The axiom pattern of a normal form, whose named classes and roles are filled
        from the columns of a row in the order mOWL's normal-form datasets use."""
        patterns = getattr(self, "_axiom_patterns", None)
        if patterns is None:
            patterns = self._axiom_patterns = {}
        if gci_name not in patterns:
            from mowl.owlapi import OWLAPIAdapter
            adapter = OWLAPIAdapter()
            a, b, c = (adapter.create_class(self._PATTERN_NS + n) for n in "ABC")
            r = adapter.create_object_property(self._PATTERN_NS + "r")
            some = adapter.create_object_some_values_from
            if gci_name == "gci0":           # [C, D]: C ⊑ D
                pair = (a, b)
            elif gci_name == "gci1":         # [C, D, E]: C ⊓ D ⊑ E
                pair = (adapter.create_object_intersection_of(a, b), c)
            elif gci_name == "gci2":         # [C, r, D]: C ⊑ ∃r.D
                pair = (a, some(r, b))
            elif gci_name == "gci3":         # [r, C, D]: ∃r.C ⊑ D
                pair = (some(r, a), b)
            else:
                from mowl.evaluation.model_scoring import UnsupportedAxiomForm
                raise UnsupportedAxiomForm(
                    f"{type(self).__name__} has no score for {gci_name!r} axioms.")
            patterns[gci_name] = pair
        return patterns[gci_name]

    # -- the parts of training -----------------------------------------------------------
    @property
    def weighting(self):
        """The objective part that combines the costs of the axiom patterns into a loss: the
        one of the module's objective, or the model's own (``alpha``, ``beta``)."""
        objective = getattr(self.module, "objective", None)
        own = getattr(objective, "weighting", None)
        return self.own_weighting if own is None else own

    @property
    def anonymous_elements(self):
        """The sampler of the anonymous entities of an epoch
        (:class:`~mowl.representation.library.logic_alc.AnonymousElements`, ``anon_e``)."""
        sampler = self.__dict__.get("_anonymous_elements")
        if sampler is None or sampler.count != self.anon_e:
            sampler = self._anonymous_elements = AnonymousElements(self.anon_e)
        return sampler

    def _scoring_context(self):
        """The entity context axioms are scored against: the named individuals and one
        fixed sample of anonymous entities (drawn once, so that scores are reproducible)."""
        context = getattr(self, "_axiom_scoring_context", None)
        if context is None:
            context = self.anonymous_elements.with_seed(
                lambda: self._entity_context(self.sample_anonymous_entities()))
            self._axiom_scoring_context = context
        return context

    def _subsumption_costs(self, sub, sup, x, e_emb):
        """Cost of ``sub ⊑ sup`` per row of ``x``: the measure of ``sub ⊓ ¬sup``, as in
        the training loss of the module."""
        module = self.module
        c_fs, nxt = module.forward_fs(sub, x, e_emb)
        d_fs, _ = module.forward_fs(sup, x, e_emb, cur_index=nxt)
        return module.get_cc_loss(module._logical_and(c_fs, module._logical_not(d_fs)))

    def axiom_costs(self, gci_name, rows, max_cells=2 ** 26):
        """Cost of every candidate axiom in ``rows`` (mOWL normal-form layout, model
        indices): the loss the module assigns to the axiom, lower is more plausible.

        :param max_cells: Bound on ``rows per chunk × entities × dimension``.
        """
        from mowl.evaluation.model_scoring import costs_in_chunks
        if self.module is None:
            self.init_module()
        sub, sup = self._axiom_pattern(gci_name)
        e_emb = self._scoring_context()
        width = 1 if e_emb is None else e_emb.numel()
        return costs_in_chunks(lambda x: self._subsumption_costs(sub, sup, x, e_emb),
                               rows.to(self.device), max_cells, width)

    def _load_datasets(self):
        """This method will create different data attributes and finally the corresponding \
            DataLoaders for each axiom pattern in each subset (training, validation and testing).
        """
        if self._datasets_loaded:
            return

        self._training_dataset = ALCDataset(
            self.dataset.ontology, self.dataset, device=self.device)
        self._training_datasets, _ = self._training_dataset.get_datasets()

        self._validation_datasets = None
        if self.dataset.validation:
            self._validation_dataset = ALCDataset(
                self.dataset.validation, self.dataset, device=self.device)
            self._validation_datasets, _ = self._validation_dataset.get_datasets()

        self._testing_datasets = None
        if self.dataset.testing:
            self._testing_dataset = ALCDataset(
                self.dataset.testing, self.dataset, device=self.device)
            self._testing_datasets, _ = self._testing_dataset.get_datasets()

        self._datasets_loaded = True

    def _load_dataloaders(self):
        if self._dataloaders_loaded:
            return

        self._load_datasets()

        self._training_dataloaders = {
            k: DataLoader(v, batch_size=self.batch_size, shuffle=True) for k, v in
            self._training_datasets.items()}

        self._validation_dataloaders = {}
        if self._validation_datasets:
            self._validation_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size) for k, v in
                self._validation_datasets.items()}

        self._testing_dataloaders = {}
        if self._testing_datasets:
            self._testing_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size) for k, v in
                self._testing_datasets.items()}

        self._dataloaders_loaded = True

    @staticmethod
    def axiom_category(axiom):
        """Classifies an axiom (or axiom pattern) into one of the three FALCON loss \
        categories: TBox, concept-assertion (ABox-EC) or relation-assertion (ABox-EE) \
        (:meth:`AxiomKindWeighting.kind \
        <mowl.representation.library.logic_alc.AxiomKindWeighting.kind>`)."""
        return AxiomKindWeighting.kind(axiom)

    def sample_anonymous_entities(self):
        """Samples a tensor of anonymous-entity embeddings for the current epoch. Half are \
        perturbations of existing named-entity embeddings and half are freshly initialised, \
        following the original FALCON implementation \
        (:class:`~mowl.representation.library.logic_alc.AnonymousElements`)."""
        return self.anonymous_elements.sample(self.module.e_embedding.weight.detach(),
                                              self.embed_dim, self.device)

    def _entity_context(self, anon_e_emb):
        """Builds the entity-membership context: all named entities followed by the sampled \
        anonymous entities (matching the original FALCON implementation). For purely \
        terminological ontologies (no individuals) only the anonymous entities are used."""
        return self.anonymous_elements.context(self.module.e_embedding.weight,
                                               len(self.dataset.individuals), anon_e_emb)

    def _epoch_loss(self, dataloaders, e_emb):
        """The loss of all batches of ``{axiom: batches}`` on the entity context ``e_emb``
        (the :attr:`weighting`), or ``None`` if no axiom is trained."""
        return self.weighting.loss(self.module, dataloaders, e_emb, self.device)

    # -- the shared training loop (mowl.training.loop) ---------------------------------------
    @property
    def training_datasets(self):
        """``{axiom pattern: its batches}`` of the training axioms of a kind the
        :attr:`weighting` trains, what the loop trains on."""
        return self._patterns(self.training_dataloaders)

    def _patterns(self, dataloaders):
        weighting = self.weighting
        return {axiom: _PatternBatches(loader) for axiom, loader in dataloaders.items()
                if weighting.kind(axiom) is not None}

    def log_training_datasets(self, datasets):
        if not datasets:
            raise ValueError("No trainable axioms were found in the training ontology.")
        self._best_loss = float("inf")

    def on_epoch_start(self, step, steps):
        """The entity context of the epoch (a fresh sample of anonymous entities)."""
        self._epoch_context = self._entity_context(self.sample_anonymous_entities())

    def _step_loss(self, datasets, step, negatives=True):
        self._batch_counts = self.weighting.batch_counts(
            {axiom: range(batches.batches) for axiom, batches in datasets.items()})
        return super()._step_loss(datasets, step, negatives=False)

    def score_rows(self, form, rows, neg=False):
        """The cost of every batch of the axiom pattern ``form`` on the context of the
        epoch (:meth:`AxiomKindWeighting.batch_costs
        <mowl.representation.library.logic_alc.AxiomKindWeighting.batch_costs>`)."""
        return self.weighting.batch_costs(self.module, form, rows, self._epoch_context,
                                          self.device)

    def combine_scores(self, form, pos_scores, neg_scores=None):
        return self.weighting.combine(form, pos_scores, self._batch_counts)

    def after_optimizer_step(self):
        super().after_optimizer_step()
        project = getattr(self.module, "project", None)
        if project is not None:
            with th.no_grad():
                project()

    def validation_loss(self):
        """The loss of the validation axioms (no gradient, evaluation mode, a fresh context),
        or ``None``."""
        with th.no_grad():
            self.module.eval()
            e_emb = self._entity_context(self.sample_anonymous_entities())
            valid = self._epoch_loss(self.validation_dataloaders, e_emb)
        return None if valid is None else valid.item()

    def validate_epoch(self, epoch, loss):
        """Validate and save the parameters (``model_filepath``) when the validation loss is
        the best so far."""
        if not self.dataset.validation:
            return
        valid = self.validation_loss()
        if valid is not None and valid < self._best_loss:
            self._best_loss = valid
            if self.model_filepath is not None:
                self.save_parameters()

    def save_parameters(self):
        th.save(self.module.state_dict(), self.model_filepath)

    def finish_training(self):
        if self.model_filepath is not None and self._best_loss == float("inf"):
            self.save_parameters()

    def init_training(self):
        if self.module is None:
            self.init_module()
        self.module = self.module.to(self.device)

    def train(self, epochs, validate_every=1):
        """Trains the model with the shared loop (:mod:`mowl.training.loop`).

        :param epochs: Number of training epochs.
        :type epochs: int
        :param validate_every: Run validation (and checkpoint the best model) every this many \
        epochs. Defaults to ``1``.
        :type validate_every: int, optional
        :return: The loss of every epoch.
        """
        self.init_training()
        return self.run_training(epochs, validate_every)

    @property
    def training_dataset(self):
        """Returns the training dataset, an instance of \
:class:`ALCDataset <mowl.datasets.alc.ALCDataset>`."""
        self._load_datasets()
        return self._training_dataset

    @property
    def validation_dataset(self):
        """Returns the validation dataset, an instance of \
:class:`ALCDataset <mowl.datasets.alc.ALCDataset>`."""
        if self.dataset.validation is None:
            raise AttributeError("Validation dataset is None.")

        self._load_datasets()
        return self._validation_dataset

    @property
    def testing_dataset(self):
        """Returns the testing dataset, an instance of \
:class:`ALCDataset <mowl.datasets.alc.ALCDataset>`."""
        if self.dataset.testing is None:
            raise AttributeError("Testing dataset is None.")

        self._load_datasets()
        return self._testing_dataset

    @property
    def training_dataloaders(self):
        """Returns the training dataloaders for each axiom pattern. Each dataloader is an \
instance of :class:`torch.utils.data.DataLoader`.

        :rtype: dict
        """
        self._load_dataloaders()
        return self._training_dataloaders

    @property
    def validation_dataloaders(self):
        """Returns the validation dataloaders for each axiom pattern. Each dataloader is an \
instance of :class:`torch.utils.data.DataLoader`.

        :rtype: dict
        """
        if self.dataset.validation is None:
            raise AttributeError("Validation dataloader is None.")

        self._load_dataloaders()
        return self._validation_dataloaders

    @property
    def testing_dataloaders(self):
        """Returns the testing dataloaders for each axiom pattern. Each dataloader is an \
instance of :class:`torch.utils.data.DataLoader`.

        :rtype: dict
        """
        if self.dataset.testing is None:
            raise AttributeError("Testing dataloader is None.")

        self._load_dataloaders()
        return self._testing_dataloaders
