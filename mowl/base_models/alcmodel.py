from mowl.base_models.model import Model
import torch as th
from torch.utils.data import DataLoader
from tqdm import trange
from mowl.datasets.alc import ALCDataset
from mowl.owlapi import (
    OWLSubClassOfAxiom, OWLEquivalentClassesAxiom, OWLDisjointClassesAxiom,
    OWLClassAssertionAxiom, OWLObjectPropertyAssertionAxiom,
)


#: Loss-category names for the three FALCON loss terms.
TBOX = "tbox"
ABOX_EC = "abox_ec"
ABOX_EE = "abox_ee"


class EmbeddingALCModel(Model):
    """Abstract class that provides basic functionalities for methods that aim to embed the \
    :math:`\\mathcal{ALC}` language.

    Subclasses must implement :meth:`init_module`, which sets ``self.module`` to an
    instance of :class:`ALCModule <mowl.nn.alc.module.ALCModule>`.

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

    def __init__(self, dataset, embed_dim=50, batch_size=256, alpha=0.3, beta=0.3,
                 anon_e=4, learning_rate=0.001, model_filepath=None, device="cpu"):
        super().__init__(dataset, model_filepath=model_filepath)

        if not isinstance(batch_size, int):
            raise TypeError("Parameter batch_size must be of type int.")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str.")

        if not (0 <= alpha <= 1 and 0 <= beta <= 1):
            raise ValueError("Parameters alpha and beta must be in the interval [0, 1].")

        if alpha + beta > 1:
            raise ValueError("The sum of alpha and beta must not exceed 1 "
                             "(the ABox-EE weight is 1 - alpha - beta).")

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
        categories: TBox, concept-assertion (ABox-EC) or relation-assertion (ABox-EE)."""
        if isinstance(axiom, (OWLSubClassOfAxiom, OWLEquivalentClassesAxiom,
                              OWLDisjointClassesAxiom)):
            return TBOX
        elif isinstance(axiom, OWLClassAssertionAxiom):
            return ABOX_EC
        elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            return ABOX_EE
        return None

    def sample_anonymous_entities(self):
        """Samples a tensor of anonymous-entity embeddings for the current epoch. Half are \
        perturbations of existing named-entity embeddings and half are freshly initialised, \
        following the original FALCON implementation."""
        named = self.module.e_embedding.weight.detach()
        n_named, dim = named.shape[0], self.embed_dim
        half = self.anon_e // 2
        n_perturbed = min(half, n_named)
        n_fresh = self.anon_e - n_perturbed

        parts = []
        if n_perturbed > 0:
            noise = th.normal(0, 0.1, size=(n_perturbed, dim)).to(self.device)
            parts.append(named[:n_perturbed] + noise)
        if n_fresh > 0:
            fresh = th.empty(n_fresh, dim).to(self.device)
            th.nn.init.xavier_uniform_(fresh)
            parts.append(fresh)
        return th.cat(parts, dim=0)

    def _entity_context(self, anon_e_emb):
        """Builds the entity-membership context: all named entities followed by the sampled \
        anonymous entities (matching the original FALCON implementation). For purely \
        terminological ontologies (no individuals) only the anonymous entities are used."""
        n_named = len(self.dataset.individuals)
        named = self.module.e_embedding.weight[:n_named]
        return th.cat([named, anon_e_emb], dim=0)

    def _epoch_loss(self, dataloaders, e_emb):
        """Computes the alpha/beta-weighted FALCON loss over a set of dataloaders."""
        losses = {TBOX: [], ABOX_EC: [], ABOX_EE: []}
        for axiom, dataloader in dataloaders.items():
            category = self.axiom_category(axiom)
            if category is None:
                continue
            for batch_data in dataloaders[axiom]:
                losses[category].append(th.mean(self.module(axiom, batch_data[0], e_emb)))

        weights = {TBOX: self.alpha, ABOX_EC: self.beta,
                   ABOX_EE: 1 - self.alpha - self.beta}
        total = None
        for category, terms in losses.items():
            if not terms:
                continue
            term = sum(terms) / len(terms)
            weighted = weights[category] * term
            total = weighted if total is None else total + weighted
        return total

    def train(self, epochs, validate_every=1):
        """Trains the model.

        :param epochs: Number of training epochs.
        :type epochs: int
        :param validate_every: Run validation (and checkpoint the best model) every this many \
        epochs. Defaults to ``1``.
        :type validate_every: int, optional
        """
        if self.module is None:
            self.init_module()
        self.module = self.module.to(self.device)

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float("inf")

        for epoch in trange(epochs):
            self.module.train()
            anon_e_emb = self.sample_anonymous_entities()
            e_emb = self._entity_context(anon_e_emb)

            loss = self._epoch_loss(self.training_dataloaders, e_emb)
            if loss is None:
                raise ValueError("No trainable axioms were found in the training ontology.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            valid_loss = None
            if self.dataset.validation and (epoch + 1) % validate_every == 0:
                with th.no_grad():
                    self.module.eval()
                    anon_e_emb = self.sample_anonymous_entities()
                    e_emb = self._entity_context(anon_e_emb)
                    valid = self._epoch_loss(self.validation_dataloaders, e_emb)
                    valid_loss = valid.item() if valid is not None else None

                if valid_loss is not None and valid_loss < best_loss:
                    best_loss = valid_loss
                    if self.model_filepath is not None:
                        th.save(self.module.state_dict(), self.model_filepath)

        if self.model_filepath is not None and best_loss == float("inf"):
            th.save(self.module.state_dict(), self.model_filepath)

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
