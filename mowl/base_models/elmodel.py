from mowl.ontology.normalize import ELNormalizerBase
from mowl.base_models.model import Model
from mowl.datasets.el import ELDataset
from mowl.projection import projector_factory
import torch as th
from torch.utils.data import DataLoader, default_collate
from tqdm import trange
import warnings

from deprecated.sphinx import versionadded, versionchanged

from org.semanticweb.owlapi.model import OWLClassExpression, OWLClass, OWLObjectSomeValuesFrom, OWLObjectIntersectionOf

import copy
import numbers
import numpy as np
import mowl.error.messages as msg
import os
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

@versionchanged(version="2.0.0", reason="Added the 'load_normalized' parameter.")
class EmbeddingELModel(Model):
    """Abstract class for :math:`\mathcal{EL}` embedding methods.

    :param dataset: mOWL dataset to use for training and evaluation.
    :type dataset: :class:`mowl.datasets.Dataset`
    :param embed_dim: The embedding dimension.
    :type embed_dim: int
    :param batch_size: The batch size to use for training.
    :type batch_size: int
    :param extended: If `True`, the model is supposed with 7 EL normal forms. This will be \
reflected on the :class:`DataLoaders` that will be generated and also the model must \
    contain 7 loss functions. If `False`, the model will work with 4 normal forms only, \
merging the 3 extra to their corresponding origin normal forms. Defaults to True
    :type extended: bool, optional
    :param load_normalized: If `True`, the ontology is assumed to be normalized and GCIs are extracted directly. Defaults to False.
    :type load_normalized: bool, optional
    :param device: The device to use for training. Defaults to "cpu".
    :type device: str, optional
    :param neg_sampling_gcis: List of GCI names for which negative sampling should be applied \
during training. If ``None`` (default), negative sampling is applied automatically to all GCIs \
declared in the module's :attr:`~mowl.nn.ELModule.neg_capable_gcis` (i.e. only what the module \
actually supports). Pass an explicit list to override this — a :class:`NotImplementedError` is \
raised at the start of training if any requested GCI is not in ``neg_capable_gcis``. Bot GCIs \
(``"gci0_bot"``, ``"gci1_bot"``, ``"gci3_bot"``) are never subject to negative sampling.
    :type neg_sampling_gcis: list of str, optional
    :param normalizer: Normalizer used to transform the ontologies into normal forms. Passed \
through to :class:`~mowl.datasets.el.ELDataset`. If not provided, an \
:class:`~mowl.ontology.normalize.ELNormalizer` is used. Pass \
:class:`~mowl.ontology.normalize.ELNormalizerOld` to reproduce results obtained before \
mOWL 2.2.0. Defaults to ``None``.
    :type normalizer: :class:`~mowl.ontology.normalize.ELNormalizerBase`, optional

    .. versionchanged:: 2.2.0
        Added the ``normalizer`` parameter.

    :param load_role_axioms: If `True`, the :math:`\\mathcal{EL}^{++}` role axioms of the \
ontology (role inclusions and role chains) are loaded as the ``role_inclusion`` and \
``role_chain`` datasets and trained on like any other normal form. The module must implement \
``role_inclusion_loss`` and ``role_chain_loss`` and declare \
:attr:`~mowl.nn.ELModule.role_axiom_capable`, otherwise a :class:`NotImplementedError` is \
raised at the start of training. Defaults to False.
    :type load_role_axioms: bool, optional
    """

    #: Default per-GCI negative sampling configuration used by :meth:`get_negative_sampling_config`.
    #: Bot GCIs are intentionally excluded — subsumption by bottom has no meaningful negative.
    #: Each entry specifies the entity pool(s) to sample from (``"classes"`` or
    #: ``"individuals"``) and which column(s) of the data tensor to corrupt with random
    #: indices. ``corrupt_column`` may be a single int or a list of ints (one set of
    #: negatives is generated per column); ``index_pool`` may be a single pool name applied
    #: to all columns or a list of pool names, one per column.
    _DEFAULT_NEG_SAMPLING_CONFIG = {
        "gci0":     {"index_pool": "classes",     "corrupt_column": 1},
        "gci1":     {"index_pool": "classes",     "corrupt_column": 2},
        "gci2":     {"index_pool": "classes",     "corrupt_column": 2},
        "gci3":     {"index_pool": "classes",     "corrupt_column": 2},
        "class_assertion":           {"index_pool": "classes",     "corrupt_column": 1},
        "object_property_assertion": {"index_pool": "individuals", "corrupt_column": 2},
    }

    #: Normal forms that are loaded only when ``load_role_axioms=True`` and that require a
    #: module declaring :attr:`role_axiom_capable <mowl.nn.ELModule.role_axiom_capable>`.
    ROLE_AXIOM_GCIS = ("role_inclusion", "role_chain")

    def __init__(self, dataset, embed_dim, batch_size, extended=True, model_filepath=None,
                 load_normalized=False, device="cpu", learning_rate=0.001,
                 neg_sampling_gcis=None, normalizer=None, load_role_axioms=False):
        super().__init__(dataset, model_filepath=model_filepath)

        if not isinstance(embed_dim, int):
            raise TypeError("Parameter 'embed_dim' must be of type int.")

        if not isinstance(batch_size, int):
            raise TypeError("Parameter batch_size must be of type int.")

        if not isinstance(extended, bool):
            raise TypeError("Optional parameter extended must be of type bool.")

        if not isinstance(load_normalized, bool):
            raise TypeError("Optional parameter load_normalized must be of type bool.")

        if not isinstance(load_role_axioms, bool):
            raise TypeError("Optional parameter load_role_axioms must be of type bool.")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str.")

        if normalizer is not None and not isinstance(normalizer, ELNormalizerBase):
            raise TypeError("Optional parameter normalizer must be a subclass of \
mowl.ontology.normalize.ELNormalizerBase")

        self.normalizer = normalizer

        # Index dictionaries extended with the auxiliary entities that normalization
        # introduces. Populated by _load_datasets and shared by every ELDataset it builds.
        self._el_class_index_dict = None
        self._el_object_property_index_dict = None

        self._datasets_loaded = False
        self._dataloaders_loaded = False
        self._extended = extended
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.device = device
        self.load_normalized = load_normalized
        self.learning_rate = learning_rate
        self.neg_sampling_gcis = neg_sampling_gcis
        self.load_role_axioms = load_role_axioms

        self._training_datasets = None
        self._validation_datasets = None
        self._testing_datasets = None

        self._loaded_eval = False
        self._eval_gci_name = None

    @property
    def eval_gci_name(self):
        """The GCI type to use for evaluation (e.g., 'gci0', 'gci1', 'gci2', 'gci3').
        Must be explicitly set before evaluation.

        :rtype: str
        """
        return self._eval_gci_name

    @eval_gci_name.setter
    def eval_gci_name(self, value):
        valid_gci_names = ["gci0", "gci1", "gci2", "gci3"]
        if self._extended:
            valid_gci_names.extend(["gci0_bot", "gci1_bot", "gci3_bot"])
        if value not in valid_gci_names:
            raise ValueError(f"eval_gci_name must be one of {valid_gci_names}, got '{value}'")
        self._eval_gci_name = value

    @property
    def class_index_dict(self):
        """Dictionary with class names as keys and class indexes as values, extended with the \
auxiliary concepts that normalization introduces.

        Those concepts are not in the signature of the ontology, so they are absent from
        :attr:`mowl.base_models.Model.class_index_dict`, but the model still has to embed
        them. They are appended after the ontology classes, which leaves the index of every
        ontology class unchanged. Reading this property normalizes the ontologies.

        :rtype: dict

        .. versionchanged:: 2.2.0
            Includes the auxiliary concepts introduced during normalization.
        """
        self._load_datasets()
        return self._el_class_index_dict

    @property
    def object_property_index_dict(self):
        """Dictionary with object property names as keys and indexes as values, extended with \
the auxiliary object properties that normalization introduces.

        :rtype: dict

        .. versionchanged:: 2.2.0
            Includes the auxiliary object properties introduced during normalization.
        """
        self._load_datasets()
        return self._el_object_property_index_dict

    def init_module(self):
        raise NotImplementedError

    def _load_datasets(self):
        """This method will create different data attributes and finally the corresponding \
            DataLoaders for each GCI type in each subset (training, validation and testing).
        """
        if self._datasets_loaded:
            return

        # Get ontology paths for caching if available (PathDataset provides these)
        ontology_path = getattr(self.dataset, 'ontology_path', None)
        validation_path = getattr(self.dataset, 'validation_path', None)
        testing_path = getattr(self.dataset, 'testing_path', None)

        # One index dictionary per entity kind, shared by the three ELDatasets below so that
        # the auxiliary entities any of them introduces are visible to all of them and to the
        # model. Seeded from the ontology signature, hence the base-class properties: reading
        # the overridden ones here would recurse back into this method. Individuals are left
        # to ELDataset, as before: normalization introduces no auxiliary individuals.
        self._el_class_index_dict = Model.class_index_dict.fget(self)
        self._el_object_property_index_dict = Model.object_property_index_dict.fget(self)

        training_el_dataset = ELDataset(self.dataset.ontology,
                                        self._el_class_index_dict,
                                        self._el_object_property_index_dict,
                                        extended=self._extended,
                                        load_normalized=self.load_normalized,
                                        device=self.device,
                                        ontology_path=ontology_path,
                                        normalizer=self.normalizer,
                                        load_role_axioms=self.load_role_axioms)

        self._training_datasets = training_el_dataset.get_gci_datasets()

        self._validation_datasets = None
        if self.dataset.validation:
            validation_el_dataset = ELDataset(self.dataset.validation,
                                              self._el_class_index_dict,
                                              self._el_object_property_index_dict,
                                              extended=self._extended, device=self.device,
                                              ontology_path=validation_path,
                                              normalizer=self.normalizer,
                                              load_role_axioms=self.load_role_axioms)

            self._validation_datasets = validation_el_dataset.get_gci_datasets()

        self._testing_datasets = None
        if self.dataset.testing:
            testing_el_dataset = ELDataset(self.dataset.testing,
                                           self._el_class_index_dict,
                                           self._el_object_property_index_dict,
                                           extended=self._extended, device=self.device,
                                           ontology_path=testing_path,
                                           normalizer=self.normalizer,
                                           load_role_axioms=self.load_role_axioms)

            self._testing_datasets = testing_el_dataset.get_gci_datasets()

        self._datasets_loaded = True

    def _load_dataloaders(self):
        if self._dataloaders_loaded:
            return

        self._load_datasets()

        self._training_dataloaders = {
            k: DataLoader(v, batch_size=self.batch_size, pin_memory=False) for k, v in
            self._training_datasets.items()}

        if self._validation_datasets:
            self._validation_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size, pin_memory=False) for k, v in
                self._validation_datasets.items()}

        if self._testing_datasets:
            self._testing_dataloaders = {
                k: DataLoader(v, batch_size=self.batch_size, pin_memory=False) for k, v in
                self._testing_datasets.items()}

        self._dataloaders_loaded = True

    @property
    def training_datasets(self):
        """Returns the training datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        self._load_datasets()
        return self._training_datasets

    @property
    def validation_datasets(self):
        """Returns the validation datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        if self.dataset.validation is None:
            raise AttributeError("Validation dataset is None.")

        self._load_datasets()
        return self._validation_datasets

    @property
    def testing_datasets(self):
        """Returns the testing datasets for each GCI type. Each dataset is an instance \
of :class:`mowl.datasets.el.ELDataset`

        :rtype: dict
        """
        if self.dataset.testing is None:
            raise AttributeError("Testing dataset is None.")

        self._load_datasets()
        return self._testing_datasets

    @property
    def training_dataloaders(self):
        """Returns the training dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        self._load_dataloaders()
        return self._training_dataloaders

    @property
    def validation_dataloaders(self):
        """Returns the validation dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        if self.dataset.validation is None:
            raise AttributeError("Validation dataloader is None.")

        self._load_dataloaders()
        return self._validation_dataloaders

    @property
    def testing_dataloaders(self):
        """Returns the testing dataloaders for each GCI type. Each dataloader is an instance \
of :class:`torch.utils.data.DataLoader`

        :rtype: dict
        """
        if self.dataset.testing is None:
            raise AttributeError("Testing dataloader is None.")

        self._load_dataloaders()
        return self._testing_dataloaders

    # ==================== Training Methods ====================

    def get_negative_sampling_config(self):
        """Returns the active negative sampling configuration.

        When ``neg_sampling_gcis`` is ``None`` (the default), the configuration is derived
        automatically from the intersection of :attr:`_DEFAULT_NEG_SAMPLING_CONFIG` and the
        module's :attr:`~mowl.nn.ELModule.neg_capable_gcis` — so only GCIs that the module
        genuinely supports are included.

        When ``neg_sampling_gcis`` is set explicitly, only those GCIs are included. Training
        will raise :class:`NotImplementedError` if any of them are absent from
        ``neg_capable_gcis``.

        Override this method to customise which GCI types require negative sampling
        and how negatives should be generated.

        :return: Dictionary mapping GCI names to their negative sampling config.
            Each entry has:

            - ``'index_pool'``: ``'classes'`` or ``'individuals'`` — pool(s) to sample
              from. A single name applies to all corrupted columns, or a list of names
              (one per column, must match the length of ``'corrupt_column'``).
            - ``'corrupt_column'``: int or list of ints — which column(s) of the data
              tensor to replace with random indices. When a list is given, one set of
              negative samples is generated per column and concatenated, so the model
              sees K negatives per positive.

        :rtype: dict
        """
        if self.neg_sampling_gcis is None:
            return {k: v for k, v in self._DEFAULT_NEG_SAMPLING_CONFIG.items()
                    if k in self.module.neg_capable_gcis}
        return {k: v for k, v in self._DEFAULT_NEG_SAMPLING_CONFIG.items()
                if k in self.neg_sampling_gcis}

    #: Entity pools that ``index_pool`` may name, mapped to the model attribute holding the
    #: corresponding entity-to-index dictionary.
    _NEG_SAMPLING_POOLS = {"classes": "class_index_dict",
                           "individuals": "individual_index_dict"}

    @staticmethod
    def _as_column_index(value, prefix):
        """Returns ``value`` as a column index. Booleans and non-integers are rejected \
        here rather than being mistaken for a column number later on.

        :meta private:
        """
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(
                f"{prefix}'corrupt_column' must be an integer or a list of integers, but "
                f"{value!r} is of type {type(value).__name__}.")
        return int(value)

    def _resolve_neg_config(self, gci_name, num_columns=None, require_pool=True):
        """Normalises the negative sampling entry of one GCI into the pair of lists that \
        :meth:`generate_negatives` works with.

        ``'corrupt_column'`` may be a single column index or a list of them, and \
        ``'index_pool'`` a single pool name or one name per column; both forms are \
        returned here as lists of the same length. Any integer type is accepted for a \
        column (``numpy`` integers included), booleans are not.

        :param gci_name: Name of the GCI type, which must be present in \
        :meth:`get_negative_sampling_config`.
        :type gci_name: str
        :param num_columns: Width of the GCI's data tensor. When given, every column is \
        checked against it. Defaults to ``None``.
        :type num_columns: int, optional
        :param require_pool: Whether a missing ``'index_pool'`` is an error. Pass ``False`` \
        to validate only what a subclass overriding :meth:`generate_negatives` still needs, \
        in which case the returned pools may be ``None``. Defaults to ``True``.
        :type require_pool: bool, optional
        :raises ValueError: if the entry is malformed
        :rtype: tuple(list of int, list of str or None)

        :meta private:
        """
        cfg = self.get_negative_sampling_config()[gci_name]
        prefix = f"Negative sampling config for '{gci_name}': "

        if "corrupt_column" not in cfg:
            raise ValueError(f"{prefix}the entry has no 'corrupt_column' key.")

        corrupt_column = cfg["corrupt_column"]
        if isinstance(corrupt_column, (list, tuple)):
            columns = [self._as_column_index(column, prefix) for column in corrupt_column]
            if not columns:
                raise ValueError(
                    f"{prefix}'corrupt_column' is empty. Give at least one column, or drop "
                    "the entry to disable negative sampling for this normal form.")
        else:
            columns = [self._as_column_index(corrupt_column, prefix)]

        if num_columns is not None:
            for column in columns:
                if not 0 <= column < num_columns:
                    raise ValueError(
                        f"{prefix}corrupt column {column} is out of range (the data tensor "
                        f"has {num_columns} columns).")

        if "index_pool" not in cfg:
            if require_pool:
                raise ValueError(
                    f"{prefix}the entry has no 'index_pool' key. Give one of "
                    f"{sorted(self._NEG_SAMPLING_POOLS)}, or override generate_negatives() "
                    "to sample from a pool of your own.")
            return columns, None

        index_pool = cfg["index_pool"]
        if isinstance(index_pool, str):
            pools = [index_pool] * len(columns)
        elif isinstance(index_pool, (list, tuple)):
            pools = list(index_pool)
            if len(pools) != len(columns):
                raise ValueError(
                    f"{prefix}'index_pool' has {len(pools)} entries but 'corrupt_column' "
                    f"has {len(columns)}. Either give a single pool name or one pool per "
                    "corrupted column.")
        else:
            raise ValueError(
                f"{prefix}'index_pool' must be a pool name or a list of pool names, but "
                f"{index_pool!r} is of type {type(index_pool).__name__}.")

        for pool in pools:
            if pool not in self._NEG_SAMPLING_POOLS:
                raise ValueError(f"Unknown index_pool: {pool}")

        return columns, pools

    def generate_negatives(self, gci_name, gci_dataset):
        """Generate negative samples for a given GCI type.

        One or more columns of the data tensor are corrupted with random entity
        indices, according to the GCI's negative sampling configuration (see
        :meth:`get_negative_sampling_config`). When ``'corrupt_column'`` holds
        several columns, one set of negative samples is generated per column
        (each corrupting only its own column) and the sets are concatenated,
        i.e. K negatives per positive for K columns.

        Override this method for custom negative sampling strategies.

        :param gci_name: Name of the GCI type (e.g., 'gci2')
        :type gci_name: str
        :param gci_dataset: The dataset containing positive samples
        :type gci_dataset: torch.Tensor
        :return: Negative samples tensor, or None if no negatives for this GCI type
        :rtype: torch.Tensor or None
        """
        config = self.get_negative_sampling_config()
        if gci_name not in config:
            return None

        data = gci_dataset[:]
        columns, pools = self._resolve_neg_config(gci_name, num_columns=data.shape[1])

        neg_blocks = []
        for pool, column in zip(pools, columns):
            all_ids = list(getattr(self, self._NEG_SAMPLING_POOLS[pool]).values())

            idxs_for_negs = np.random.choice(all_ids, size=len(gci_dataset), replace=True)
            rand_index = th.tensor(idxs_for_negs, dtype=th.long, device=self.device)

            # Build negative data by replacing the specified column
            neg_block = th.cat([data[:, :column], rand_index.unsqueeze(1)], dim=1)
            if column + 1 < data.shape[1]:
                neg_block = th.cat([neg_block, data[:, column + 1:]], dim=1)
            neg_blocks.append(neg_block)

        return th.cat(neg_blocks, dim=0)

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute loss from positive and negative scores.

        Override this method to use different loss functions (e.g., MSE loss).

        .. warning::
           ``neg_scores`` is **not** aligned row-by-row with ``pos_scores``. A negative
           sampling configuration that corrupts *K* columns yields ``K`` negatives per
           positive (see :meth:`generate_negatives`), so ``neg_scores`` holds
           ``K * len(pos_scores)`` rows. Reduce each tensor before combining them, as the
           implementations here and in :class:`ELBE <mowl.models.ELBE>` do. Pairing them
           elementwise -- ``pos_scores - neg_scores`` -- silently broadcasts into an
           ``(n, K*n)`` matrix whose mean is still a scalar, so training runs and optimises
           the wrong objective.

        :param pos_scores: Scores for positive samples (should be minimized)
        :type pos_scores: torch.Tensor
        :param neg_scores: Scores for negative samples (should be maximized), or None. May \
        contain more rows than ``pos_scores`` -- see the warning above.
        :type neg_scores: torch.Tensor or None
        :return: Combined loss value
        :rtype: torch.Tensor
        """
        loss = th.mean(pos_scores)
        if neg_scores is not None:
            loss += th.mean(neg_scores)
        return loss

    def get_regularization_loss(self):
        """Get regularization loss from the module.

        Override this method if your module has a regularization loss.

        :return: Regularization loss value
        :rtype: torch.Tensor
        """
        if hasattr(self.module, 'regularization_loss'):
            return self.module.regularization_loss()
        return 0

    def get_optimizer(self):
        """Create and return the optimizer.

        Override this method to use a different optimizer or configuration.

        :return: Optimizer instance
        :rtype: torch.optim.Optimizer
        """
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train the model.

        This is the generic training loop for EL embedding models. Subclasses can
        customize behavior by overriding:
        - :meth:`get_negative_sampling_config`: Configure which GCIs need negatives
        - :meth:`generate_negatives`: Custom negative sampling strategy
        - :meth:`compute_loss`: Custom loss computation (e.g., MSE loss)
        - :meth:`get_regularization_loss`: Add regularization
        - :meth:`get_optimizer`: Use different optimizer

        :param epochs: Number of training epochs
        :type epochs: int
        :param validate_every: Validate and log every N epochs. Defaults to 1.
        :type validate_every: int, optional
        :param epoch_callback: Optional callable invoked after each epoch as
            ``epoch_callback(epoch, model)``, where *epoch* is the 0-based epoch
            index and *model* is this model instance. Use it to capture snapshots
            for animation, custom logging, or early stopping. Defaults to ``None``.
        :type epoch_callback: callable, optional
        """
        logger.warning(
            'You are using the default training method. If you want to use a customized '
            'training method (e.g., different negative sampling, etc.), please override '
            'the appropriate methods in a subclass.'
        )

        # Verify that the module can consume the EL++ role axioms, if they were loaded
        role_axiom_gcis = [gci for gci in self.ROLE_AXIOM_GCIS
                           if gci in self.training_datasets]
        if role_axiom_gcis and not self.module.role_axiom_capable:
            raise NotImplementedError(
                f"'load_role_axioms=True' loaded the EL++ role axiom dataset(s) "
                f"{role_axiom_gcis}, but '{type(self.module).__name__}' does not declare "
                f"'role_axiom_capable = True'. Implement 'role_inclusion_loss' and "
                f"'role_chain_loss' on the module and set the flag, or construct the model "
                f"with 'load_role_axioms=False' (the default) to train on the concept "
                f"normal forms only."
            )

        # Verify that every GCI configured for negative sampling has a true negative loss
        neg_config = self.get_negative_sampling_config()
        if neg_config:
            incapable = [gci for gci in neg_config
                         if gci not in self.module.neg_capable_gcis]
            if incapable:
                capable = sorted(self.module.neg_capable_gcis) or ["none"]
                raise NotImplementedError(
                    f"Negative sampling was requested for GCI(s) {incapable}, but "
                    f"'{type(self.module).__name__}' does not implement a negative loss "
                    f"for these GCIs (i.e. their loss function ignores neg=True). "
                    f"Either implement the negative loss by handling 'neg=True' in the "
                    f"corresponding loss function(s), or restrict negative sampling to "
                    f"the supported GCIs via the 'neg_sampling_gcis' parameter. "
                    f"GCIs with negative loss support in this module: {capable}."
                )

        # Verify that every negative sampling entry is well formed, before the first epoch
        # rather than in the middle of one. A missing 'index_pool' is not an error here: a
        # subclass may override generate_negatives() and sample from a pool of its own, as
        # the shipped PPI examples do.
        for gci_name in neg_config:
            gci_dataset = self.training_datasets.get(gci_name)
            data = None if gci_dataset is None else gci_dataset.data
            num_columns = data.shape[1] if data is not None and data.dim() == 2 else None
            self._resolve_neg_config(gci_name, num_columns=num_columns, require_pool=False)

        # Log dataset sizes
        points_per_dataset = {k: len(v) for k, v in self.training_datasets.items()}
        string = "Training datasets: \n"
        for k, v in points_per_dataset.items():
            string += f"\t{k}: {v}\n"
        logger.info(string)

        optimizer = self.get_optimizer()
        best_loss = float('inf')

        for epoch in trange(epochs):
            self.module.train()

            train_loss = 0
            loss = th.tensor(0.0, device=self.device)

            for gci_name, gci_dataset in self.training_datasets.items():
                if len(gci_dataset) == 0:
                    continue

                # Compute positive loss
                pos_scores = self.module(gci_dataset[:], gci_name)

                # Generate and compute negative loss if applicable
                neg_data = self.generate_negatives(gci_name, gci_dataset)
                neg_scores = None
                if neg_data is not None:
                    neg_scores = self.module(neg_data, gci_name, neg=True)

                loss = loss + self.compute_loss(pos_scores, neg_scores)

            # Add regularization loss
            loss = loss + self.get_regularization_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            if epoch_callback is not None:
                epoch_callback(epoch, self)

            # Validation
            if (epoch + 1) % validate_every == 0:
                if self.dataset.validation is not None:
                    if self._eval_gci_name is None:
                        raise ValueError(
                            "eval_gci_name must be set before training with validation. "
                            "Set model.eval_gci_name to one of: 'gci0', 'gci1', 'gci2', 'gci3'"
                        )
                    with th.no_grad():
                        self.module.eval()
                        valid_loss = 0
                        gci_data = self.validation_datasets[self._eval_gci_name][:]
                        vloss = th.mean(self.module(gci_data, self._eval_gci_name))
                        valid_loss += vloss.detach().item()

                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            th.save(self.module.state_dict(), self.model_filepath)
                    print(f'Epoch {epoch+1}: Train loss: {train_loss} Valid loss: {valid_loss}')
                else:
                    print(f'Epoch {epoch+1}: Train loss: {train_loss}')

    def eval_method(self, data):
        """Evaluation method used for scoring. Override if needed.

        :param data: Input data for evaluation
        :type data: torch.Tensor
        :return: Evaluation scores
        :rtype: torch.Tensor
        :raises ValueError: If eval_gci_name has not been set
        """
        if self._eval_gci_name is None:
            raise ValueError(
                "eval_gci_name must be set before evaluation. "
                "Set model.eval_gci_name to one of: 'gci0', 'gci1', 'gci2', 'gci3'"
            )
        return self.module(data, self._eval_gci_name)

    @property
    def evaluation_model(self):
        """Returns the evaluation model for use with evaluators.

        If a custom evaluation model has been set via the setter, it is returned.
        Otherwise, for EL models, this returns the module which can be called
        with (data, gci_name). Requires eval_gci_name to be set in the latter case.

        :rtype: torch.nn.Module
        :raises ValueError: If no custom model is set and eval_gci_name has not been set
        """
        if self._evaluation_model is not None:
            return self._evaluation_model
        if self._eval_gci_name is None:
            raise ValueError(
                "eval_gci_name must be set before evaluation. "
                "Set model.eval_gci_name to one of: 'gci0', 'gci1', 'gci2', 'gci3'"
            )
        return self.module

    @evaluation_model.setter
    def evaluation_model(self, value):
        """Set a custom evaluation model.

        :param value: The custom evaluation model to use
        :type value: torch.nn.Module
        """
        self._evaluation_model = value

    def get_embeddings(self):
        """Get trained embeddings for entities, relations, and individuals.

        :return: Tuple of (entity_embeddings, relation_embeddings, individual_embeddings)
        :rtype: tuple
        """
        self.init_module()
        print('Load the best model', self.model_filepath)
        self.load_best_model()

        ent_embeds = {
            k: v for k, v in zip(self.class_index_dict.keys(),
                                 self.module.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {
            k: v for k, v in zip(self.object_property_index_dict.keys(),
                                 self.module.rel_embed.weight.cpu().detach().numpy())}
        if self.module.ind_embed is not None:
            ind_embeds = {
                k: v for k, v in zip(self.individual_index_dict.keys(),
                                     self.module.ind_embed.weight.cpu().detach().numpy())}
        else:
            ind_embeds = None
        return ent_embeds, rel_embeds, ind_embeds

    def load_best_model(self):
        """Load the best model from the model filepath."""
        self.init_module()
        self.module.load_state_dict(th.load(self.model_filepath, weights_only=True))
        self.module.eval()

    @versionadded(version="0.2.0")
    def score(self, axiom):
        """
        Returns the score of the given axiom.

        :param axiom: The axiom to score.
        :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
        """

        def data_point_to_tensor(data_point):
            data_point = th.tensor(data_point, dtype=th.long, device=self.device)
            data_point = data_point.unsqueeze(0)
            return data_point
        
        not_el_error_msg = "This axiom does not belong to the EL description logic specification."
        sub, super_ = axiom.getSubClass(), axiom.getSuperClass()

        if not isinstance(sub, OWLClassExpression):
            raise TypeError("Parameter sub must be of type OWLClassExpression.")

        if isinstance(sub, OWLClass):
            sub_id = self.dataset.class_to_id[sub]
            if isinstance(super_, OWLClass):
                super_id = self.dataset.class_to_id[super_]
                if super_.isOWLNothing():
                    if self.extended:
                        gci_name = "gci0_bot"
                    else:
                        gci_name = "gci0"
                else:
                    gci_name = "gci0"

                gci_data = data_point_to_tensor([sub_id, super_id])

            elif isinstance(super_, OWLObjectSomeValuesFrom):
                rel = super_.getProperty()
                filler = super_.getFiller()
                if not isinstance(filler, OWLClass):
                    raise TypeError(not_el_error_msg)
                
                rel_id = self.dataset.object_property_to_id[rel]
                filler_id = self.dataset.class_to_id[filler]
                gci_name = "gci2"
                gci_data = data_point_to_tensor([sub_id, rel_id, filler_id])
                
        elif isinstance(sub, OWLObjectSomeValuesFrom):
            rel = sub.getProperty()
            filler = sub.getFiller()
            if not isinstance(filler, OWLClass):
                raise TypeError(not_el_error_msg)
            if not isinstance(super_, OWLClass):
                raise TypeError(not_el_error_msg)

            rel_id = self.dataset.object_property_to_id[rel]
            filler_id = self.dataset.class_to_id[filler]
            super_id = self.dataset.class_to_id[super_]
            if super_.isOWLNothing():
                if self.extended:
                    gci_name = "gci3_bot"
                else:
                    gci_name = "gci3"
            else:
                gci_name = "gci3"
            
            gci_data = data_point_to_tensor([rel_id, filler_id, super_id])
            
        elif isinstance(sub, OWLObjectIntersectionOf):
            operands = sub.getOperandsAsList()
            if len(operands) != 2:
                raise TypeError(not_el_error_msg)
            left, right = tuple(operands)
            if not isinstance(left, OWLClass):
                raise TypeError(not_el_error_msg)
            if not isinstance(right, OWLClass):
                raise TypeError(not_el_error_msg)
            if not isinstance(super_, OWLClass):
                raise TypeError(not_el_error_msg)

            left_id = self.dataset.class_to_id[left]
            right_id = self.dataset.class_to_id[right]
            super_id = self.dataset.class_to_id[super_]

            if super_.isOWLNothing():
                if self.extended:
                    gci_name = "gci1_bot"
                else:
                    gci_name = "gci1"
            else:
                gci_name = "gci1"

            gci_data = data_point_to_tensor([left_id, right_id, super_id])
            
        else:
            raise TypeError("This axiom does not belong to EL.")

        
        score = self.module(gci_data, gci_name)
        return score


    @property
    def class_embeddings(self):
        class_embeds = {
            k: v for k, v in zip(self.class_index_dict.keys(),
                                 self.module.class_embed.weight.cpu().detach().numpy())}
        return class_embeds

    @property
    def object_property_embeddings(self):
        rel_embeds = {
            k: v for k, v in zip(self.object_property_index_dict.keys(),
                                 self.module.rel_embed.weight.cpu().detach().numpy())}
        
        return rel_embeds
        
    
    @property
    def individual_embeddings(self):
        if self.module.ind_embed is None:
            return dict()
        
        ind_embeds = {
            k: v for k, v in zip(self.individual_index_dict.keys(),
                                 self.module.ind_embed.weight.cpu().detach().numpy())}
        
        return ind_embeds
        


    def add_axioms(self, *axioms):
        prev_class_embeds = None
        prev_object_property_embeds = None
        prev_individual_embeds = None
        
        if len(self.class_embeddings) > 0:
            prev_class_embeds = copy.deepcopy(self.class_embeddings)

        if len(self.object_property_embeddings) > 0:
            prev_object_property_embeds = copy.deepcopy(self.object_property_embeddings)

        if len(self.individual_embeddings) > 0:
            prev_individual_embeds = copy.deepcopy(self.individual_embeddings)

        self.dataset.add_axioms(*axioms)

        # The new axioms change the signature of the ontology, and can change which
        # auxiliary concepts normalization introduces, so the datasets and the index
        # dictionaries built from them are stale. Dropping them makes the next read of
        # class_index_dict renormalize.
        self._datasets_loaded = False
        self._dataloaders_loaded = False
        self._el_class_index_dict = None
        self._el_object_property_index_dict = None

        # The rows are rebuilt in index-dictionary order rather than dataset.classes order,
        # so that they stay aligned with the vocabulary class_embeddings reads them back
        # through. The two differ by the auxiliary entities appended to the dictionaries.
        if prev_class_embeds is not None:
            new_class_embeds = []
            for cls in self.class_index_dict:
                if cls in prev_class_embeds:
                    new_class_embeds.append(prev_class_embeds[cls])
                else:
                    new_class_embeds.append(np.random.normal(size=self.embed_dim))

            new_class_embeds = np.asarray(new_class_embeds)
            self.module.class_embed.weight.data = th.from_numpy(new_class_embeds).float()

        if prev_object_property_embeds is not None:
            new_object_property_embeds = []
            for rel in self.object_property_index_dict:
                if rel in prev_object_property_embeds:
                    new_object_property_embeds.append(prev_object_property_embeds[rel])
                else:
                    new_object_property_embeds.append(np.random.normal(size=self.embed_dim))

            new_object_property_embeds = np.asarray(new_object_property_embeds)
            self.module.rel_embed.weight.data = th.from_numpy(new_object_property_embeds).float()

        if prev_individual_embeds is not None:
            new_individual_embeds = []
            for ind in self.dataset.individuals:
                ind = str(ind.toStringID())
                if ind in prev_individual_embeds:
                    new_individual_embeds.append(prev_individual_embeds[ind])
                else:
                    new_individual_embeds.append(np.random.normal(size=self.embed_dim))
            
            new_individual_embeds = np.asarray(new_individual_embeds)
            self.module.ind_embed.weight.data = th.from_numpy(new_individual_embeds).float()

            

    def from_pretrained(self, model):
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the model file.")

        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")

        #self._model_filepath = model

        
        self._is_pretrained = True
        if not isinstance(model, str):
            raise TypeError

        self.module.load_state_dict(th.load(model, weights_only=True))
        #self._kge_method = kge_method
    



    def load_pairwise_eval_data(self):

        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        head_classes, tail_classes = self.dataset.evaluation_classes
        self._head_entities = head_classes.as_str
        self._tail_entities = tail_classes.as_str
                        
        eval_projector = projector_factory('taxonomy_rels', taxonomy=False,
                                           relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True


    @property
    def training_set(self):
        self.load_pairwise_eval_data()
        return self._training_set

    @property
    def testing_set(self):
        self.load_pairwise_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        self.load_pairwise_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        self.load_pairwise_eval_data()
        return self._tail_entities
