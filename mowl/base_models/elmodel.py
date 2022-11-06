from mowl.ontology.normalize import ELNormalizer
from mowl.base_models.model import EmbeddingModel
import torch as th
from torch.utils.data import DataLoader, default_collate
from mowl.datasets.el import ELDataset


class EmbeddingELModel(EmbeddingModel):
    """Abstract class that provides basic functionalities for methods that aim to embed EL \
    language.

    :param extended: If `True`, the model is supposed with 7 EL normal forms. This will be \
reflected on the :class:`DataLoaders` that will be generated and also the model must \
    contain 7 loss functions. If `False`, the model will work with 4 normal forms only, \
merging the 3 extra to their corresponding origin normal forms. Defaults to True
    :type extended: bool, optional
    """

    def __init__(self, dataset, batch_size, extended=True, model_filepath=None, device="cpu"):
        super().__init__(dataset, model_filepath=model_filepath)

        if not isinstance(batch_size, int):
            raise TypeError("Parameter batch_size must be of type int.")

        if not isinstance(extended, bool):
            raise TypeError("Optional parameter extended must be of type bool.")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str.")

        self._datasets_loaded = False
        self._dataloaders_loaded = False
        self._extended = extended
        self.batch_size = batch_size
        self.device = device

        self._training_datasets = None
        self._validation_datasets = None
        self._testing_datasets = None

    def _load_datasets(self):
        """This method will create different data attributes and finally the corresponding \
            DataLoaders for each GCI type in each subset (training, validation and testing).
        """
        if self._datasets_loaded:
            return

        training_el_dataset = ELDataset(self.dataset.ontology, self.class_index_dict,
                                        self.object_property_index_dict,
                                        extended=self._extended, device=self.device)

        self._training_datasets = training_el_dataset.get_gci_datasets()

        self._validation_datasets = None
        if self.dataset.validation:
            validation_el_dataset = ELDataset(self.dataset.validation, self.class_index_dict,
                                              self.object_property_index_dict,
                                              extended=self._extended, device=self.device)

            self._validation_datasets = validation_el_dataset.get_gci_datasets()

        self._testing_datasets = None
        if self.dataset.testing:
            testing_el_dataset = ELDataset(self.dataset.testing, self.class_index_dict,
                                           self.object_property_index_dict,
                                           extended=self._extended, device=self.device)

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
