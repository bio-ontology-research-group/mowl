from mowl.base_models.model import EmbeddingModel
import torch as th
from torch.utils.data import DataLoader, default_collate
from mowl.datasets.el import ELDataset


class EmbeddingELModel(EmbeddingModel):
    """Abstract class that provides basic functionalities for methods that aim to embed EL language.

    :param extended: If `True`, the model is supposed with 7 EL normal forms. This will be reflected on the :class:`DataLoaders` that will be generated and also the model must contain 7 loss functions. If `False`, the model will work with 4 normal forms only, merging the 3 extra to their corresponding origin normal forms. Defaults to True
    :type extended: bool, optional
    """
    
    def __init__(self, dataset, batch_size, extended = True, model_filepath = None, device = "cpu"):
        super().__init__(dataset, model_filepath = model_filepath)

        self._datasets_loaded = False
        self._dataloaders_loaded = False
        self._extended = extended
        self.batch_size = batch_size
        self.device = device

        self._training_datasets = None
        self._validation_datasets = None
        self._testing_datasets = None
        

    def _load_datasets(self):
        """This method will create different data attributes and finally the corresponding DataLoaders for each GCI type in each subset (training, validation and testing).
        """
        if self._datasets_loaded:
            return

        self._training_datasets = ELDataset(self.dataset.ontology, self.class_index_dict, self.object_property_index_dict, extended = self._extended, device = self.device)

        self._validation_datasets = None
        if self.dataset.validation:
            self._validation_datasets = ELDataset(self.dataset.validation, self.class_index_dict, self.object_property_index_dict, extended = self._extended, device = self.device)

        self._testing_datasets = None
        if self.dataset.testing:
            self._testing_datasets = ELDataset(self.dataset.testing, self.class_index_dict, self.object_property_index_dict, extended = self._extended, device = self.device)

        self._datasets_loaded = True

    def _load_dataloaders():
        if self._dataloaders_loaded:
            return

        self._load_datasets()
        
        self._training_dataloaders = {k: DataLoader(v, batch_size=self.batch_size, pin_memory = False) for k,v in self._training_datasets.get_gci_datasets().items()}

        if self._validation_datasets:
            self._validation_dataloaders = {k: DataLoader(v, batch_size=self.batch_size, pin_memory = False) for k,v in self._validation_datasets.get_gci_datasets().items()}

        if self._testing_datasets:
            self._testing_dataloaders = {k: DataLoader(v, batch_size=self.batch_size, pin_memory = False) for k,v in self._testing_datasets.get_gci_datasets().items()}

        self._dataloaders_loaded = True

    @property
    def training_datasets(self):
        self._load_datasets()
        return self._training_datasets

    @property
    def validation_datasets(self):
        if self.dataset.validation is None:
            raise AttributeError("Validation dataset is None.")

        self._load_datasets()
        return self._validation_datasets

    @property
    def testing_datasets(self):
        if self.dataset.testing is None:
            raise AttributeError("Testing dataset is None.")

        self._load_datasets()
        return self._testing_datasets
        
    @property
    def training_dataloaders(self):
        self._load_dataloaders()
        return self._training_dataloaders

    @property
    def validation_dataloaders(self):
        if self.dataset.validation is None:
            raise AttributeError("Validation dataset is None.")

        self._load_dataloaders()
        return self._validation_dataloaders

    @property
    def testing_dataloaders(self):
        if self.dataset.testing is None:
            raise AttributeError("Testing dataset is None.")

        self._load_dataloaders()
        return self._testing_dataloaders


    
        
from mowl.reasoning.normalize import ELNormalizer
import torch as th

class EmbeddingELModelOld(EmbeddingModel):

    def __init__(self, dataset):
        super().__init__(dataset)

        self._data_loaded = False
        self.extended = None

    def init_model(self):
        raise NotImplementedError()
    
    def load_best_model(self):
        self.load_data(extended = self.extended)
        self.init_model()
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()
    
    def gci0_loss(self):
        raise NotImplementedError()

    def gci0_bot_loss(self):
        raise NotImplementedError()

    def gci1_loss(self):
        raise NotImplementedError()

    def gci1_bot_loss(self):
        raise NotImplementedError()

    def gci2_loss(self):
        raise NotImplementedError()

    def gci3_loss(self):
        raise NotImplementedError()

    def gci3_bot_loss(self):
        raise NotImplementedError()


    def load_data(self, extended = True):
        if self._data_loaded:
            return

        normalizer = ELNormalizer()
        all_axioms = []
        self.training_axioms = normalizer.normalize(self.dataset.ontology)
        all_axioms.append(self.training_axioms)
        
        if not self.dataset.validation is None:
            self.validation_axioms = normalizer.normalize(self.dataset.validation)
            all_axioms.append(self.validation_axioms)
            
        if not self.dataset.testing is None:
            self.testing_axioms = normalizer.normalize(self.dataset.testing)
            all_axioms.append(self.testing_axioms)
            
        classes = set()
        relations = set()

        for axioms_dict in all_axioms:
            for axiom in axioms_dict["gci0"]:
                classes.add(axiom.subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci0_bot"]:
                classes.add(axiom.subclass)
                classes.add(axiom.superclass)
            
            for axiom in axioms_dict["gci1"]:
                classes.add(axiom.left_subclass)
                classes.add(axiom.right_subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci1_bot"]:
                classes.add(axiom.left_subclass)
                classes.add(axiom.right_subclass)
                classes.add(axiom.superclass)

            for axiom in axioms_dict["gci2"]:
                classes.add(axiom.subclass)
                classes.add(axiom.filler)
                relations.add(axiom.object_property)

            for axiom in axioms_dict["gci3"]:
                classes.add(axiom.superclass)
                classes.add(axiom.filler)
                relations.add(axiom.object_property)

            for axiom in axioms_dict["gci3_bot"]:
                classes.add(axiom.superclass)
                classes.add(axiom.filler)
                relations.add(axiom.object_property)

        self.classes_index_dict = {v: k  for k, v in enumerate(list(classes))}
        self.relations_index_dict = {v: k for k, v in enumerate(list(relations))}

        training_nfs = self.load_normal_forms(self.training_axioms, self.classes_index_dict, self.relations_index_dict, extended)
        validation_nfs = self.load_normal_forms(self.validation_axioms, self.classes_index_dict, self.relations_index_dict, extended)
        testing_nfs = self.load_normal_forms(self.testing_axioms, self.classes_index_dict, self.relations_index_dict, extended)
        
        self.train_nfs = self.gcis_to_tensors(training_nfs, self.device)
        self.valid_nfs = self.gcis_to_tensors(validation_nfs, self.device)
        self.test_nfs = self.gcis_to_tensors(testing_nfs, self.device)
        self._data_loaded = True

        
    def load_normal_forms(self, axioms_dict, classes_dict, relations_dict, extended = True):
        gci0 =     []
        gci0_bot = []
        gci1 =     []
        gci1_bot = []
        gci2 =     []
        gci3 =     []
        gci3_bot = []

        for axiom in axioms_dict["gci0"]:
            cl1 = classes_dict[axiom.subclass]
            cl2 = classes_dict[axiom.superclass]
            gci0.append((cl1, cl2))

        for axiom in axioms_dict["gci0_bot"]:
            cl1 = classes_dict[axiom.subclass]
            cl2 = classes_dict[axiom.superclass]
            if extended:
                gci0_bot.append((cl1, cl2))
            else:
                gci0.append((cl1, cl2))
                
        for axiom in axioms_dict["gci1"]:
            cl1 = classes_dict[axiom.left_subclass]
            cl2 = classes_dict[axiom.right_subclass]
            cl3 = classes_dict[axiom.superclass]
            gci1.append((cl1, cl2, cl3))

        for axiom in axioms_dict["gci1_bot"]:
            cl1 = classes_dict[axiom.left_subclass]
            cl2 = classes_dict[axiom.right_subclass]
            cl3 = classes_dict[axiom.superclass]
            if extended:
                gci1_bot.append((cl1, cl2, cl3))
            else:
                gci1_bot.append((cl1, cl2, cl3))
                
        for axiom in axioms_dict["gci2"]:
            cl1 = classes_dict[axiom.subclass]
            rel = relations_dict[axiom.object_property]
            cl2 = classes_dict[axiom.filler]
            gci2.append((cl1, rel, cl2))
        
        for axiom in axioms_dict["gci3"]:
            rel = relations_dict[axiom.object_property]
            cl1 = classes_dict[axiom.filler]
            cl2 = classes_dict[axiom.superclass]
            gci3.append((rel, cl1, cl2))

        for axiom in axioms_dict["gci3_bot"]:
            rel = relations_dict[axiom.object_property]
            cl1 = classes_dict[axiom.filler]
            cl2 = classes_dict[axiom.superclass]
            if extended:
                gci3_bot.append((rel, cl1, cl2))
            else:
                gci3.append((rel, cl1, cl2))
                
        return gci0, gci1, gci2, gci3, gci0_bot, gci1_bot,gci3_bot
        
    def gcis_to_tensors(self, gcis, device):
        gci0, gci1, gci2, gci3, gci0_bot, gci1_bot,gci3_bot = gcis

        gci0     = th.LongTensor(gci0).to(device)
        gci0_bot = th.LongTensor(gci0_bot).to(device)
        gci1     = th.LongTensor(gci1).to(device)
        gci1_bot = th.LongTensor(gci1_bot).to(device)
        gci2     = th.LongTensor(gci2).to(device)
        gci3     = th.LongTensor(gci3).to(device)
        gci3_bot = th.LongTensor(gci3_bot).to(device)
        gcis = gci0, gci1, gci2, gci3, gci0_bot, gci1_bot,gci3_bot
        return gcis

