import torch as th
from torch.utils.data import DataLoader
from mowl.reasoning.normalize import ELNormalizer, GCI
from mowl.datasets.gci import GCIDataset
import random

class ELDataset():
    """This class provides data-related methods to work with :math:`\mathcal{EL}` description logic language. In general, it receives an ontology, normalizes it into 4 or 7 :math:`\mathcal{EL}` normal forms and returns a :class:`torch.utils.data.Dataset` per normal form. In the process, the classes and object properties names are mapped to an integer values to create the datasets and the corresponding dictionaries can be input or created from scratch.

    :param ontology: Input ontology that will be normalized into :math:`\mathcal{EL}` normal forms
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param extended: If true, the normalization process will return 7 normal forms. If false, only 4 normal forms. See :doc:`/embedding_el/index` for more information. Defaults to ``True``.
    :type extended: bool, optional
    :param class_index_dict: Dictionary containing information `class name --> index`. If not provided, a dictionary will be created from the ontology classes. Defaults to ``None``.
    :type class_index_dict: dict, optional
    :param object_property_index_dict: Dictionary containing information `object property name --> index`. If not provided, a dictionary will be created from the ontology object properties. Defaults to ``None``.
    :type object_property_index_dict: dict, optional
    """
    
    def __init__(self, ontology, class_index_dict = None, object_property_index_dict = None, extended = True, device = "cpu"):
        self._ontology = ontology
        self._loaded = False
        self._extended = extended
        self._class_index_dict = class_index_dict
        self._object_property_index_dict = object_property_index_dict
        self.device = device

        self._gci0_dataset = None
        self._gci1_dataset = None
        self._gci2_dataset = None
        self._gci3_dataset = None
        self._gci0_bot_dataset = None
        self._gci1_bot_dataset = None
        self._gci3_bot_dataset = None

    def load(self):
        if self._loaded:
            return
        
        normalizer = ELNormalizer()
        
        gcis = normalizer.normalize(self._ontology)

        classes = set()
        relations = set()
        for k, v in gcis.items():
            new_classes, new_relations = GCI.get_entities(v)
            classes |= set(new_classes)
            relations |= set(new_relations)

        if self._class_index_dict is None:
            self._class_index_dict = {v:k for k,v in enumerate(classes)}
        if self._object_property_index_dict is None:
            self._object_property_index_dict = {v:k for k,v in enumerate(relations)}

        if not self._extended:
            gci0 = gcis["gci0"] + gcis["gci0_bot"]
            gci1 = gcis["gci1"] + gcis["gci1_bot"]
            gci2 = gcis["gci2"]
            gci3 = gcis["gci3"] + gcis["gci3_bot"]
        
            random.shuffle(gci0)
            random.shuffle(gci1)
            random.shuffle(gci2)
            random.shuffle(gci3)

            self._gci0_dataset = GCI0Dataset(gci0, self._class_index_dict, device = self.device)
            self._gci1_dataset = GCI1Dataset(gci1, self._class_index_dict, device = self.device)
            self._gci2_dataset = GCI2Dataset(gci2, self._class_index_dict, object_property_index_dict = self._object_property_index_dict, device = self.device)
            self._gci3_dataset = GCI3Dataset(gci3, self._class_index_dict, object_property_index_dict = self._object_property_index_dict, device = self.device)
        else:
            gci0     = gcis["gci0"]
            gci0_bot = gcis["gci0_bot"]
            gci1     = gcis["gci1"]
            gci1_bot = gcis["gci1_bot"]
            gci2     = gcis["gci2"]
            gci3     = gcis["gci3"]
            gci3_bot = gcis["gci3_bot"]
        
            random.shuffle(gci0)
            random.shuffle(gci0_bot)
            random.shuffle(gci1)
            random.shuffle(gci1_bot)
            random.shuffle(gci2)
            random.shuffle(gci3)
            random.shuffle(gci3_bot)

            self._gci0_dataset     = GCI0Dataset(gci0, self._class_index_dict, device = self.device)
            self._gci0_bot_dataset = GCI0Dataset(gci0_bot, self._class_index_dict, device = self.device)
            self._gci1_dataset     = GCI1Dataset(gci1, self._class_index_dict, device = self.device)
            self._gci1_bot_dataset = GCI1Dataset(gci1_bot, self._class_index_dict, device = self.device)
            self._gci2_dataset     = GCI2Dataset(gci2, self._class_index_dict, object_property_index_dict = self._object_property_index_dict, device = self.device)
            self._gci3_dataset     = GCI3Dataset(gci3, self._class_index_dict, object_property_index_dict = self._object_property_index_dict, device = self.device)
            self._gci3_bot_dataset = GCI3Dataset(gci3_bot, self._class_index_dict, object_property_index_dict = self._object_property_index_dict, device = self.device)

        self._loaded = True
            
    def get_gci_datasets(self):
        """Returns a dictionary containing the name of the normal forms as keys and the corresponding datasets as values. This method will return 7 datasets if the class parameter `extended` is True, otherwise it will return only 4 datasets.

        :rtype: dict
        """
        datasets = {
            "gci0"    : self.gci0_dataset,
            "gci1"    : self.gci1_dataset,
            "gci2"    : self.gci2_dataset,
            "gci3"    : self.gci3_dataset
        }

        if self._extended:
            datasets["gci0_bot"] = self.gci0_bot_dataset
            datasets["gci1_bot"] = self.gci1_bot_dataset
            datasets["gci3_bot"] = self.gci3_bot_dataset

        return datasets
    
    @property
    def class_index_dict(self):
        """Returns indexed dictionary with class names present in the dataset.
        
        :rtype: dict
        """
        self.load()
        return self._class_index_dict

    @property
    def object_property_index_dict(self):
        """Returns indexed dictionary with object property names present in the dataset.
        
        :rtype: dict
        """

        self.load()
        return self._object_property_index_dict

    @property
    def gci0_dataset(self):
        self.load()
        return self._gci0_dataset

    @property
    def gci1_dataset(self):
        self.load()
        return self._gci1_dataset

    @property
    def gci2_dataset(self):
        self.load()
        return self._gci2_dataset

    @property
    def gci3_dataset(self):
        self.load()
        return self._gci3_dataset

    @property
    def gci0_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter was set to False")
        
        self.load()
        return self._gci0_bot_dataset


    @property
    def gci1_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter was set to False")
        
        self.load()
        return self._gci1_bot_dataset
    
    @property
    def gci3_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter was set to False")
        
        self.load()
        return self._gci3_bot_dataset

    
        
class GCI0Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push_to_device(self, data):
        pretensor = []
        for gci in data:
            subclass = self.class_index_dict[gci.subclass]
            superclass = self.class_index_dict[gci.superclass]
            pretensor.append([subclass, superclass])
        tensor = th.tensor(pretensor).to(self.device)
        return tensor
        
    def get_data_(self):
        for gci in self.data:
            subclass = self.class_index_dict[gci.subclass]
            superclass = self.class_index_dict[gci.superclass]
            yield subclass, superclass

class GCI1Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push_to_device(self, data):
        pretensor = []
        for gci in data:
            left_subclass = self.class_index_dict[gci.left_subclass]
            right_subclass = self.class_index_dict[gci.right_subclass]
            superclass = self.class_index_dict[gci.superclass]
            pretensor.append([left_subclass, right_subclass, superclass])

        tensor = th.tensor(pretensor).to(self.device)
        return tensor
    
    def get_data_(self):
        for gci in self.data:
            left_subclass = self.class_index_dict[gci.left_subclass]
            right_subclass = self.class_index_dict[gci.right_subclass]
            superclass = self.class_index_dict[gci.superclass]
            yield left_subclass, right_subclass, superclass

class GCI2Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push_to_device(self, data):
        pretensor = []
        for gci in data:
            subclass = self.class_index_dict[gci.subclass]
            object_property = self.object_property_index_dict[gci.object_property]
            filler = self.class_index_dict[gci.filler]
            pretensor.append([subclass, object_property, filler])
        tensor = th.tensor(pretensor).to(self.device)
        return tensor
        
    def get_data_(self):
        for gci in self.data:
            subclass = self.class_index_dict[gci.subclass]
            object_property = self.object_property_index_dict[gci.object_property]
            filler = self.class_index_dict[gci.filler]
            yield subclass, object_property, filler
        
class GCI3Dataset(GCIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def push_to_device(self, data):
        pretensor = []
        for gci in data:
            object_property = self.object_property_index_dict[gci.object_property]
            filler = self.class_index_dict[gci.filler]
            superclass = self.class_index_dict[gci.superclass]
            pretensor.append([object_property, filler, superclass])
        tensor = th.tensor(pretensor).to(self.device)
        return tensor
            
    def get_data_(self):
        for gci in self.data:
            object_property = self.object_property_index_dict[gci.object_property]
            filler = self.class_index_dict[gci.filler]
            superclass = self.class_index_dict[gci.superclass]
            yield object_property, filler, superclass

