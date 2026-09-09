import torch as th
from torch.utils.data import Dataset
from mowl.ontology.normalize import ELNormalizer, GCI, extract_role_axioms
from mowl.datasets.gci import GCIDataset, ClassAssertionDataset, ObjectPropertyAssertionDataset
import logging
import random
from org.semanticweb.owlapi.model import OWLOntology

logger = logging.getLogger(__name__)


class ELDataset():
    """This class provides data-related methods to work with :math:`\mathcal{EL}` description \
    logic language. In general, it receives an ontology, normalizes it into 4 or 7 \
    :math:`\mathcal{EL}` normal forms and returns a :class:`torch.utils.data.Dataset` per normal \
    form. In the process, the classes and object properties names are mapped to an integer values \
    to create the datasets and the corresponding dictionaries can be input or created from scratch.

    :param ontology: Input ontology that will be normalized into :math:`\mathcal{EL}` normal forms
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param extended: If true, the normalization process will return 7 normal forms. If false, \
    only 4 normal forms. See :doc:`/embedding_el/index` for more information. Defaults to \
    ``True``.
    :type extended: bool, optional
    :param class_index_dict: Dictionary containing information `class name --> index`. If not \
    provided, a dictionary will be created from the ontology classes. Defaults to ``None``.
    :type class_index_dict: dict, optional
    :param object_property_index_dict: Dictionary containing information `object property \
    name --> index`. If not provided, a dictionary will be created from the ontology object \
    properties. Defaults to ``None``.
    :type object_property_index_dict: dict, optional
    :param load_normalized: If true, the ontology is assumed to be already normalized and the \
    normalization process will be skipped. Defaults to ``False``.
    :type load_normalized: bool, optional
    :param ontology_path: Path to the original ontology file. If provided, the normalized \
    ontology will be cached to ``<ontology_path>_mowl_el_normalized.owl`` and loaded from cache \
    on subsequent calls. This significantly speeds up repeated normalization of the same \
    ontology. Defaults to ``None``.
    :type ontology_path: str, optional
    :param use_cache: Whether to use caching when ``ontology_path`` is provided. Defaults to \
    ``True``.
    :type use_cache: bool, optional
    :param load_role_axioms: Whether to also extract the two :math:`\\mathcal{EL}^{++}` role \
    axiom normal forms of the ontology -- role inclusions :math:`R \\sqsubseteq S` and role \
    chains :math:`R \\circ T \\sqsubseteq S` -- and expose them through \
    :meth:`get_gci_datasets` under the ``role_inclusion`` and ``role_chain`` keys. Only \
    :class:`ELModule <mowl.nn.ELModule>` subclasses that implement the corresponding losses \
    can consume them, so they are left out unless asked for. Defaults to ``False``.
    :type load_role_axioms: bool, optional
    """

    def __init__(self,
                 ontology,
                 class_index_dict=None,
                 object_property_index_dict=None,
                 individual_index_dict=None,
                 extended=True,
                 load_normalized=False,
                 device="cpu",
                 ontology_path=None,
                 use_cache=True,
                 load_role_axioms=False
                 ):

        if not isinstance(ontology, OWLOntology):
            raise TypeError("Parameter ontology must be of type \
org.semanticweb.owlapi.model.OWLOntology.")

        if not isinstance(class_index_dict, dict) and class_index_dict is not None:
            raise TypeError("Optional parameter class_index_dict must be of type dict")

        obj = object_property_index_dict
        if not isinstance(obj, dict) and obj is not None:
            raise TypeError("Optional parameter object_property_index_dict must be of type dict")

        ind = individual_index_dict
        if not isinstance(ind, dict) and ind is not None:
            raise TypeError("Optional parameter individual_index_dict must be of type dict")

        if not isinstance(extended, bool):
            raise TypeError("Optional parameter extended must be of type bool")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str")

        if ontology_path is not None and not isinstance(ontology_path, str):
            raise TypeError("Optional parameter ontology_path must be of type str")

        if not isinstance(use_cache, bool):
            raise TypeError("Optional parameter use_cache must be of type bool")

        if not isinstance(load_role_axioms, bool):
            raise TypeError("Optional parameter load_role_axioms must be of type bool")

        self._ontology = ontology
        self._loaded = False
        self._extended = extended
        self._class_index_dict = class_index_dict
        self._object_property_index_dict = object_property_index_dict
        self._individual_index_dict = individual_index_dict
        self.device = device
        self.load_normalized = load_normalized
        self.ontology_path = ontology_path
        self.use_cache = use_cache
        self.load_role_axioms = load_role_axioms
        self._role_inclusions = []
        self._role_chains = []
        
        self._gci0_dataset = None
        self._gci1_dataset = None
        self._gci2_dataset = None
        self._gci3_dataset = None
        self._gci0_bot_dataset = None
        self._gci1_bot_dataset = None
        self._gci3_bot_dataset = None
        self._class_assertion_dataset = None
        self._object_property_assertion_dataset = None
        self._role_inclusion_dataset = None
        self._role_chain_dataset = None

    def load(self):
        if self._loaded:
            return

        normalizer = ELNormalizer()

        gcis = normalizer.normalize(
            self._ontology,
            load=self.load_normalized,
            ontology_path=self.ontology_path,
            use_cache=self.use_cache
        )

        classes = set()
        relations = set()
        individuals = set()
        for k, v in gcis.items():
            new_classes, new_relations, new_individuals = GCI.get_entities(v)
            classes |= set(new_classes)
            relations |= set(new_relations)
            individuals |= set(new_individuals)

        # Role inclusions/chains are not EL GCIs, so the normalizer does not return them.
        # They are extracted directly from the ontology (EL++ language), and only when the
        # caller asked for them: scanning the axiom closure is not free, and no module can
        # train on them without implementing the two extra losses.
        if self.load_role_axioms:
            self._role_inclusions, self._role_chains = extract_role_axioms(self._ontology)

        if self._object_property_index_dict is None:
            # A new index dictionary is created below; make sure it covers the
            # properties appearing in the role axioms as well.
            _, role_relations, _ = GCI.get_entities(self._role_inclusions)
            relations |= role_relations
            _, role_relations, _ = GCI.get_entities(self._role_chains)
            relations |= role_relations

        classes = sorted(list(classes))
        relations = sorted(list(relations))
        individuals = sorted(list(individuals))

        
        if self._class_index_dict is None:
            self._class_index_dict = {v: k for k, v in enumerate(classes)}
        if self._object_property_index_dict is None:
            self._object_property_index_dict = {v: k for k, v in enumerate(relations)}
        if self._individual_index_dict is None:
            self._individual_index_dict = {v: k for k, v in enumerate(individuals)}

        if self._role_inclusions or self._role_chains:
            # An externally provided index dictionary may not cover all the properties
            # used by the role axioms (e.g. owl:topObjectProperty); drop the ones that
            # cannot be indexed instead of failing.
            indexed = self._object_property_index_dict.keys()
            self._role_inclusions = [
                role_inclusion for role_inclusion in self._role_inclusions
                if {role_inclusion.sub_property, role_inclusion.super_property} <= indexed
            ]
            self._role_chains = [
                role_chain for role_chain in self._role_chains
                if set(role_chain.sub_chain) | {role_chain.super_property} <= indexed
            ]

        # Only binary chains fit the (*, 3) tensor of RoleChainDataset. Chains are
        # dropped here, before the datasets are built, so that a dataset is never
        # created from an empty list of axioms.
        binary_chains = []
        for role_chain in self._role_chains:
            if len(role_chain.sub_chain) == 2:
                binary_chains.append(role_chain)
            else:
                logger.warning("Role chain of length %d is not supported (only chains of "
                               "two properties are representable) and will be ignored: %s",
                               len(role_chain.sub_chain), role_chain)
        self._role_chains = binary_chains

        if not self._extended:
            gci0 = gcis["gci0"] + gcis["gci0_bot"]
            gci1 = gcis["gci1"] + gcis["gci1_bot"]
            gci2 = gcis["gci2"]
            gci3 = gcis["gci3"] + gcis["gci3_bot"]

            random.shuffle(gci0)
            random.shuffle(gci1)
            random.shuffle(gci2)
            random.shuffle(gci3)

            self._gci0_dataset = GCI0Dataset(gci0, self._class_index_dict, device=self.device)
            self._gci1_dataset = GCI1Dataset(gci1, self._class_index_dict, device=self.device)
            self._gci2_dataset = GCI2Dataset(
                gci2, self._class_index_dict,
                object_property_index_dict=self._object_property_index_dict, device=self.device)
            self._gci3_dataset = GCI3Dataset(
                gci3, self._class_index_dict,
                object_property_index_dict=self._object_property_index_dict, device=self.device)
        else:
            gci0 = gcis["gci0"]
            gci0_bot = gcis["gci0_bot"]
            gci1 = gcis["gci1"]
            gci1_bot = gcis["gci1_bot"]
            gci2 = gcis["gci2"]
            gci3 = gcis["gci3"]
            gci3_bot = gcis["gci3_bot"]

            random.shuffle(gci0)
            random.shuffle(gci0_bot)
            random.shuffle(gci1)
            random.shuffle(gci1_bot)
            random.shuffle(gci2)
            random.shuffle(gci3)
            random.shuffle(gci3_bot)

            self._gci0_dataset = GCI0Dataset(gci0, self._class_index_dict, device=self.device)
            self._gci0_bot_dataset = GCI0Dataset(
                gci0_bot, self._class_index_dict,
                device=self.device)
            self._gci1_dataset = GCI1Dataset(gci1, self._class_index_dict, device=self.device)
            self._gci1_bot_dataset = GCI1Dataset(
                gci1_bot, self._class_index_dict,
                device=self.device)
            self._gci2_dataset = GCI2Dataset(
                gci2, self._class_index_dict,
                object_property_index_dict=self._object_property_index_dict,
                device=self.device)
            self._gci3_dataset = GCI3Dataset(
                gci3, self._class_index_dict,
                object_property_index_dict=self._object_property_index_dict,
                device=self.device)
            self._gci3_bot_dataset = GCI3Dataset(
                gci3_bot, self._class_index_dict,
                object_property_index_dict=self._object_property_index_dict,
                device=self.device)

        if len(gcis["class_assertion"]) > 0:
            gci_class_assertion = gcis["class_assertion"]
            random.shuffle(gci_class_assertion)
            self._class_assertion_dataset = ClassAssertionDataset(
                gci_class_assertion, self._class_index_dict, self._individual_index_dict, device=self.device)

        if len(gcis["object_property_assertion"]) > 0:
            gci_object_property_assertion = gcis["object_property_assertion"]
            random.shuffle(gci_object_property_assertion)
            self._object_property_assertion_dataset = ObjectPropertyAssertionDataset(
                gci_object_property_assertion, self._object_property_index_dict, self._individual_index_dict, device=self.device)

        if len(self._role_inclusions) > 0:
            random.shuffle(self._role_inclusions)
            self._role_inclusion_dataset = RoleInclusionDataset(
                self._role_inclusions,
                object_property_index_dict=self._object_property_index_dict,
                device=self.device)

        if len(self._role_chains) > 0:
            random.shuffle(self._role_chains)
            self._role_chain_dataset = RoleChainDataset(
                self._role_chains,
                object_property_index_dict=self._object_property_index_dict,
                device=self.device)

        self._loaded = True

    def get_gci_datasets(self):
        """Returns a dictionary containing the name of the normal forms as keys and the \
        corresponding datasets as values.

        The GCI datasets are always present: 7 of them if the class parameter `extended` is \
        ``True``, otherwise 4. The ``class_assertion`` and ``object_property_assertion`` keys \
        are added when the ontology contains axioms of those forms, and the \
        ``role_inclusion`` and ``role_chain`` keys when it contains role axioms *and* the \
        dataset was built with ``load_role_axioms=True``.

        :rtype: dict
        """
        datasets = {
            "gci0": self.gci0_dataset,
            "gci1": self.gci1_dataset,
            "gci2": self.gci2_dataset,
            "gci3": self.gci3_dataset
        }

        if self._extended:
            datasets["gci0_bot"] = self.gci0_bot_dataset
            datasets["gci1_bot"] = self.gci1_bot_dataset
            datasets["gci3_bot"] = self.gci3_bot_dataset

        if self.class_assertion_dataset is not None:
            datasets["class_assertion"] = self.class_assertion_dataset

        if self.object_property_assertion_dataset is not None:
            datasets["object_property_assertion"] = self.object_property_assertion_dataset

        if self.role_inclusion_dataset is not None:
            datasets["role_inclusion"] = self.role_inclusion_dataset

        if self.role_chain_dataset is not None:
            datasets["role_chain"] = self.role_chain_dataset

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
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        self.load()
        return self._gci0_bot_dataset

    @property
    def gci1_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        self.load()
        return self._gci1_bot_dataset

    @property
    def gci3_bot_dataset(self):
        if not self._extended:
            raise AttributeError("Extended normal forms do not exist because `extended` parameter \
                was set to False")

        self.load()
        return self._gci3_bot_dataset


    @property
    def class_assertion_dataset(self):
        self.load()
        return self._class_assertion_dataset

    @property
    def object_property_assertion_dataset(self):
        return self._object_property_assertion_dataset

    @property
    def role_inclusion_dataset(self):
        """Returns the dataset with the role inclusion axioms :math:`R \\sqsubseteq S`, if any.

        :rtype: :class:`RoleInclusionDataset` or ``None``
        """
        self.load()
        return self._role_inclusion_dataset

    @property
    def role_chain_dataset(self):
        """Returns the dataset with the role chain axioms :math:`R \\circ T \\sqsubseteq S`, \
        if any.

        :rtype: :class:`RoleChainDataset` or ``None``
        """
        self.load()
        return self._role_chain_dataset
    
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


class RoleAxiomDataset(Dataset):
    """Base class for the datasets of the :math:`\\mathcal{EL}^{++}` role axiom normal forms.

    These axioms index object properties only, so -- unlike the ``GCIDataset`` of the \
    concept normal forms -- no class index dictionary is involved. Subclasses declare the \
    number of columns of the data tensor in \
    :attr:`arity` and how to read them off an axiom in :meth:`get_indices`.

    :param data: List of role axioms
    :type data: list
    :param object_property_index_dict: Dictionary mapping object property IRIs to indices
    :type object_property_index_dict: dict
    :param device: Device to push the data tensor to, defaults to ``"cpu"``
    :type device: str, optional
    """

    #: Number of object property columns of the data tensor.
    arity = None

    def __init__(self, data, object_property_index_dict, device="cpu"):
        super().__init__()
        self.object_property_index_dict = object_property_index_dict
        self.device = device
        self._data = self.push_to_device(data)

    @property
    def data(self):
        return self._data

    def get_indices(self, axiom):
        """Returns the object property indices of one axiom, in tensor column order.

        :rtype: list(int)
        """
        raise NotImplementedError()

    def push_to_device(self, data):
        pretensor = [self.get_indices(axiom) for axiom in data]
        # reshape keeps the (0, arity) shape of an empty dataset, which th.tensor([])
        # would otherwise collapse to (0,).
        tensor = th.tensor(pretensor, dtype=th.long).reshape(-1, self.arity)
        return tensor.to(self.device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class RoleInclusionDataset(RoleAxiomDataset):
    """Dataset of role inclusion axioms :math:`R \\sqsubseteq S`.

    The data tensor has shape :math:`(*, 2)`, with :math:`R` at ``[:,0]`` and :math:`S` at \
    ``[:,1]``.
    """

    arity = 2

    def get_indices(self, axiom):
        return [self.object_property_index_dict[axiom.sub_property],
                self.object_property_index_dict[axiom.super_property]]


class RoleChainDataset(RoleAxiomDataset):
    """Dataset of role chain axioms :math:`R \\circ T \\sqsubseteq S` (transitive property \
    declarations are represented as :math:`R \\circ R \\sqsubseteq R`).

    The data tensor has shape :math:`(*, 3)`, with :math:`R` at ``[:,0]``, :math:`T` at \
    ``[:,1]`` and :math:`S` at ``[:,2]``.
    """

    arity = 3

    def get_indices(self, axiom):
        sub_property, filler_property = axiom.sub_chain
        return [self.object_property_index_dict[sub_property],
                self.object_property_index_dict[filler_property],
                self.object_property_index_dict[axiom.super_property]]



