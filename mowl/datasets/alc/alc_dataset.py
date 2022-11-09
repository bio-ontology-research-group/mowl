import torch as th
from torch.utils.data import DataLoader
from mowl.ontology.normalize import ELNormalizer, GCI
import random
from mowl.owlapi import (
    OWLOntology, OWLClass, OWLObjectProperty, OWLSubClassOfAxiom,
    OWLEquivalentClassesAxiom, OWLObjectSomeValuesFrom, ClassExpressionType,
    Imports
)
from mowl.owlapi.defaults import TOP
from mowl.owlapi.constants import R, THING
from mowl.owlapi.adapter import OWLAPIAdapter

class ALCDataset():
    """This class provides data-related methods to work with :math:`\mathcal{ALC}` description \
    logic language. In general, it receives an ontology, groups axioms by similar patterns \
    and returns a :class:`torch.utils.data.Dataset`. In the process, the classes and object properties names are mapped to an integer values \
    to create the datasets and the corresponding dictionaries can be input or created from scratch.

    :param ontology: Input ontology
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param class_index_dict: Dictionary containing information `class name --> index`. If not \
    provided, a dictionary will be created from the ontology classes. Defaults to ``None``.
    :type class_index_dict: dict, optional
    :param object_property_index_dict: Dictionary containing information `object property \
    name --> index`. If not provided, a dictionary will be created from the ontology object \
    properties. Defaults to ``None``.
    :type object_property_index_dict: dict, optional
    """

    def __init__(
        self,
        ontology,
        class_index_dict=None,
        object_property_index_dict=None,
        device="cpu"
    ):

        if not isinstance(ontology, OWLOntology):
            raise TypeError("Parameter ontology must be of type \
org.semanticweb.owlapi.model.OWLOntology.")

        if not isinstance(class_index_dict, dict) and class_index_dict is not None:
            raise TypeError("Optional parameter class_index_dict must be of type dict")

        obj = object_property_index_dict
        if not isinstance(obj, dict) and obj is not None:
            raise TypeError("Optional parameter object_property_index_dict must be of type dict")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str")

        self._ontology = ontology
        self._loaded = False
        self._class_index_dict = class_index_dict
        self._object_property_index_dict = object_property_index_dict
        self.device = device

        self.adapter = OWLAPIAdapter()
        self.thing = self.adapter.create_class(THING)
        self.r = self.adapter.create_object_property(R)

    def get_grouped_axioms(self):
        res = {}
        for axiom in self._ontology.getTBoxAxioms(Imports.INCLUDED):
            axiom_pattern = self.get_axiom_pattern(axiom)
            if axiom_pattern not in res:
                res[axiom_pattern] = [axiom,]
            else:
                res[axiom_pattern].append(axiom)
        return res


    def get_axiom_pattern(self, axiom):

        def get_cexpr_pattern(cexpr, index=0):
            expr_type = cexpr.getClassExpressionType()
            if expr_type == ClassExpressionType.OWL_CLASS:
                if index == 0:
                    return self.thing
                return self.adapter.create_class(THING + f'_{index}')
            elif expr_type == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
                return self.adapter.create_object_some_values_from(
                    self.r, get_cexpr_pattern(cexpr.getFiller()))
            elif expr_type == ClassExpressionType.OBJECT_ALL_VALUES_FROM:
                return self.adapter.create_object_all_values_from(
                    self.r, get_cexpr_pattern(cexpr.getFiller()))
            elif expr_type == ClassExpressionType.OBJECT_INTERSECTION_OF:
                cexprs = [get_cexpr_pattern(expr, index=i) for i, expr in enumerate(cexpr.getOperandsAsList())]
                return self.adapter.create_object_intersection_of(*cexprs)
            elif expr_type == ClassExpressionType.OBJECT_UNION_OF:
                cexprs = [get_cexpr_pattern(expr, index=i) for i, expr in enumerate(cexpr.getOperandsAsList())]
                return self.adapter.create_object_union_of(*cexprs)
            elif expr_type == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                return self.adapter.create_complement_of(
                    get_cexpr_pattern(cexpr.getOperand()))
            raise NotImplementedError()

        if isinstance(axiom, OWLSubClassOfAxiom):
            return self.adapter.create_subclass_of(
                get_cexpr_pattern(axiom.getSubClass()),
                get_cexpr_pattern(axiom.getSuperClass())
            )
        elif isinstance(axiom, OWLEquivalentClassesAxiom):
            cexprs = [get_cexpr_pattern(cexpr)
                      for cexpr in axiom.getClassExpressions()]
            return self.adapter.create_equivalent_classes(*cexprs)
        else:
            raise NotImplementedError()

    def get_axiom_vector(self, axiom):

        def get_cexpr_vector(cexpr):
            expr_type = cexpr.getClassExpressionType()
            if expr_type == ClassExpressionType.OWL_CLASS:
                return [self.class_index_dict[cexpr.asOWLClass()],]
            elif expr_type == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
                return ([self.object_property_index_dict[cexpr.getProperty()],]
                        + get_cexpr_vector(cexpr.getFiller()))
            elif expr_type == ClassExpressionType.OBJECT_ALL_VALUES_FROM:
                return ([self.object_property_index_dict[cexpr.getProperty()],]
                        + get_cexpr_vector(cexpr.getFiller()))
            elif expr_type == ClassExpressionType.OBJECT_INTERSECTION_OF:
                cexprs = []
                for expr in cexpr.getOperandsAsList():
                    cexprs += get_cexpr_vector(expr)
                return cexprs
            elif expr_type == ClassExpressionType.OBJECT_UNION_OF:
                cexprs = []
                for expr in cexpr.getOperandsAsList():
                    cexprs += get_cexpr_vector(expr)
                return cexprs
            elif expr_type == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                return get_cexpr_vector(cexpr.getOperand())
            raise NotImplementedError()

        if isinstance(axiom, OWLSubClassOfAxiom):
            return (get_cexpr_vector(axiom.getSubClass())
                    + get_cexpr_vector(axiom.getSuperClass()))
        elif isinstance(axiom, OWLEquivalentClassesAxiom):
            cexprs = []
            for cexpr in axiom.getClassExpressions():
                cexprs += get_cexpr_vector(cexpr)
            return cexprs
        else:
            raise NotImplementedError()


    def load(self):
        if self._loaded:
            return
        
        classes = self._ontology.getClassesInSignature()
        relations = self._ontology.getObjectPropertiesInSignature()

        if self._class_index_dict is None:
            self._class_index_dict = {v: k for k, v in enumerate(classes)}
        if self._object_property_index_dict is None:
            self._object_property_index_dict = {v: k for k, v in enumerate(relations)}

        self._grouped_axioms = self.get_grouped_axioms()
        
        self._loaded = True

    def get_datasets(self):
        """Returns a dictionary containing the name of the axiom pattern as keys and the \
        corresponding datasets as values.

        :rtype: dict
        """
        datasets = {}
        for ax_pattern, axioms in self.grouped_axioms.items():
            axiom_vectors = []
            for axiom in axioms:
                axiom_vectors.append(self.get_axiom_vector(axiom))
            datasets[ax_pattern] = axiom_vectors
        return datasets

    @property
    def grouped_axioms(self):
        """Returns a dictionary with grouped axioms where keys are axiom patterns and
        the values are lists of correspoding axioms for the patterns
        """
        self.load()
        return self._grouped_axioms
    
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

