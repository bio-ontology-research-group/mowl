import torch
from torch.utils.data import TensorDataset
from mowl.owlapi import (
    OWLOntology, OWLClass, OWLObjectProperty, OWLSubClassOfAxiom,
    OWLEquivalentClassesAxiom, OWLObjectSomeValuesFrom, ClassExpressionType,
    Imports, OWLNaryAxiom, OWLDisjointClassesAxiom, OWLClassAssertionAxiom,
    OWLObjectPropertyAssertionAxiom

)
from mowl.owlapi.defaults import TOP
from mowl.owlapi.constants import R, THING, INDIVIDUAL
from mowl.owlapi.adapter import OWLAPIAdapter


import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ALCDataset():
    """This class provides data-related methods to work with
    :math:`\mathcal{ALC}` description logic language. 

    In general, it
    receives an ontology, groups axioms by similar patterns and
    returns a :class:`torch.utils.data.Dataset`. In the process, the
    classes and object properties names are mapped to an integer
    values  to create the datasets and the corresponding dictionaries
    can be input or created from scratch.

    .. warning::

        This class is on development.

    :param ontology: Input ontology
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param class_index_dict: Dictionary containing information `class \
    name --> index`. If not provided, a dictionary will be created \
    from the ontology classes. Defaults to ``None``.
    :type class_index_dict: dict, optional
    :param object_property_index_dict: Dictionary containing \
    information `object property name --> index`. If not provided, a \
    dictionary will be created from the ontology object \
    properties. Defaults to ``None``.
    :type object_property_index_dict: dict, optional
    """

    def __init__(self, ontology, dataset, device="cpu"):

        if not isinstance(ontology, OWLOntology):
            raise TypeError(
                "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology.")

        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str")

        self._ontology = ontology
        self._dataset = dataset
        self._loaded = False
        self.device = device

        self.adapter = OWLAPIAdapter()
        self.thing = self.adapter.create_class(THING)
        self.r = self.adapter.create_object_property(R)
        self.ind = self.adapter.create_individual(INDIVIDUAL)
        self.obj_prop_assertion_pat = self.adapter.create_object_property_assertion(
            self.r, self.ind, self.ind)

    @property
    def class_to_id(self):
        return self._dataset.class_to_id

    @property
    def individual_to_id(self):
        return self._dataset.individual_to_id

    @property
    def object_property_to_id(self):
        return self._dataset.object_property_to_id

    def get_grouped_axioms(self):
        res = dict()
        for axiom in self._ontology.getAxioms(Imports.INCLUDED):
            axioms = [axiom, ]
            if isinstance(axiom, OWLNaryAxiom):
                axioms = axiom.asPairwiseAxioms()
            for ax in axioms:
                axiom_pattern = self.get_axiom_pattern(axiom)
                if axiom_pattern is None:
                    continue
                if axiom_pattern not in res:
                    res[axiom_pattern] = [ax, ]
                else:
                    res[axiom_pattern].append(ax)
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
                    self.r, get_cexpr_pattern(cexpr.getFiller(), index=index))
            elif expr_type == ClassExpressionType.OBJECT_ALL_VALUES_FROM:
                return self.adapter.create_object_all_values_from(
                    self.r, get_cexpr_pattern(cexpr.getFiller(), index=index))
            elif expr_type == ClassExpressionType.OBJECT_INTERSECTION_OF:
                cexprs = [get_cexpr_pattern(expr, index=i)
                          for i, expr in enumerate(cexpr.getOperandsAsList())]
                return self.adapter.create_object_intersection_of(*cexprs)
            elif expr_type == ClassExpressionType.OBJECT_UNION_OF:
                cexprs = [get_cexpr_pattern(expr, index=i)
                          for i, expr in enumerate(cexpr.getOperandsAsList())]
                return self.adapter.create_object_union_of(*cexprs)
            elif expr_type == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                return self.adapter.create_complement_of(
                    get_cexpr_pattern(cexpr.getOperand(), index=index))
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
        elif isinstance(axiom, OWLDisjointClassesAxiom):
            cexprs = [get_cexpr_pattern(cexpr)
                      for cexpr in axiom.getClassExpressions()]
            return self.adapter.create_disjoint_classes(*cexprs)
        elif isinstance(axiom, OWLClassAssertionAxiom):
            cexpr = get_cexpr_pattern(axiom.getClassExpression())
            return self.adapter.create_class_assertion(cexpr, self.ind)
        elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            return self.obj_prop_assertion_pat

        return None

    def get_axiom_vector(self, axiom):

        def get_cexpr_vector(cexpr):
            expr_type = cexpr.getClassExpressionType()
            if expr_type == ClassExpressionType.OWL_CLASS:
                return [self.class_to_id[cexpr.asOWLClass()], ]
            elif expr_type == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
                return [self.object_property_to_id[cexpr.getProperty()], ] \
                    + get_cexpr_vector(cexpr.getFiller())
            elif expr_type == ClassExpressionType.OBJECT_ALL_VALUES_FROM:
                return [self.object_property_to_id[cexpr.getProperty()], ] \
                    + get_cexpr_vector(cexpr.getFiller())
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
            return get_cexpr_vector(axiom.getSubClass()) \
                + get_cexpr_vector(axiom.getSuperClass())
        elif isinstance(axiom, OWLEquivalentClassesAxiom):
            cexprs = []
            for cexpr in axiom.getClassExpressions():
                cexprs += get_cexpr_vector(cexpr)
            return cexprs
        elif isinstance(axiom, OWLDisjointClassesAxiom):
            cexprs = []
            for cexpr in axiom.getClassExpressions():
                cexprs += get_cexpr_vector(cexpr)
            return cexprs
        elif isinstance(axiom, OWLClassAssertionAxiom):
            ind = axiom.getIndividual()
            vector = [self.individual_to_id[ind], ]
            vector += get_cexpr_vector(axiom.getClassExpression())
            return vector
        elif isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            ind1 = axiom.getObject()
            ind2 = axiom.getSubject()
            prop = axiom.getProperty()
            return [self.individual_to_id[ind1],
                    self.object_property_to_id[prop],
                    self.individual_to_id[ind2]]
        return None

    def load(self):
        if self._loaded:
            return

        self._grouped_axioms = self.get_grouped_axioms()

        self._loaded = True

    def get_datasets(self, min_count=0):
        """Returns a dictionary containing the name of the axiom
        pattern as keys and the corresponding TensorDatasets as values. If the number of axioms for a given pattern is less than min_count, the pattern is not included in the dictionary and axioms is 

        :param min_count: The minimum number of occurrences of an axiom pattern.
        :type min_count: int
        :rtype: tuple(dict, list)
        """
        datasets = {}
        rest_of_axioms = []
        for ax_pattern, axioms in self.grouped_axioms.items():
            if len(axioms) < min_count:
                rest_of_axioms += axioms
                logger.debug(f"Skipping {ax_pattern} with {len(axioms)} axioms")
            else:
                logger.debug(f"Creating dataset for {ax_pattern} with {len(axioms)} axioms")
                axiom_vectors = []
                for axiom in axioms:
                    vector = self.get_axiom_vector(axiom)
                    axiom_vectors.append(vector)
                axiom_tensor = torch.tensor(axiom_vectors, dtype=torch.int64)
                datasets[ax_pattern] = TensorDataset(axiom_tensor)

        return datasets, rest_of_axioms

    @property
    def grouped_axioms(self):
        """Returns a dictionary with grouped axioms where keys are
        axiom patterns and the values are lists of correspoding axioms
        for the patterns """
        self.load()
        return self._grouped_axioms

    def get_obj_prop_assertion_data(self):
        return self.get_datasets()[self.obj_prop_assertion_pat]
