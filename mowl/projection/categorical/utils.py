import torch as th
import random
import os
import numpy as np
from itertools import chain, combinations, product

from org.semanticweb.owlapi.model import AxiomType as ax
from org.semanticweb.owlapi.model import ClassExpressionType as ct


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def pairs(iterable):
    num_items = len(iterable)
    power_set = list(powerset(iterable))
    product_set = list(product(power_set, power_set))

    curated_set = []
    for i1, i2 in product_set:
        if i1 == i2:
            continue
        if len(i1) + len(i2) != num_items:
            continue
        if len(i1) == 0 or len(i1) == num_items:
            continue
        if len(i2) == 0 or len(i2) == num_items:
            continue
        curated_set.append((i1, i2))

    return curated_set


IGNORED_AXIOM_TYPES = [ax.ANNOTATION_ASSERTION,
                       ax.ASYMMETRIC_OBJECT_PROPERTY,
                       ax.DECLARATION,
                       ax.EQUIVALENT_OBJECT_PROPERTIES,
                       ax.FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_OBJECT_PROPERTIES,
                       ax.IRREFLEXIVE_OBJECT_PROPERTY,
                       ax.OBJECT_PROPERTY_DOMAIN,
                       ax.OBJECT_PROPERTY_RANGE,
                       ax.REFLEXIVE_OBJECT_PROPERTY,
                       ax.SUB_PROPERTY_CHAIN_OF,
                       ax.SUB_ANNOTATION_PROPERTY_OF,
                       ax.SUB_OBJECT_PROPERTY,
                       ax.SWRL_RULE,
                       ax.SYMMETRIC_OBJECT_PROPERTY,
                       ax.TRANSITIVE_OBJECT_PROPERTY
                       ]

IGNORED_EXPRESSION_TYPES = [ct.OBJECT_EXACT_CARDINALITY,
                            ct.OBJECT_MIN_CARDINALITY,
                            ct.OBJECT_HAS_SELF,
                            ct.OBJECT_HAS_VALUE,
                            ct.OBJECT_ONE_OF,
                            ct.DATA_EXACT_CARDINALITY,
                            ct.DATA_MIN_CARDINALITY,
                            ct.DATA_HAS_VALUE,
                            ct.DATA_SOME_VALUES_FROM,
                            ct.DATA_MAX_CARDINALITY,
                            ct.OBJECT_MAX_CARDINALITY,
                            ct.DATA_ALL_VALUES_FROM
                            ]

 
