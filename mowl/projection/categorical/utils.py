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

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False



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

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. All tensors must have the same size at dimension 0.
        :param batch_size: batch size to load. Defaults to 32.
        :type batch_size: int, optional
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object. Defaults to False.
        :type shuffle: bool, optional
        """

        # Type checking
        if not all(isinstance(t, th.Tensor) for t in tensors):
            raise TypeError("All non-optional parameters must be Tensors")

        if not isinstance(batch_size, int):
            raise TypeError("Optional parameter batch_size must be of type int")

        if not isinstance(shuffle, bool):
            raise TypeError("Optional parameter shuffle must be of type bool")

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = th.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


