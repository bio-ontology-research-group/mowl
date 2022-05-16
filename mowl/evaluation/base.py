import numpy as np
import torch as th

import torch.nn as nn
import click as ck
from mowl.projection.edge import Edge

from mowl.projection.factory import projector_factory
from mowl.datasets.build_ontology import PREFIXES
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import logging
from tqdm import tqdm

class Evaluator():
    """
    Abstract class for evaluation of models.
    
    :param class_embeddings: Embeddings dictionary for ontology classes
    :type class_embeddings: dict or :class:`gensim.models.keyedvectors.KeyedVectors`
    :param testing_set: List of triples in the testing set.
    :type testing_set: list of :class:`mowl.projection.Edge`
    :param eval_method: Function that computes the score of the predictions
    :type eval_method: callable
    :param relation_embeddings: Embeddings dictionary for ontology classes
    :type relation_embeddings: dict or :class:`gensim.models.keyedvectors.KeyedVectors`, optional
    :param training_set: List of triples in the training set. If not set, filtered metrics will not be computed
    :type training_set: list of :class:`mowl.projection.Edge`, optional
    :param head_entities: Entities, that are the head of each triple, to be considered in the evaluation 
    :type head_entities: list of str
    :param filter_fn_head: Criterion to filter the head entities
    :type filter_fn_head: callable, optional
    :param filter_fn_tail: Criterion to filter the tail entities
    :type filter_fn_tail: callable, optional
    """

    def __init__(self,
                 device = "cpu"
                 ):
        self.device = device
        
    def embeddings_to_dict(self, embeddings):
        embeddings_dict = dict()
        if isinstance(embeddings, KeyedVectors):
            for idx, word in enumerate(embeddings.index_to_key):
                embeddings_dict[word] = embeddings[word]
        elif isinstance(embeddings, dict):
            embeddings_dict = embeddings
        else:
            raise TypeError("Embeddings type {type(embeddings)} not recognized. Expected types are dict or gensim.models.keyedvectors.KeyedVectors")

        return embeddings_dict

    
    def load_data(self):
        raise NotImplementedError()

        
    def evaluate(self, show = False):
        raise NotImplementedError()






        




class EvaluationMethod(nn.Module):

    def __init__(self, embeddings, embeddings_relation = None, device = "cpu"):
        super().__init__()
        num_classes = len(embeddings)        
        embedding_size = len(embeddings[0])

        self.embeddings = nn.Embedding(num_classes, embedding_size)
        self.embeddings.weight = nn.parameter.Parameter(embeddings)
        if not embeddings_relation is None:
            num_rels = len(embeddings_relation)
            self.embeddings_relation = nn.Embedding(num_rels, embedding_size)
            self.embeddings_relation.weight = nn.parameter.Parameter(embeddings_relation)

    def forward(self):
        raise NotImplementedError()

class CosineSimilarity(EvaluationMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        s, d = x[:,0], x[:,2]
        srcs = self.embeddings(s)
        dsts = self.embeddings(d)

        x = th.sum(srcs*dsts, dim=1)
        return 1-th.sigmoid(x)
        
def compute_rank_roc(ranks, n_entities):                                                                               
    auc_x = list(ranks.keys())                                                                                      
    auc_x.sort()                                                                                                    
    auc_y = []                                                                                                      
    tpr = 0                                                                                                         
    sum_rank = sum(ranks.values())
    
    for x in auc_x:                                                                                                 
        tpr += ranks[x]                                                                                             
        auc_y.append(tpr / sum_rank)
        
    auc_x.append(n_entities)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_entities
    return auc
                    
