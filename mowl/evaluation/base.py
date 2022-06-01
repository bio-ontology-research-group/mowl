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



class AxiomsRankBasedEvaluator():

    """
    :param axioms: A set of axioms over which perform the evaluation. The axioms can be given as an .owl file or as a list of OWLAxioms.
    :param embeddinds_data: A list of dictionaries, each of which has as keys the names of the entities and as values the embedding vectors
    :param eval_method: The evaluation method for the axioms. 
    :param axioms_to_filter: Axioms to be put at the bottom of the rankings. If the axioms are empty, filtered metrics will not be computed.
    """
    
    def __init__(
            self,
            axioms,
            eval_method,
            axioms_to_filter = None,
            device = "cpu",
            verbose = False
    ):

        self.axioms = self._init_axioms(axioms)
        self.eval_method = eval_method
        self.device = device
        self.verbose = verbose
        if axioms_to_filter is None:
            self._compute_filtered_metrics = False
        else:
            self._compute_filtered_metrics = True
            
        self.axioms_to_filter = self._init_axioms(axioms_to_filter)
        
        return

    def _init_axioms(self, axioms):
        """This method must transform the axioms into the appropriate data structure to be used by the ``eval_method``. This method accesses the ``axioms`` variable, which can be an OWL file or a list of OWLAxioms.
        """
        raise NotImplementedError()

    def compute_axiom_rank(self, axiom):
        raise NotImplementedError()
    
    def __call__(self):

        tops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        ftops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        mean_rank = 0
        fmean_rank = 0
        ranks = {}
        franks = {}

        n = 0
        for axiom in tqdm(self.axioms):
            rank, frank, worst_rank = self.compute_axiom_rank(axiom)

            if rank is None:
                continue

            n = n+1
            for top in tops:
                if rank <= top:
                    tops[top] += 1

            mean_rank += rank

            if rank not in ranks:
                ranks[rank] = 0
            ranks[rank] += 1

            # Filtered rank
            if self._compute_filtered_metrics:
                for ftop in ftops:
                    if frank <= ftop:
                        ftops[ftop] += 1

                if rank not in franks:
                    franks[rank] = 0
                franks[rank] += 1

                fmean_rank += frank

        tops = {k: v/n for k, v in tops.items()}
        ftops = {k: v/n for k, v in ftops.items()}

        mean_rank, fmean_rank = mean_rank/n, fmean_rank/n

        rank_auc = compute_rank_roc(ranks, worst_rank)
        frank_auc = compute_rank_roc(franks, worst_rank)

        self._metrics = {f"hits@{k}": tops[k] for k in tops}
        self._metrics["mean_rank"] = mean_rank
        self._metrics["rank_auc"] = rank_auc
        self._fmetrics = {f"hits@{k}": ftops[k] for k in ftops}
        self._fmetrics["mean_rank"] = fmean_rank
        self._fmetrics["rank_auc"] = frank_auc

        return


    def print_metrics(self):

        to_print = "Normal:\t"
        for name, value in self._metrics.items():
            to_print += f"{name}: {value:.2f}\t"

        to_print += "\nFiltered:\t"
        for name, value in self._fmetrics.items():
            to_print += f"{name}: {value:.2f}\t"


        print(to_print)


    
class CosineSimilarity(EvaluationMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        s, d = x[:,0], x[:,2]
        srcs = self.embeddings(s)
        dsts = self.embeddings(d)

        x = th.sum(srcs*dsts, dim=1)
        return 1-th.sigmoid(x)
        
def compute_rank_roc(ranks, worst_rank): 

    auc_x = list(ranks.keys())                                                                                      
    auc_x.sort()                                                                                                    
    auc_y = []                                                                                                      
    tpr = 0                                                                                                         
    sum_rank = sum(ranks.values()) #number of evaluation points
    
    for x in auc_x:                                                                                                 
        tpr += ranks[x]                                                                                             
        auc_y.append(tpr / sum_rank)
        
    auc_x.append(worst_rank)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / worst_rank
    return auc
                    
