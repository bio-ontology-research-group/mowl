import numpy as np
import torch as th
from scipy.stats import rankdata
import torch.nn as nn
import click as ck
from mowl.graph.edge import Edge
from mowl.graph.util import prettyFormat
from mowl.datasets.build_ontology import PREFIXES
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

class Evaluator():
    """
    Abstract class for evaluation of models.
    """

    def __init__(self,
                 embeddings,
                 training_set: list,
                 testing_set: list,
                 mode = "cosine_similarity",
                 device = "cpu"
                 ):

        self.training_set = training_set
        self.testing_set = testing_set
        self.mode = mode

        self.head_entities = None
        self.head_entity_names = None
        self.head_entity_name_index = None
        self.tail_entities = None
        self.tail_entity_names = None
        self.tail_entity_name_index = None
        self.trlabels = None
        
        self.embeddings = {}
        self.device = device
        if isinstance(embeddings, KeyedVectors):
            for idx, word in enumerate(embeddings.index_to_key):
                self.embeddings[word] = embeddings[word]
        elif isinstance(embeddings, dict):
            self.embeddings = embeddings
        else:
            raise TypeError("Embeddings type {type(embeddings)} not recognized. Expected types are dict or gensim.models.keyedvectors.KeyedVectors")


    def load_data(self):
        raise NotImplementedError()

        
    def evaluate(self, show = False):
            
        self.load_data()
        if self.mode == "cosine_similarity":
            model = CosineSimilarity(list(self.head_entities.values()), list(self.tail_entities.values())).to(self.device)
        else:
            raise ValueError("Model not defined")
        
        top1 = 0
        top10 = 0
        top100 = 0
        mean_rank = 0
        ftop1 = 0
        ftop10 = 0
        ftop100 = 0
        fmean_rank = 0
        ranks = {}
        franks = {}
        eval_data = self.testing_set

        cs = {c for c,r,d in eval_data} 

        n = len(eval_data)
        num_head_entities = len(self.head_entity_names)
        num_tail_entities = len(self.tail_entity_names)
                
        labels = np.zeros((num_head_entities, num_tail_entities), dtype=np.int32)
        preds = np.zeros((num_head_entities, num_tail_entities), dtype=np.float32)

        with ck.progressbar(eval_data) as prog_data: 
            for c, _, d in prog_data: 
                c_name = c 
                d_name = d
                
                c, d = self.head_entity_name_index[c], self.tail_entity_name_index[d]
                
                labels[c, d] = 1 

                data = th.tensor([[c, x] for x in self.tail_entity_name_index.values()]).to(self.device)
                res = model(data).cpu().detach().numpy()                                                                                                                   
                preds[c, :] = res                                                                                
                index = rankdata(res, method='average')

                rank = index[d]

                if rank == 1:
                    top1 += 1
                if rank <= 10:
                    top10 += 1
                if rank <= 100:
                    top100 += 1

                mean_rank += rank
                if rank not in ranks:
                    ranks[rank] = 0
                ranks[rank] += 1

                # Filtered rank
               
                index = rankdata((res * self.trlabels[c, :]), method='average')

                rank = index[d]

                if rank == 1:
                    ftop1 += 1
                if rank <= 10:
                    ftop10 += 1
                if rank <= 100:
                    ftop100 += 1
                fmean_rank += rank

                if rank not in franks:
                    franks[rank] = 0
                franks[rank] += 1

            top1 /= n
            top10 /= n
            top100 /= n
            mean_rank /= n
            
            ftop1 /= n
            ftop10 /= n
            ftop100 /= n
            fmean_rank /= n

        rank_auc = compute_rank_roc(ranks, num_tail_entities)
        frank_auc = compute_rank_roc(franks, num_tail_entities)

        if show:
            print(f'Hits@1:   {top1:.2f} Filtered:   {ftop1:.2f}')
            print(f'Hits@10:  {top10:.2f} Filtered:   {ftop10:.2f}')
            print(f'Hits@100: {top100:.2f} Filtered:   {ftop100:.2f}')
            print(f'MR:       {mean_rank:.2f} Filtered: {fmean_rank:.2f}')
            print(f'AUC:      {rank_auc:.2f} Filtered:   {frank_auc:.2f}')

            

        self.metrics = {
            "hits_1": top1,
            "hits_10": top10,
            "hits_100": top100,
            "mean_rank": mean_rank,
            "rank_auc": rank_auc,
            "fhits_1": ftop1,
            "fhits_10": ftop10,
            "fhits_100": ftop100,
            "fmean_rank": fmean_rank,
            "frank_auc": frank_auc
        }

        print('Evaluation finished. Access the results using the "metrics" attribute.')
                                                                                         


class PPIEvaluator(Evaluator):

    """
    Evaluation model for protein-protein interactions
  
    """
    def __init__(
            self,
            embeddings,
            training_set: list,
            testing_set: list,
            mode = "cosine_similarity",
            device = "cpu",
    ):
        super().__init__(embeddings, training_set, testing_set, mode, device)
       
        _, self.rels = Edge.getEntitiesAndRelations(training_set)
        self.rels_dict = {v:k for k,v in enumerate(self.rels)}

        self.training_set = [x.astuple() for x in training_set]
        self.testing_set = [x.astuple() for x in testing_set]
       
        self._data_loaded = False
        self.mode = mode
        self.metrics = {}
        self.device = "cpu" #"cuda" if th.cuda.is_available else "cpu"

    def load_data(self):
        if self._data_loaded:
            return
        self.head_entities = dict() #name -> embedding
        self.tail_entities = dict()
        for k, v in self.embeddings.items():
            if not "4932" in k:
                continue
            k = prettyFormat(k)
            if not k.startswith('<http://purl.obolibrary.org/obo/GO_') and not k.startswith("GO"):
                self.head_entities[k] = v
                self.tail_entities[k] = v
                
        self.head_entity_names = list(self.head_entities.keys())
        self.head_entity_name_index = {v:k for k,v in enumerate(self.head_entity_names)} # name -> index
        self.tail_entity_names = list(self.tail_entities.keys())
        self.tail_entity_name_index = {v:k for k,v in enumerate(self.tail_entity_names)} # name -> index


        print(f"Entities dictionary created. Number of proteins: {len(self.head_entity_names)}.")
        self.trlabels = np.ones((len(self.head_entity_names), len(self.tail_entity_names)), dtype=np.int32)

        for c,r,d in self.training_set:
            if c not in self.head_entity_names or d not in self.tail_entity_names: 
                continue                                                                                            

            c, d =  self.head_entity_name_index[c], self.tail_entity_name_index[d] 
            
            self.trlabels[c, d] = 10000
        print("Training labels created")




class GDAEvaluator(Evaluator):

    """
    Evaluation model for protein-protein interactions
  
    """
    def __init__(
            self,
            embeddings,
            training_set: list,
            testing_set: list,
            mode = "cosine_similarity",
            device = "cpu",
    ):
        super().__init__(embeddings, training_set, testing_set, mode, device)
       
        _, self.rels = Edge.getEntitiesAndRelations(training_set)
        self.rels_dict = {v:k for k,v in enumerate(self.rels)}

        self.training_set = [x.astuple() for x in training_set]
        self.testing_set = [x.astuple() for x in testing_set]
       
        self._data_loaded = False
        self.mode = mode
        self.metrics = {}
        self.device = "cpu" #"cuda" if th.cuda.is_available else "cpu"

    def load_data(self):
        if self._data_loaded:
            return
        self.head_entities = dict() #name -> embedding
        self.tail_entities = dict()
        
        for k, v in self.embeddings.items():
            k = prettyFormat(k)
            if k.isnumeric():
                self.head_entities[k] = v
            if k.startswith('OMIM:'):
                self.tail_entities[k] = v

        self.head_entity_names = list(self.head_entities.keys())
        self.head_entity_name_index = {v:k for k,v in enumerate(self.head_entity_names)} # name -> index
        self.tail_entity_names = list(self.tail_entities.keys())
        self.tail_entity_name_index = {v:k for k,v in enumerate(self.tail_entity_names)} # name -> index


        print(f"Entities dictionary created. Number of genes: {len(self.head_entity_names)}. Number of diseases: {len(self.head_entity_names)}")
        self.trlabels = np.ones((len(self.head_entity_names), len(self.tail_entity_names)), dtype=np.int32)

        for c,r,d in self.training_set:
            if c not in self.head_entity_names or d not in self.tail_entity_names: 
                continue                                                                                            

            c, d =  self.head_entity_name_index[c], self.tail_entity_name_index[d] 
            
            self.trlabels[c, d] = 10000
        print("Training labels created")
                                   


        

        

class CosineSimilarity(nn.Module):

    def __init__(self, embeddings_head, embeddings_tail):
        super().__init__()
        num_classes_head = len(embeddings_head)
        num_classes_tail = len(embeddings_tail)
        embedding_size = len(embeddings_head[0])

        self.embeddings_head = nn.Embedding(num_classes_head, embedding_size)
        self.embeddings_head.weight = nn.parameter.Parameter(th.tensor(np.array(embeddings_head)))
        self.embeddings_tail = nn.Embedding(num_classes_tail, embedding_size)
        self.embeddings_tail.weight = nn.parameter.Parameter(th.tensor(np.array(embeddings_tail)))

    def forward(self, x):
        s, d = x[:,0], x[:,1]
        srcs = self.embeddings_head(s)
        dsts = self.embeddings_tail(d)

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
                    
