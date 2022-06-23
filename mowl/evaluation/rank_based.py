from mowl.evaluation.base import Evaluator, compute_rank_roc
import logging
import numpy as np
from tqdm import tqdm
from mowl.projection.factory import projector_factory
from scipy.stats import rankdata
import torch as th

class RankBasedEvaluator(Evaluator):

    """
    This class corresponds to evaluation based on ranking. That is, for each testing triple :math:`(h,r,t)`, scores are computed for triples :math:`(h,r,t')` for all possible :math:`t'`. After that, the ranking of the testing triple :math:`(h,r,t)` score is obtained.

    :param device: Use `cpu` or `cuda`
    :type device: str
    :param homogeneous: This parameter indicates whether or not the head entities are of the same type of the tail entities.
 
    """
    
    def __init__(self,
                 class_index_emb,
                 relation_index_emb,
                 testing_set,
                 eval_method,
                 training_set,
                 head_entities,
                 tail_entities,
                 device):
        super().__init__(device)


        self.eval_method = eval_method
        self.compute_filtered_metrics = True
        if training_set is None:
            self.compute_filtered_metrics = False
            logging.info("Training set was not input. Filtered metrics will not be available.")

        self.device = device
        self._data_loaded : bool

        self.relation_index_emb = relation_index_emb
        
        self.head_entities = head_entities
        self.head_name_indexemb : dict
        self.head_indexemb_indexsc : dict
        
        self.tail_entities = tail_entities
        self.tail_name_indexemb : dict
        self.tail_indexemb_indexsc : dict
        
        self.class_index_emb = class_index_emb
        self.training_set = [x.astuple() for x in training_set]
        self.testing_set = [x.astuple() for x in testing_set]

        self._loaded_ht_data = False
        self._loaded_tr_scores = False

        self.filter_head_tail_data()
        
        self.training_scores = np.ones((len(self.head_entities), len(self.tail_entities)), dtype=np.int32)
        self.testing_scores = np.ones((len(self.head_entities), len(self.tail_entities)), dtype=np.int32)
        self.testing_predictions = np.zeros((len(self.head_entities), len(self.tail_entities)), dtype=np.int32)
        
        self.load_training_scores()
        
        
    def filter_head_tail_data(self):

        if self._loaded_ht_data:
            return

        new_head_entities = set()
        new_tail_entities = set()
                     
        for e in self.head_entities:
            if e in self.class_index_emb:
                new_head_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)

        for e in self.tail_entities:
            if e in self.class_index_emb:
                new_tail_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)

        self.head_entities = new_head_entities
        self.tail_entities = new_tail_entities
        
        self.head_name_indexemb = {k: self.class_index_emb[k] for k in self.head_entities}
        self.tail_name_indexemb = {k: self.class_index_emb[k] for k in self.tail_entities}

        self.head_indexemb_indexsc = {v: k for k, v in enumerate(self.head_name_indexemb.values())}
        self.tail_indexemb_indexsc = {v: k for k, v in enumerate(self.tail_name_indexemb.values())}
                
        self._loaded_ht_data = True
        
    def load_training_scores(self):

        if self._loaded_tr_scores or not self.compute_filtered_metrics:
            return

        # careful here: c must be in head entities and d must be in tail entities
        for c, _, d in self.training_set:
            if (not c in self.head_entities) or not (d in self.tail_entities):
                continue
            
            c, d = self.head_name_indexemb[c], self.tail_name_indexemb[d]
            c, d = self.head_indexemb_indexsc[c], self.tail_indexemb_indexsc[d]
            
            self.training_scores[c, d] = 10000

        logging.info("Training scores created")
        self._loaded_tr_scores = True
        
    def evaluate(self, show = False):

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
        
        num_head_entities = len(self.head_entities)
        num_tail_entities = len(self.tail_entities)

        worst_rank = num_tail_entities
        
        n = len(self.testing_set)

        for c, r, d in tqdm(self.testing_set):

            if not (c in self.head_entities) or not (d in self.tail_entities):
                
                n-=1
                if not d in self.tail_entities:
                    worst_rank -= 1
                continue

            # Embedding indices
            c_emb_idx, d_emb_idx = self.head_name_indexemb[c], self.tail_name_indexemb[d]

            # Scores matrix labels
            c_sc_idx, d_sc_idx = self.head_indexemb_indexsc[c_emb_idx], self.tail_indexemb_indexsc[d_emb_idx]


            r = self.relation_index_emb[r]

            data = th.tensor([[c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities]).to(self.device)
            res = self.eval_method(data).squeeze().cpu().detach().numpy()                                                                                                                   
            self.testing_predictions[c_sc_idx, :] = res                                                                                
            index = rankdata(res, method='average')
            rank = index[d_sc_idx]

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

            if self.compute_filtered_metrics:
                index = rankdata((res * self.training_scores[c_sc_idx, :]), method='average')
                rank = index[d_sc_idx]

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

        rank_auc = compute_rank_roc(ranks, worst_rank)
        frank_auc = compute_rank_roc(franks, worst_rank)

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
                                                                                         

class ModelRankBasedEvaluator(RankBasedEvaluator):

    def __init__(self,
                 model,
                 device = "cpu",
                 ):

        self.model = model
        class_embeddings, relation_embeddings = self.model.get_embeddings()
        
        self.class_embeddings = self.embeddings_to_dict(class_embeddings)
        class_index_emb = {v: k for k, v in enumerate(self.class_embeddings.keys())}


        testing_set = model.testing_set
        training_set = model.training_set
        head_entities = model.head_entities
        tail_entities = model.tail_entities
        eval_method = model.eval_method


        relation = testing_set[0].rel()
        if relation_embeddings is None:
            relation_index_emb = {relation: -1}
        else:
            relation_index_emb = {v: k for k, v in enumerate(relation_embeddings.keys())}
            



        super().__init__(
            class_index_emb,
            relation_index_emb,
            testing_set,
            eval_method,
            training_set,
            head_entities,
            tail_entities,
            device
        )
        
        
        


class EmbeddingsRankBasedEvaluator(RankBasedEvaluator):

    def __init__(self,
                 class_embeddings,
                 testing_set,
                 eval_method,
                 training_set = None,
                 relation_embeddings = None,
                 head_entities = None,
                 tail_entities = None,
                 device = "cpu"):

        self.class_embeddings = self.embeddings_to_dict(class_embeddings)
        class_embeddings_values = th.tensor(list(self.class_embeddings.values())).to(device)
        class_index_emb = {v: k for k, v in enumerate(self.class_embeddings.keys())}

        relation = testing_set[0].rel()
        
        if relation_embeddings is None:
            rel_embeds_values = None
            relation_index_emb = {relation: -1}
        else:
            self.relation_embeddings = self.embeddings_to_dict(relation_embeddings)
            rel_embeds_values = th.tensor(list(self.relation_embeddings.values())).to(device)
            relation_index_emb = {v: k for k, v in enumerate(self.relation_embeddings.keys())}

        self.eval_method = eval_method(class_embeddings_values, rel_embeds_values, device = device).to(device)
     

        if tail_entities is None:
            if head_entities is None:
                logging.info("Neither head nor tail entites input. Head and tail entities will be extracted from testing and training data.")

                head_test_entities, _, tail_test_entities = zip(*[x.astuple() for x in testing_set])

                if not training_set is None:
                    head_train_entities, _, tail_train_entities = zip(*[x.astuple() for x in training_set])
                    
                head_entities = set(head_test_entities) | set(head_train_entities)
                tail_entities = set(tail_test_entities) | set(tail_train_entities)

                
            else:
                logging.info("Tail entities not input. It will be assumed that tail entities are the same as head entities.")
                tail_entities = head_entities

        super().__init__(
            class_index_emb,
            relation_index_emb,
            testing_set,
            self.eval_method,
            training_set,
            head_entities,
            tail_entities,
            device
        )

        
   
