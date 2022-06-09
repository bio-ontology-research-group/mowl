from mowl.evaluation.base import AxiomsRankBasedEvaluator
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import logging
import numpy as np
from scipy.stats import rankdata
import torch as th
import random
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from mowl.evaluation.base import compute_rank_roc
class CatEmbeddingsPPIEvaluator(AxiomsRankBasedEvaluator):

    def __init__(
            self,
            eval_method,
            axioms_to_filter,
            class_name_indexemb,
            rel_name_indexemb,
            proteins,
            device = "cpu",
            verbose = False
    ):

        super().__init__(eval_method, axioms_to_filter, device, verbose)

        self.class_name_indexemb = class_name_indexemb
        self.relation_name_indexemb = rel_name_indexemb

        self.head_entities_raw = list(set(proteins))
        self.tail_entities_raw = self.head_entities_raw[:]

        self._loaded_filtered_scores = False
        self._loaded_eval_data = False
        self._loaded_ht_data = False
        self._n_prots = len(proteins)

        self._load_head_tail_entities()
        
    def _load_head_tail_entities(self):
        if self._loaded_ht_data:
            return

        self.head_entities = set()
        for e in self.head_entities_raw:
            if e in self.class_name_indexemb:
                self.head_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)       

        self.tail_entities = set()
        for e in self.tail_entities_raw:
            if e in self.class_name_indexemb:
                self.tail_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)

        self.head_name_indexemb = {k: self.class_name_indexemb[k] for k in self.head_entities}
        self.tail_name_indexemb = {k: self.class_name_indexemb[k] for k in self.tail_entities}

        self.head_indexemb_indexsc = {v: k for k, v in enumerate(self.head_name_indexemb.values())}
        self.tail_indexemb_indexsc = {v: k for k, v in enumerate(self.tail_name_indexemb.values())}
                
        self._loaded_ht_data = True

        

    def _load_filtered_scores(self):
        if self._loaded_filtered_scores:
            return self.filtered_scores
        
        filtered_scores = np.ones((len(self.head_entities), len(self.tail_entities)), dtype=np.int32)

        if self._compute_filtered_metrics:
        # careful here: c must be in head entities and d must be in tail entities
            for axiom in self.axioms_to_filter:
                c, _, d = axiom.astuple()
                if (not c in self.head_entities) or not (d in self.tail_entities):
                    continue
            
                c, d = self.head_name_indexemb[c], self.tail_name_indexemb[d]
                c, d = self.head_indexemb_indexsc[c], self.tail_indexemb_indexsc[d]
            
                filtered_scores[c, d] = 10000

            logging.info("Training scores created")

        self._loaded_filtered_scores = True
        return filtered_scores
        
    
    def _init_axioms(self, axioms):

        if axioms is None:
            return None
        
        projector = projector_factory("taxonomy_rels", relations = ["http://interacts"])

        edges = projector.project(axioms)

        #edges = [(self.head_name_indexemb[e.src()], self.relation_name_indexemb[e.rel()], self.tail_name_indexemb[e.dst()]) for e in edges]
        edges = random.sample(edges, 1000)
        return edges # List of Edges

    def _init_axioms_to_filter(self, axioms):
        if axioms is None:
            return None
        
        projector = projector_factory("taxonomy_rels", relations = ["http://interacts"])

        edges = projector.project(axioms)

        #edges = [(self.head_name_indexemb[e.src()], self.relation_name_indexemb[e.rel()], self.tail_name_indexemb[e.dst()]) for e in edges]
        
        return edges # List of Edges



    def get_predictions(self, samples=None, save = True):
        logging.info("Computing prediction on %s", str(self.device))
    
#        if samples is None:
#            logging.info("No data points specified. Proceeding to compute predictions on test set")
#            model.load_state_dict(th.load( self.model_filepath))
#            model = model.to(self.device)
#            _, _, test_nf3, _ = self.test_nfs

#            eval_data = test_nf3

 #       else:
 #           eval_data = samples

        test_model = TestModulePPI(self.eval_method)

        preds = np.zeros((len(self.head_entities), len(self.tail_entities)), dtype=np.float32)
            
        test_dataset = TestDatasetPPI(self.axioms, self.class_name_indexemb, self.head_indexemb_indexsc, self.tail_indexemb_indexsc, self.relation_name_indexemb["http://interacts"])
    
        bs = 8
        test_dl = DataLoader(test_dataset, batch_size = bs)

        for idxs, batch in tqdm(test_dl):

            res = test_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            preds[idxs,:] = res

        

#        if save:
#            with open(self.predictions_file, "wb") as f:
#                pkl.dump(preds, f)

        return preds

    
    def compute_axiom_rank(self, axiom, predictions):
        
        c, r, d = axiom.astuple()
        
        if not (c in self.head_entities) or not (d in self.tail_entities):
            return None, None, None

        # Embedding indices
        c_emb_idx, d_emb_idx = self.head_name_indexemb[c], self.tail_name_indexemb[d]

        # Scores matrix labels
        c_sc_idx, d_sc_idx = self.head_indexemb_indexsc[c_emb_idx], self.tail_indexemb_indexsc[d_emb_idx]

        r = self.relation_name_indexemb[r]

#        data = th.tensor([[c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities]).to(self.device)

        res = predictions[c_sc_idx,:]
#        res = self.eval_method(data).squeeze().cpu().detach().numpy()
        
        #self.testing_predictions[c_sc_idx, :] = res                                                                                
        index = rankdata(res, method='average')
        rank = index[d_sc_idx]

        findex = rankdata((res * self.filtered_scores[c_sc_idx, :]), method='average')
        frank = findex[d_sc_idx]

        return rank, frank, len(self.tail_entities)

class CatEmbeddingsIntersectionEvaluator(AxiomsRankBasedEvaluator):

    def __init__(
            self,
            eval_method,
            class_name_indexemb,
            product_generator,
            device = "cpu",
            verbose = False
    ):

        super().__init__(eval_method, None, device, verbose)

        self.class_name_indexemb = class_name_indexemb
        self.product_generator = product_generator
        
                        
        self._loaded_eval_data = False
        self._loaded_ht_data = False        
    
    def _init_axioms(self, axioms):
        return axioms
                                                     

    def _init_axioms_to_filter(self, axioms):
        return axioms



    def get_predictions(self, samples=None, save = True):
        logging.info("Computing prediction on %s", str(self.device))
    
#        if samples is None:
#            logging.info("No data points specified. Proceeding to compute predictions on test set")
#            model.load_state_dict(th.load( self.model_filepath))
#            model = model.to(self.device)
#            _, _, test_nf3, _ = self.test_nfs

#            eval_data = test_nf3

 #       else:
 #           eval_data = samples

        test_model = TestModuleIntersection(self.product_generator, self.eval_method, self.embeddings).to(self.device)

        preds = np.zeros((len(samples), len(self.embeddings)), dtype=np.float32)
            
        test_dataset = TestDatasetIntersection(samples, self.class_name_indexemb)
    
        bs = 4
        test_dl = DataLoader(test_dataset, batch_size = bs)

        
        for idx_l, idx_r, batch in tqdm(test_dl):

            idxs = []
            for l,r in zip(idx_l, idx_r):
                l = l.detach().item()
                r = r.detach().item()
                idxs.append(self.left_side_dict[(l,r)])
                
            res = test_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            preds[idxs,:] = res

        

#        if save:
#            with open(self.predictions_file, "wb") as f:
#                pkl.dump(preds, f)

        return preds

    
    def compute_axiom_rank(self, axiom, predictions):
        
        l, r, d = axiom

        lr = self.left_side_dict[(l,r)]

#        data = th.tensor([[c_emb_idx, r, self.tail_name_indexemb[x]] for x in self.tail_entities]).to(self.device)

        res = predictions[lr,:]
#        res = self.eval_method(data).squeeze().cpu().detach().numpy()
        
        #self.testing_predictions[c_sc_idx, :] = res                                                                                
        index = rankdata(res, method='average')
        rank = index[d]

        findex = rankdata((res), method='average')
        frank = findex[d]

        return rank, frank, len(self.embeddings)


    def __call__(self, axioms, embeddings, init_axioms = False):
        self.embeddings = embeddings
        self.left_side_dict = {v[:-1]:k for k,v in enumerate(axioms)}
        predictions = self.get_predictions(axioms)
        tops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        ftops = {1: 0, 3: 0, 5: 0, 10:0, 100:0, 1000:0}
        mean_rank = 0
        fmean_rank = 0
        ranks = {}
        franks = {}
        
        n = 0
        for axiom in tqdm(axioms):
            rank, frank, worst_rank = self.compute_axiom_rank(axiom, predictions)
            
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




class TestModulePPI(nn.Module):
    def __init__(self, method):
        super().__init__()

        self.method = method        

    def forward(self, x):
        bs, num_prots, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        x = self.method(x)

        x = x.reshape(bs, num_prots)

        return x



class TestDatasetPPI(IterableDataset):
    def __init__(self, data, class_name_indexemb, head_indexemb_indexsc, tail_indexemb_indexsc, r):
        super().__init__()
        self.data = data
        self.class_name_indexemb = class_name_indexemb
        self.head_indexemb_indexsc = head_indexemb_indexsc
        self.len_data = len(data)

        self.predata = np.array([[0, r, x] for x in tail_indexemb_indexsc])
        
    def get_data(self):
        for edge in self.data:
            c, r, d = edge.astuple()
            c, d = self.class_name_indexemb[c], self.class_name_indexemb[d]
            c_emb = c #.detach().item()
            c = self.head_indexemb_indexsc[c]
            new_array = np.array(self.predata, copy = True)
            new_array[:,0] = c_emb
            
            tensor = new_array
            yield c, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data


    
class TestModuleIntersection(nn.Module):
    def __init__(self, prod_generator,  method, embeddings):
        super().__init__()

        self.prod_generator = prod_generator
        self.method = method

        embeddings = list(embeddings.values())
        self.embeddings = nn.Embedding(len(embeddings), len(embeddings[0]))

    def forward(self, x):
        bs, num_axioms, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        c_left = x[:,0]
        c_right = x[:,1]
        d = x[:,2]

        c_left = self.embeddings(c_left)
        c_right = self.embeddings(c_right)
        d = self.embeddings(d)
        
        intersection, _ = self.prod_generator(c_left, c_right)
        
        scores = self.method(intersection, d)
        

#        x = self.method(x)

        x = scores.reshape(bs, num_axioms)

        return x



class TestDatasetIntersection(IterableDataset):
    def __init__(self, data, class_name_indexemb):
        super().__init__()
        self.data = data
        self.len_data = len(data)
        self.class_name_indexemb = class_name_indexemb
        self.predata = np.array([[0,0, x] for x in list(class_name_indexemb.values())])
        
    def get_data(self):
        for axiom in self.data:
            c_left, c_right, d = axiom
#            c_left, c_right = self.class_name_indexemb[c_left], self.class_name_indexemb[c_right]
            new_array = np.array(self.predata, copy = True)
            new_array[:,0] = c_left
            new_array[:,1] = c_right
            
            tensor = new_array
            yield c_left, c_right, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data



    
