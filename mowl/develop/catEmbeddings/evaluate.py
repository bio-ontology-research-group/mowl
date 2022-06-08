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

        test_model = TestModule(self.eval_method)

        preds = np.zeros((len(self.head_entities), len(self.tail_entities)), dtype=np.float32)
            
        test_dataset = TestDataset(self.axioms, self.class_name_indexemb, self.head_indexemb_indexsc, self.tail_indexemb_indexsc, self.relation_name_indexemb["http://interacts"])
    
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



class TestModule(nn.Module):
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



class TestDataset(IterableDataset):
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
