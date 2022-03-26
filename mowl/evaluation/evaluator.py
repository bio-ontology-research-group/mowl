import numpy as np
import torch as th
from scipy.stats import rankdata
import torch.nn as nn
import click as ck
from mowl.graph.edge import Edge
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

class Evaluator():
    """
    Abstract class for evaluation of models.
    """
    
    
    def __init__(self, embeddings, device = "cpu"):
        self.embeddings = {}
        self.device = device
        if isinstance(embeddings, KeyedVectors):
            for idx, word in enumerate(embeddings.index_to_key):
                self.embeddings[word] = embeddings[word]
        elif isinstance(embeddings, dict):
            self.embeddings = embeddings
        else:
            raise TypeError("Embeddings type {type(embeddings)} not recognized. Expected types are dict or gensim.models.keyedvectors.KeyedVectors")

    def evaluate():
        raise NotImplementedError()

class PPIEvaluator(Evaluator):

    """
    Evaluation model for protein-protein interactions
   
    """
    def __init__(self, embeddings, training_set: list, testing_set: list, mode = "cosine_similarity", device = "cpu"):
        super().__init__(embeddings, device)
        
        _, self.rels = Edge.getEntitiesAndRelations(training_set)
        self.rels_dict = {v:k for k,v in enumerate(self.rels)}

        self.training_set = [x.astuple() for x in training_set].to(device)
        self.testing_set = [x.astuple() for x in testing_set].to(device)
        
        self._data_loaded = False
        self.mode = mode
        self.metrics = {}
        self.device = "cpu" #"cuda" if th.cuda.is_available else "cpu"

        
        
    def load_data(self):
        if self._data_loaded:
            return
        self.proteins = dict() #name -> embedding

        for k, v in self.embeddings.items():
            if not k.startswith('<http://purl.obolibrary.org/obo/GO_') and not k.startswith("GO"):                   
                self.proteins[k] = v

        self.prot_names = list(self.proteins.keys())
        self.prot_name_index = {v:k for k,v in enumerate(self.prot_names)} # name -> index


        print(f"Proteins dictionary created. Number of proteins: {len(self.prot_names)}")
        self.trlabels = {}

        for c,r,d in self.training_set:
            
            if c not in self.prot_names or d not in self.prot_names: 
                continue                                                                                              

            c, d =  self.prot_name_index[c], self.prot_name_index[d] 
            r = self.rels_dict[r]
            
            if r not in self.trlabels:                                                                                
                self.trlabels[r] = np.ones((len(self.prot_names), len(self.prot_names)), dtype=np.int32)              
            self.trlabels[r][c, d] = 10000                                                            
        print("Training labels created")                                                                                     
                                   

    def evaluate(self, show = False):
        self.load_data()
        if self.mode == "cosine_similarity":
            model = CosineSimilarity(list(self.proteins.values())).to(self.device)
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
        labels = {} 
        preds = {} 
        ranks = {} 
        franks = {}
        eval_data = self.testing_set

        cs = {c for c,r,d in eval_data} 

        n = len(eval_data)
        num_prots = len(self.prot_names)

        with ck.progressbar(eval_data) as prog_data: 
            for c, r, d in prog_data: 
                c_name = c 
                d_name = d
                c, d = self.prot_name_index[c], self.prot_name_index[d]
                r = self.rels_dict[r]

                if r not in labels: 
                    labels[r] = np.zeros((num_prots, num_prots), dtype=np.int32) 
                if r not in preds: 
                    preds[r] = np.zeros((num_prots, num_prots), dtype=np.float32) 

                labels[r][c, d] = 1 

                data = th.tensor([[c, r, x] for x in self.prot_name_index.values()]).to(self.device)

                res = model(data).cpu().detach().numpy()                                                                                                                    


                preds[r][c, :] = res                                                                                                                                                 
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
               
                index = rankdata((res * self.trlabels[r][c, :]), method='average')                                                                                                        

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

        rank_auc = compute_rank_roc(ranks, num_prots)                                                                                                                                
        frank_auc = compute_rank_roc(franks, num_prots)                                                                                                                              

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
                                                                                         

        

        

class CosineSimilarity(nn.Module):

    def __init__(self, embeddings):
        super().__init__()
        num_classes = len(embeddings)
        embedding_size = len(embeddings[0])
        
        self.embeddings = nn.Embedding(num_classes, embedding_size)
        self.embeddings.weight = nn.parameter.Parameter(th.tensor(np.array(embeddings)))
        
    def forward(self, x):
        s, d = x[:,0], x[:,2]
        srcs = self.embeddings(s)
        dsts = self.embeddings(d)

        x = th.sum(srcs*dsts, dim=1)
        return 1-th.sigmoid(x)
        
def compute_rank_roc(ranks, n_prots):                                                                                 
    auc_x = list(ranks.keys())                                                                                        
    auc_x.sort()                                                                                                      
    auc_y = []                                                                                                        
    tpr = 0                                                                                                           
    sum_rank = sum(ranks.values())                                                                                    
    for x in auc_x:                                                                                                   
        tpr += ranks[x]                                                                                               
        auc_y.append(tpr / sum_rank)                                                                                  
    auc_x.append(n_prots)                                                                                             
    auc_y.append(1)                                                                                                   
    auc = np.trapz(auc_y, auc_x) / n_prots                                                                            
    return auc                                                                                                        
                    
