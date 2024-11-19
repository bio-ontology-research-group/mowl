from mowl.evaluation import Evaluator, RankingEvaluator
from mowl.projection import TaxonomyProjector, Edge
import torch as th


class SubsumptionEvaluatorOld(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tuples(self, ontology):
        projector = TaxonomyProjector()
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()

        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)

    def get_logits(self, model, batch):
        heads, tails = batch[:, 0], batch[:, 1]
        num_heads, num_tails = len(heads), len(tails)

        heads = heads.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        eval_tails = th.arange(len(self.evaluation_tails), device=heads.device).repeat(num_heads).unsqueeze(1)
        logits_heads = model(th.cat([heads, eval_tails], dim=-1), "gci0")
        logits_heads = logits_heads.view(-1, len(self.evaluation_tails))
        
        tails = tails.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        eval_heads = th.arange(len(self.evaluation_heads), device=tails.device).repeat(num_tails).unsqueeze(1)
        logits_tails = model(th.cat([eval_heads, tails], dim=-1), "gci0")
        logits_tails = logits_tails.view(-1, len(self.evaluation_heads))
        # print(logits_heads, logits_tails)
        return logits_heads, logits_tails

    
    def get_filtering_labels(self, num_heads, num_tails, **kwargs):
        filter_deductive_closure = kwargs["filter_deductive_closure"]
        
        if filter_deductive_closure:
            # take deductive closure tuples that are not in the testing tuples

            mask = (self.deductive_closure_tuples.unsqueeze(1) == self.test_tuples).all(dim=-1).any(dim=-1)
            deductive_closure_tuples = self.deductive_closure_tuples[~mask]
            
            
            filtering_tuples = th.cat([self.train_tuples, self.valid_tuples, deductive_closure_tuples], dim=0)
        else:
            filtering_tuples = th.cat([self.train_tuples, self.valid_tuples], dim=0)

        filtering_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, tail in filtering_tuples:
            filtering_labels[head, tail] = 10000
        
        return filtering_labels
    


    def get_deductive_labels(self, num_heads, num_tails, class_id_to_head_id, class_id_to_tail_id):
        deductive_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, tail in self.deductive_closure_tuples:
            head = class_id_to_head_id[head.item()]
            tail = class_id_to_tail_id[tail.item()]
            deductive_labels[head, tail] = 10000
        
        return deductive_labels



class SubsumptionEvaluator(RankingEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tuples(self, ontology):
        projector = TaxonomyProjector()
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()

        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)

    def get_scores(self, model, batch):
        scores = model(batch, "gci0")
        return scores
