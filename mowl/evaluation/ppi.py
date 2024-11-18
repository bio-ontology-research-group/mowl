from mowl.evaluation import Evaluator, RankingEvaluator
from mowl.projection import TaxonomyWithRelationsProjector, Edge

import torch as th

class PPIEvaluatorOld(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tuples(self, ontology):
        projector = TaxonomyWithRelationsProjector(relations=["http://interacts_with"])
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()
        relation_str2owl = self.dataset.object_properties.to_dict()
        relation_owl2idx = self.dataset.object_properties.to_index_dict()
        
        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            relation = relation_owl2idx[relation_str2owl[e.rel]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, relation, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)

    def get_logits(self, model, batch):
        heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]
        num_heads, num_tails = len(heads), len(tails)

        heads = heads.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        rels = rels.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        eval_tails = th.arange(len(self.evaluation_tails), device=heads.device).repeat(num_heads).unsqueeze(1)
        logits_heads = model(th.cat([heads, rels, eval_tails], dim=-1), "gci2")
        logits_heads = logits_heads.view(-1, len(self.evaluation_tails))
        
        tails = tails.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        eval_heads = th.arange(len(self.evaluation_heads), device=tails.device).repeat(num_tails).unsqueeze(1)
        logits_tails = model(th.cat([eval_heads, rels, tails], dim=-1), "gci2")
        logits_tails = logits_tails.view(-1, len(self.evaluation_heads))
        # print(logits_heads, logits_tails)
        return logits_heads, logits_tails

    
    def get_filtering_labels(self, num_heads, num_tails, class_id_to_head_id, class_id_to_tail_id, **kwargs):

        filtering_tuples = th.cat([self.train_tuples, self.valid_tuples], dim=0)
        filtering_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, rel, tail in filtering_tuples:
            head = class_id_to_head_id[head.item()]
            tail = class_id_to_tail_id[tail.item()]
            filtering_labels[head, tail] = 10000
        
        return filtering_labels
    


class PPIEvaluator(RankingEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tuples(self, ontology):
        projector = TaxonomyWithRelationsProjector(relations=["http://interacts_with"])
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()
        relation_str2owl = self.dataset.object_properties.to_dict()
        relation_owl2idx = self.dataset.object_properties.to_index_dict()
        
        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            relation = relation_owl2idx[relation_str2owl[e.rel]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, relation, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)

    def get_scores(self, model, batch):
        scores = model(batch, "gci2")
        return scores

    

    

