from mowl.evaluation.base import AxiomsRankBasedEvaluator
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import logging
import numpy as np
from scipy.stats import rankdata
import torch as th


class BoxSquaredELPPIEvaluator(AxiomsRankBasedEvaluator):

    def __init__(
            self,
            axioms,
            eval_method,
            axioms_to_filter,
            class_name_indexemb,
            rel_name_indexemb,
            device="cpu",
            verbose=False
    ):

        super().__init__(axioms, eval_method, axioms_to_filter, device, verbose)

        self.class_name_indexemb = class_name_indexemb
        self.relation_name_indexemb = rel_name_indexemb

        self._loaded_training_scores = False
        self._loaded_eval_data = False
        self._loaded_ht_data = False

    def _load_head_tail_entities(self):
        if self._loaded_ht_data:
            return

        ents, _ = Edge.getEntitiesAndRelations(self.axioms)
        ents_filter, _ = Edge.getEntitiesAndRelations(self.axioms_to_filter)

        entities = list(set(ents) | set(ents_filter))

        self.head_entities = set()
        for e in entities:
            if e in self.class_name_indexemb:
                self.head_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)

        self.tail_entities = set()
        for e in entities:
            if e in self.class_name_indexemb:
                self.tail_entities.add(e)
            else:
                logging.info("Entity %s not present in the embeddings dictionary. Ignoring it.", e)

        self.head_name_indexemb = {k: self.class_name_indexemb[k] for k in self.head_entities}
        self.tail_name_indexemb = {k: self.class_name_indexemb[k] for k in self.tail_entities}

        self.head_indexemb_indexsc = {v: k for k, v in enumerate(self.head_name_indexemb.values())}
        self.tail_indexemb_indexsc = {v: k for k, v in enumerate(self.tail_name_indexemb.values())}

        self._loaded_ht_data = True

    def _load_training_scores(self):
        if self._loaded_training_scores:
            return self.training_scores

        self._load_head_tail_entities()

        training_scores = np.ones((len(self.head_entities), len(self.tail_entities)),
                                  dtype=np.int32)

        if self._compute_filtered_metrics:
            # careful here: c must be in head entities and d must be in tail entities
            for axiom in self.axioms_to_filter:
                c, _, d = axiom.astuple()
                if (c not in self.head_entities) or not (d in self.tail_entities):
                    continue

                c, d = self.head_name_indexemb[c], self.tail_name_indexemb[d]
                c, d = self.head_indexemb_indexsc[c], self.tail_indexemb_indexsc[d]

                training_scores[c, d] = 10000

            logging.info("Training scores created")

        self._loaded_training_scores = True
        return training_scores

    def _init_axioms(self, axioms):

        if axioms is None:
            return None

        projector = projector_factory("taxonomy_rels", relations=["http://interacts_with"])

        edges = projector.project(axioms)
        return edges  # List of Edges

    def compute_axiom_rank(self, axiom):

        self.training_scores = self._load_training_scores()

        c, r, d = axiom.astuple()

        if not (c in self.head_entities) or not (d in self.tail_entities):
            return None, None, None

        # Embedding indices
        c_emb_idx, d_emb_idx = self.head_name_indexemb[c], self.tail_name_indexemb[d]

        # Scores matrix labels
        c_sc_idx, d_sc_idx = self.head_indexemb_indexsc[c_emb_idx],
        self.tail_indexemb_indexsc[d_emb_idx]

        r = self.relation_name_indexemb[r]

        data = th.tensor([
            [c_emb_idx, r, self.tail_name_indexemb[x]] for x in
            self.tail_entities]).to(self.device)

        res = self.eval_method(data).squeeze().cpu().detach().numpy()

        # self.testing_predictions[c_sc_idx, :] = res
        index = rankdata(res, method='average')
        rank = index[d_sc_idx]

        findex = rankdata((res * self.training_scores[c_sc_idx, :]), method='average')
        frank = findex[d_sc_idx]

        return rank, frank, len(self.tail_entities)
