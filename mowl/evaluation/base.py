import numpy as np
import torch as th

from mowl.utils.data import FastTensorDataLoader
from mowl.error import messages as msg

import logging
from deprecated.sphinx import versionchanged
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

@versionchanged(version="1.0.0", reason="Updated Evaluator with a new API.")
class Evaluator:
    """
    Base evaluation class for ontology embedding methods.

    :param dataset: mOWL dataset object. Required to obtain the ontology entities (classes, individuals, object properties, etc.).
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param device: Device to use for the evaluation. Defaults to 'cpu'.
    :type device: str, optional
    :param batch_size: Batch size for evaluation. Defaults to 16.
    :type batch_size: int, optional
    """
    
    def __init__(self, dataset, device="cpu", batch_size=16):


        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.train_tuples = self.create_tuples(dataset.ontology)
        self.valid_tuples = self.create_tuples(dataset.validation)
        self.test_tuples = self.create_tuples(dataset.testing)
        self._deductive_closure_tuples = None

        self.class_to_id = {c: i for i, c in enumerate(self.dataset.classes.as_str)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}

        self.relation_to_id = {r: i for i, r in enumerate(self.dataset.object_properties.as_str)}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}

        eval_heads, eval_tails = self.dataset.evaluation_classes
        self.class_id_to_head_id = {self.class_to_id[c]: i for i, c in enumerate(eval_heads.as_str)}
        self.class_id_to_tail_id = {self.class_to_id[c]: i for i, c in enumerate(eval_tails.as_str)}
         
        eval_heads, eval_tails = self.dataset.evaluation_classes
        
        print(f"Number of evaluation classes: {len(eval_heads)}")
        self.evaluation_heads = th.tensor([self.class_to_id[c] for c in eval_heads.as_str], dtype=th.long)
        self.evaluation_tails = th.tensor([self.class_to_id[c] for c in eval_tails.as_str], dtype=th.long)


    @property
    def deductive_closure_tuples(self):
        if self._deductive_closure_tuples is None:
            self._deductive_closure_tuples = self.create_tuples(self.dataset.deductive_closure_ontology)
        return self._deductive_closure_tuples
        
    def create_tuples(self, ontology):
        raise NotImplementedError

    def get_logits(self, batch):
        raise NotImplementedError

    
    def evaluate_base(self, model, eval_tuples, mode="test",
                      include_deductive_closure=False,
                      exclude_testing_set=False,
                      filter_deductive_closure=False,
                      **kwargs):

        num_heads, num_tails = len(self.evaluation_heads), len(self.evaluation_tails)
        model.eval()
        if not mode in ["valid", "test"]:
            raise ValueError(f"Mode must be either 'valid' or 'test', not {mode}")


        if include_deductive_closure:
            mask1 = (self.deductive_closure_tuples.unsqueeze(1) == self.train_tuples).all(dim=-1).any(dim=-1)
            mask2 = (self.deductive_closure_tuples.unsqueeze(1) == self.valid_tuples).all(dim=-1).any(dim=-1)
            mask = mask1 | mask2
            deductive_closure_tuples = self.deductive_closure_tuples[~mask]

            if exclude_testing_set:
                eval_tuples = deductive_closure_tuples # only deductive closure
            else:
                eval_tuples = th.cat([eval_tuples, deductive_closure_tuples], dim=0)
            
        dataloader = FastTensorDataLoader(eval_tuples, batch_size=self.batch_size, shuffle=False)

        metrics = dict()
        mrr, fmrr = 0, 0
        mr, fmr = 0, 0
        ranks, franks = dict(), dict()

        if mode == "test":
            hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
            f_hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        
            filtering_labels = self.get_filtering_labels(num_heads,
                                                         num_tails,
                                                         self.class_id_to_head_id,
                                                         self.class_id_to_tail_id,
                                                         filter_deductive_closure=filter_deductive_closure)
            if include_deductive_closure:
                deductive_labels = self.get_deductive_labels(num_heads, num_tails, **kwargs)

        num_sides = 2
                
        with th.no_grad():
            for batch, in dataloader:
                if batch.shape[1] == 2:
                    heads, tails = batch[:, 0], batch[:, 1]
                elif batch.shape[1] == 3:
                    heads, tails = batch[:, 0], batch[:, 2]
                else:
                    raise ValueError("Batch shape must be either (n, 2) or (n, 3)")
                aux_heads = heads.clone()
                aux_tails = tails.clone()
        
                batch = batch.to(self.device)
                logits_heads, logits_tails = self.get_logits(model, batch, **kwargs)

                if logits_heads is None:
                    num_sides -= 1
                else:
                    for i, head in enumerate(aux_heads):
                        tail = tails[i]
                        head = th.where(self.evaluation_heads == head)[0].item()
                        tail = th.where(self.evaluation_tails == tail)[0].item()
                        preds = logits_heads[i]

                        if include_deductive_closure:
                            ded_labels = deductive_labels[head].to(preds.device)
                            ded_labels[tail] = 1
                            preds = preds * ded_labels


                        order = th.argsort(preds, descending=False)
                        rank = th.where(order == tail)[0].item() + 1
                        mr += rank
                        mrr += 1 / rank

                        if mode == "test":
                            f_preds = preds * filtering_labels[head].to(preds.device)

                            if include_deductive_closure:
                                # when evaluating with deductive closure
                                # axioms, for a testing axiom we need to
                                # filter the other deductive closure
                                # axioms. Otherwise, we could, in the best
                                # case, score many true axioms at the top
                                # and will never get, for example, good
                                # hits@1.
                                ded_labels = deductive_labels[head].to(preds.device)
                                ded_labels[tail] = 1
                                f_preds = f_preds * ded_labels

                            f_order = th.argsort(f_preds, descending=False)
                            f_rank = th.where(f_order == tail)[0].item() + 1
                            fmr += f_rank
                            fmrr += 1 / f_rank


                        if mode == "test":
                            for k in hits_k:
                                if rank <= int(k):
                                    hits_k[k] += 1

                            for k in f_hits_k:
                                if f_rank <= int(k):
                                    f_hits_k[k] += 1

                            if rank not in ranks:
                                ranks[rank] = 0
                            ranks[rank] += 1

                            if f_rank not in franks:
                                franks[f_rank] = 0
                            franks[f_rank] += 1

                if logits_tails is None:
                    num_sides -= 1
                else:
                            
                    for i, tail in enumerate(aux_tails):
                        head = aux_heads[i]
                        head = th.where(self.evaluation_heads == head)[0].item()
                        tail = th.where(self.evaluation_tails == tail)[0].item()
                        preds = logits_tails[i]

                        if include_deductive_closure:
                            ded_labels = deductive_labels[:, tail].to(preds.device)
                            ded_labels[head] = 1
                            preds = preds * ded_labels

                        order = th.argsort(preds, descending=False)
                        rank = th.where(order == head)[0].item() + 1
                        mr += rank
                        mrr += 1 / rank

                        if mode == "test":
                            f_preds = preds * filtering_labels[:, tail].to(preds.device)

                            if include_deductive_closure:
                                ded_labels = deductive_labels[:, tail].to(preds.device)
                                ded_labels[head] = 1
                                f_preds = f_preds * ded_labels


                            f_order = th.argsort(f_preds, descending=False)
                            f_rank = th.where(f_order == head)[0].item() + 1
                            fmr += f_rank
                            fmrr += 1 / f_rank


                        if mode == "test":
                            for k in hits_k:
                                if rank <= int(k):
                                    hits_k[k] += 1

                            for k in f_hits_k:
                                if f_rank <= int(k):
                                    f_hits_k[k] += 1

                            if rank not in ranks:
                                ranks[rank] = 0
                            ranks[rank] += 1

                            if f_rank not in franks:
                                franks[f_rank] = 0
                            franks[f_rank] += 1
            
                            
            mr = mr / (num_sides * len(eval_tuples))
            mrr = mrr / (num_sides * len(eval_tuples))

            metrics["mr"] = mr
            metrics["mrr"] = mrr

            if mode == "test":
                fmr = fmr / (num_sides * len(eval_tuples))
                fmrr = fmrr / (num_sides * len(eval_tuples))
                auc = compute_rank_roc(ranks, num_tails)
                f_auc = compute_rank_roc(franks, num_tails)

                metrics["f_mr"] = fmr
                metrics["f_mrr"] = fmrr
                metrics["auc"] = auc
                metrics["f_auc"] = f_auc
                
                for k in hits_k:
                    hits_k[k] = hits_k[k] / (num_sides * len(eval_tuples))
                    metrics[f"hits@{k}"] = hits_k[k]
                    
                for k in f_hits_k:
                    f_hits_k[k] = f_hits_k[k] / (num_sides * len(eval_tuples))
                    metrics[f"f_hits@{k}"] = f_hits_k[k]

            metrics = {f"{mode}_{k}": v for k, v in metrics.items()}
            return metrics

        
    def evaluate(self, *args,
                 include_deductive_closure=False,
                 exclude_testing_set=False,
                 filter_deductive_closure=False,
                 **kwargs):

        """
        :param include_deductive_closure: Whether to evaluate using deductive closure axioms as positives. Defaults to False.
        :type include_deductive_closure: bool, optional
        
        :param exclude_testing_set: Whether to exclude the testing set from the evaluation. Defaults to False.
        :type exclude_testing_set: bool, optional
        :param filter_deductive_closure: Whether to filter deductive closure axioms from the evaluation. Defaults to False.
        :type filter_deductive_closure: bool, optional
        """


        
        if not isinstance(include_deductive_closure, bool):
            raise TypeError(msg.get_type_error_message("include_deductive_closure", "bool", type(include_deductive_closure)))

        if not isinstance(exclude_testing_set, bool):
            raise TypeError(msg.get_type_error_message("exclude_testing_set", "bool", type(exclude_testing_set)))

        if not isinstance(filter_deductive_closure, bool):
            raise TypeError(msg.get_type_error_message("filter_deductive_closure", "bool", type(filter_deductive_closure)))


        logger.info(f"Evaluating in device: {self.device}")
        logger.info(f"Evaluating with deductive closure: {include_deductive_closure}")
        logger.info(f"Excluding testing set: {exclude_testing_set}")
        logger.info(f"Filtering deductive closure: {filter_deductive_closure}")


        
        model = args[0]
        mode = kwargs.get("mode")
        
        if mode == "valid":
            eval_tuples = self.valid_tuples
        else:
            eval_tuples = self.test_tuples

        return self.evaluate_base(model, eval_tuples,
                                  include_deductive_closure=include_deductive_closure,
                                  exclude_testing_set=exclude_testing_set,
                                  filter_deductive_closure=filter_deductive_closure,
                                  **kwargs)
    
    



def compute_rank_roc(ranks, num_entities):
    n_tails = num_entities
                    
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_tails)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_tails
    return auc
