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



class BaseRankingEvaluator():
    """
    Base class for ranking evaluation of ontology embedding methods.
    """

    
    def __init__(self, heads, tails, batch_size, device):
        """
        :param heads: The indices of the head entities.
        :type heads: :class:`torch.Tensor`
        :param tails: The indices of the tail entities.
        :type tails: :class:`torch.Tensor`
        :param batch_size: The batch size for evaluation.
        :type batch_size: int
        :param device: The device to use for evaluation.
        :type device: str
        """

        self.batch_size = batch_size
        self.device = device
 
        self.heads = heads.to(self.device)
        self.tails = tails.to(self.device)
        sorted_heads = th.sort(heads)[0]
        sorted_tails = th.sort(tails)[0]
        assert (heads == sorted_heads).all(), "Heads must be sorted."
        assert (tails == sorted_tails).all(), "Tails must be sorted."

        if len(heads) != len(tails):
            logger.warning(f"Detected a different number of evaluation heads and tails. AUC metric will not be accurate in this case if you evaluate in mode='both'")

        
         
        head_idx = th.arange(len(heads), dtype=th.long, device=self.device)
        tail_idx = th.arange(len(tails), dtype=th.long, device=self.device)

        if not (heads == head_idx).all():
            logger.info(f"Head indices are incomplete. This is normal if you are predicting over a subset of entities.")
            max_head = heads.max().item() + 1
            self.mapped_heads = - th.ones(max_head, dtype=th.long, device=self.device)
            self.mapped_heads[heads] = head_idx
        else:
            self.mapped_heads = heads
            
        if not (tails == tail_idx).all():
            logger.info(f"Tail indices are incomplete. This is normal if you are predicting over a subset of entities.")
            max_tail = tails.max().item() + 1
            self.mapped_tails = - th.ones(max_tail, dtype=th.long, device=self.device)
            self.mapped_tails[tails] = tail_idx
        else:
            self.mapped_tails = tails

        
        self.filtering_labels = th.ones((len(self.heads), len(self.tails))).to(self.device)
        
    def update_filtering_labels(self, data):
        if data is None:
            return

        if data.shape[1] == 2:
            heads, tails = data[:, 0], data[:, 1]
        elif data.shape[1] == 3:
            heads, tails = data[:, 0], data[:, 2]
        else:
            raise ValueError("Data must have 2 or 3 columns.")
         
        mapped_heads = self.mapped_heads[heads]
        head_mask = mapped_heads == -1

        mapped_tails = self.mapped_tails[tails]
        tail_mask = mapped_tails == -1

        whole_mask = head_mask | tail_mask
        
        mapped_heads = mapped_heads[~whole_mask]
        mapped_tails = mapped_tails[~whole_mask]
        self.filtering_labels[mapped_heads, mapped_tails] = 10000

    def get_scores(self, evaluation_model, batch):
        logger.warning("Your are using a generic `get_scores` method. Please implement a specific one for your model.")
        return evaluation_model(batch)


    def get_expanded_scores(self, evaluation_model, batch, mode):
        batch_rels = None

        if batch.shape[1] == 2:
            batch_heads, batch_tails = batch[:, 0], batch[:, 1]
        elif batch.shape[1] == 3:
            batch_heads, batch_rels, batch_tails = batch[:, 0], batch[:, 1], batch[:, 2]
        else:
            raise ValueError("Batch must have 2 or 3 columns.")
            
        num_batch_heads, num_batch_tails = len(batch_heads), len(batch_tails)

        if mode in ["head_centric", "both"]:
            batch_heads = batch_heads.repeat_interleave(len(self.tails)).unsqueeze(1)
            eval_tails = th.arange(len(self.tails), device=self.device).repeat(num_batch_heads).unsqueeze(1)
            if batch_rels is None:
                data = th.cat([batch_heads, eval_tails], dim=1)
            else:
                aux_batch_rels = batch_rels.repeat_interleave(len(self.tails)).unsqueeze(1)
                data = th.cat([batch_heads, aux_batch_rels, eval_tails], dim=1)
                
            head_scores = self.get_scores(evaluation_model, data)
            head_scores = head_scores.view(-1, len(self.tails))

        if mode in ["tail_centric", "both"]:
            batch_tails = batch_tails.repeat_interleave(len(self.heads)).unsqueeze(1)
            eval_heads = th.arange(len(self.heads), device=self.device).repeat(num_batch_tails).unsqueeze(1)

            if batch_rels is None:
                data = th.cat([eval_heads, batch_tails], dim=1)
            else:
                aux_batch_rels = batch_rels.repeat_interleave(len(self.heads)).unsqueeze(1)
                data = th.cat([eval_heads, aux_batch_rels, batch_tails], dim=1)
            
            tail_scores = self.get_scores(evaluation_model, data)
            tail_scores = tail_scores.view(-1, len(self.heads))

        if mode == "head_centric":
            return head_scores, None
        elif mode == "tail_centric":
            return None, tail_scores
        elif mode == "both":
            return head_scores, tail_scores
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
 
    @th.no_grad()
    def compute_ranking_metrics(self, evaluation_model, test_data, filter_data=None, mode="head_centric"):
        """
        Compute the ranking metrics for the evaluation model on the test data.
        :param evaluation_model: The evaluation model.
        :type evaluation_model: :class:`torch.nn.Module`
        :param test_data: The test data containing the indices of the embeddings
        :type test_data: :class:`torch.Tensor`
        :param filter_data: The filter data containing the indices of the embeddings
        :type filter_data: :class:`torch.Tensor`
        :param mode: The mode of the evaluation.
        :type mode: str
        :return: The computed ranking metrics.
        :rtype: dict
        """

        if not mode in ["head_centric", "tail_centric", "both"]:
            raise ValueError("Invalid mode. Choose between 'head_centric', 'tail_centric' or 'both'.")

        logger.debug(f"Computing ranking metrics in {mode} mode.")
        logger.debug(f"Test data shape: {test_data.shape}")
        
        evaluation_model.to(self.device)
        evaluation_model.eval()

        num_heads = len(self.heads)
        num_tails = len(self.tails)
        
        self.update_filtering_labels(filter_data)
                                                                                                 
        dataloader = FastTensorDataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        metrics = dict()
        mrr, fmrr = 0, 0
        mr, fmr = 0, 0
        ranks, franks = dict(), dict()

        hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        f_hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        
        
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
            head_scores, tail_scores = self.get_expanded_scores(evaluation_model, batch, mode)
                        
            if head_scores is not None:
                for i, head in enumerate(aux_heads):
                    tail = tails[i]
                    head = th.where(self.heads == head)[0].item()
                    tail = th.where(self.tails == tail)[0].item()
                    preds = head_scores[i]
                    f_preds = preds * self.filtering_labels[head]
                                                                        
                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == tail)[0].item() + 1
                    mr += rank
                    mrr += 1 / rank

                    f_order = th.argsort(f_preds, descending=False)
                    f_rank = th.where(f_order == tail)[0].item() + 1
                    fmr += f_rank
                    fmrr += 1 / f_rank

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

            if tail_scores is not None:

                for i, tail in enumerate(aux_tails):
                    head = aux_heads[i]
                    head = th.where(self.heads == head)[0].item()
                    tail = th.where(self.tails == tail)[0].item()
                    preds = tail_scores[i]
                    f_preds = preds * self.filtering_labels[:, tail]

                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == head)[0].item() + 1
                    mr += rank
                    mrr += 1 / rank

                    f_order = th.argsort(f_preds, descending=False)
                    f_rank = th.where(f_order == head)[0].item() + 1
                    fmr += f_rank
                    fmrr += 1 / f_rank

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

            if mode in ["head_centric", "tail_centric"]:
                divisor = 1
            elif mode == "both":
                divisor = 2
            else:
                raise ValueError(f"Invalid mode: {mode}")

        mr = mr / (divisor * len(test_data))
        mrr = mrr / (divisor * len(test_data))

        metrics["mr"] = mr
        metrics["mrr"] = mrr

        fmr = fmr / (divisor * len(test_data))
        fmrr = fmrr / (divisor * len(test_data))

        if mode == "both":
            num_entities_for_auc = 0.5 * (num_heads + num_tails)
        elif mode == "head_centric":
            num_entities_for_auc = num_tails
        elif mode == "tail_centric":
            num_entities_for_auc = num_heads

        auc = compute_rank_roc(ranks, num_entities_for_auc)
        f_auc = compute_rank_roc(franks, num_entities_for_auc)

        metrics["f_mr"] = fmr
        metrics["f_mrr"] = fmrr
        metrics["auc"] = auc
        metrics["f_auc"] = f_auc

        for k in hits_k:
            hits_k[k] = hits_k[k] / (divisor * len(test_data))
            metrics[f"hits@{k}"] = hits_k[k]

        for k in f_hits_k:
            f_hits_k[k] = f_hits_k[k] / (divisor * len(test_data))
            metrics[f"f_hits@{k}"] = f_hits_k[k]

        return metrics
 

class RankingEvaluator(BaseRankingEvaluator):
    """
    Ranking evaluation class for ontology embedding methods. It encapsulates :class:`BaseRankingEvaluator` to support mOWL datasets
    """

    def __init__(self, dataset, batch_size=16, device="cpu"):
        """
        :param dataset: The mOWL dataset object.
        :type dataset: :class:`mowl.datasets.base.Dataset`
        :param batch_size: The batch size for evaluation.
        :type batch_size: int
        :param device: The device to use for evaluation.
        :type device: str
        """

        self.dataset = dataset
        
        self.class_to_id = {c: i for i, c in enumerate(self.dataset.classes.as_str)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}

        self.relation_to_id = {r: i for i, r in enumerate(self.dataset.object_properties.as_str)}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}

        eval_heads, eval_tails = self.dataset.evaluation_classes
        self.class_id_to_head_id = {self.class_to_id[c]: i for i, c in enumerate(eval_heads.as_str)}
        self.class_id_to_tail_id = {self.class_to_id[c]: i for i, c in enumerate(eval_tails.as_str)}
         
        eval_heads, eval_tails = self.dataset.evaluation_classes
        
        evaluation_heads_tensor = th.tensor([self.class_to_id[c] for c in eval_heads.as_str], dtype=th.long).to(device)
        evaluation_tails_tensor = th.tensor([self.class_to_id[c] for c in eval_tails.as_str], dtype=th.long).to(device)

        super().__init__(evaluation_heads_tensor, evaluation_tails_tensor, batch_size, device)

    def create_tuples(self, ontology):
        """
        Create tuples from the ontology.
        :param ontology: The ontology.
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        :return: The created tuples.
        :rtype: :class:`torch.Tensor`
        """
        raise NotImplementedError

    def evaluate(self, evaluation_model, testing_ontology, filter_ontologies = None, mode="head_centric"):
        """
        Evaluate the model on the testing ontology.
        :param testing_ontology: The testing ontology.
        :type testing_ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        :param filter_ontologies: The filter ontologies.
        :type filter_ontologies: list, optional
        :param mode: The mode of the evaluation.
        :type mode: str
        :return: The computed ranking metrics.
        :rtype: dict
        """
        testing_data = self.create_tuples(testing_ontology)

        filter_data = None
        if filter_ontologies is not None:
            filter_data = []
            for ontology in filter_ontologies:
                filter_data.append(self.create_tuples(ontology))
            filter_data = th.cat(filter_data, dim=0)
            
        return self.compute_ranking_metrics(evaluation_model, testing_data, filter_data=filter_data, mode=mode)
    
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
        self.evaluation_heads = th.tensor([self.class_to_id[c] for c in eval_heads.as_str], dtype=th.long).to(self.device)
        self.evaluation_tails = th.tensor([self.class_to_id[c] for c in eval_tails.as_str], dtype=th.long).to(self.device)


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

        model = model.to(self.device)
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
    
def compute_rank_roc(ranks, num_entities, method="riemann"):
    if method == "riemann":
        fn = riemann_sum
    elif method == "trapz":
        fn = np.trapz
    else:
        raise ValueError(f"Method {method} not recognized.")
    
    num_entities = int(num_entities)
    ranks = {k-1: v for k, v in ranks.items()}
    min_rank = min(ranks.keys())
    assert min_rank >= 0
    
    all_ranks = {k: 0 for k in range(min_rank, num_entities)}
    all_ranks.update(ranks)
    
    ranks = all_ranks

    auc_x = list(ranks.keys())
    auc_x.sort()
    
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
 
    auc = fn(auc_y, auc_x) / (num_entities - 1)
    return auc

def riemann_sum(y, x):
    dx = np.diff(x)
    heights = y[:-1]  # Use left endpoints for rectangle heights
    integral = np.sum(heights * dx)
    return integral
