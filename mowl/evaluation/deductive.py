"""Evaluation with the deductive closure, as in DELE [mashkova2024dele]_ (Section 6.2,
Tables 3-6) and its NeSy 2024 predecessor [mashkova2024nesy]_ (Section 4.5, Appendix E).

A knowledge base entails many axioms that are not asserted, and a model-constructing
embedding should rank them high. DELE therefore evaluates knowledge base completion in a
way that treats the deductive closure explicitly:

- **Test set.** Test axioms (:math:`A \\sqsubseteq B` or :math:`A \\sqsubseteq \\exists r.B`)
  that are entailed by the training ontology are removed, so that only "novel" knowledge is
  tested (DELE, Sections 6.1.1 and 6.1.2). Alternatively, the entailed but not asserted
  axioms themselves can be used as the test set (NeSy 2024, Appendix Figure 12).
- **Ranking.** For a test axiom with subclass :math:`A`, every candidate :math:`B` of the
  candidate set (the evaluation classes) is scored with the model's loss for the axiom
  type, and the rank of the true :math:`B` is taken (ties get the average rank).
- **Non-filtered (NF) and filtered (F) metrics.** Filtered metrics remove from the ranking
  every candidate whose axiom is in the deductive closure of the training and testing
  ontologies together (Tables 3, 5, 6, "F"), except the test axiom itself.
- **Metrics.** Hits@n; *macro* mean rank and AUC over all test axioms; *micro* mean rank
  and AUC, computed per subclass :math:`A` and averaged over the subclasses that occur in
  the test set (DELE, Eqs. 19-22).

:class:`DeductiveRankingEvaluator` does the ranking for given test and filter sets;
:class:`DELEEvaluation` is the complete evaluation regime (an
:class:`~mowl.evaluation.regime.EvaluationRegime`).
"""

import logging
from collections import defaultdict

import numpy as np
import torch as th

from mowl.evaluation.base import RankingEvaluator, BaseRankingEvaluator
from mowl.evaluation.regime import EvaluationRegime

logger = logging.getLogger(__name__)

#: Metrics reported by :class:`DeductiveRankingEvaluator` (each also with prefix ``f_``).
BASE_METRICS = ("hits@1", "hits@3", "hits@10", "hits@100", "macro_mr", "micro_mr", "mrr",
                "macro_auc", "micro_auc")


def rank_auc(ranks, num_candidates):
    """Area under the rank-based ROC curve of :func:`mowl.evaluation.base.compute_rank_roc`
    in closed form, :math:`\\frac{1}{n}\\sum_i \\frac{N - r_i}{N - 1}`. For integer ranks it
    equals ``compute_rank_roc`` with the default Riemann sum; unlike it, it also accepts the
    fractional ranks produced by averaging ties.

    :param ranks: Ranks (1 is best).
    :param num_candidates: Size :math:`N` of the candidate set.
    :rtype: float
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    if len(ranks) == 0 or num_candidates < 2:
        return float("nan")
    return float(np.mean((num_candidates - ranks) / (num_candidates - 1)))


def summarise_ranks(ranks, heads, num_candidates, hits_at=(1, 3, 10, 100), prefix=""):
    """Turn per-axiom ranks into DELE's metrics.

    :param ranks: Rank of each test axiom.
    :param heads: Subclass index of each test axiom (groups for the micro metrics).
    :param num_candidates: Size of the candidate set (for the AUC).
    :rtype: dict
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    heads = np.asarray(heads)
    out = {}
    if len(ranks) == 0:
        return out
    for k in hits_at:
        out[f"{prefix}hits@{k}"] = float(np.mean(ranks <= k))
    out[f"{prefix}macro_mr"] = float(np.mean(ranks))
    out[f"{prefix}mrr"] = float(np.mean(1.0 / ranks))
    out[f"{prefix}macro_auc"] = rank_auc(ranks, num_candidates)
    groups = defaultdict(list)
    for h, r in zip(heads.tolist(), ranks.tolist()):
        groups[h].append(r)
    out[f"{prefix}micro_mr"] = float(np.mean([np.mean(v) for v in groups.values()]))
    out[f"{prefix}micro_auc"] = float(np.mean([rank_auc(v, num_candidates)
                                               for v in groups.values()]))
    return out


def tie_rank(scores, target, valid=None, tie_method="average"):
    """Rank of position ``target`` in ``scores`` (lower score is better).

    :param scores: 1-D tensor of candidate scores.
    :param target: Position of the true candidate.
    :param valid: Optional boolean mask of the candidates that take part in the ranking
        (the target always does).
    :param tie_method: ``"average"`` (as ``scipy.stats.rankdata``, used by DELE),
        ``"min"`` (optimistic) or ``"max"`` (pessimistic).
    :rtype: float
    """
    s_t = scores[target]
    less = scores < s_t
    equal = scores == s_t
    if valid is not None:
        valid = valid.clone()
        valid[target] = True
        less = less & valid
        equal = equal & valid
    n_less = int(less.sum())
    n_eq = int(equal.sum())  # includes the target
    if tie_method == "average":
        return n_less + (n_eq + 1) / 2.0
    if tie_method == "min":
        return float(n_less + 1)
    if tie_method == "max":
        return float(n_less + n_eq)
    raise ValueError(f"Unknown tie_method {tie_method!r}")


class DeductiveRankingEvaluator(RankingEvaluator):
    """Head-centric ranking evaluator with DELE's filtering and macro/micro metrics.

    Indices are those of ``dataset.classes`` and ``dataset.object_properties`` (the prefix
    of an :class:`~mowl.base_models.elmodel.EmbeddingELModel`'s index dictionaries).

    :param dataset: The mOWL dataset.
    :param gci_name: Axiom type evaluated: ``"gci0"`` (:math:`A \\sqsubseteq B`) or
        ``"gci2"`` (:math:`A \\sqsubseteq \\exists r.B`).
    :param relation: IRI of :math:`r` (``gci2`` only); ``None`` takes every role.
    :param heads: IRIs (or class indices) of the subclass candidates; defaults to the first
        element of ``dataset.evaluation_classes``.
    :param tails: IRIs (or class indices) of the superclass/filler candidates; defaults to
        the second element of ``dataset.evaluation_classes``.
    :param batch_size: Number of test axioms scored at once.
    :param device: Torch device.
    :param tie_method: See :func:`tie_rank`. Defaults to ``"average"`` (DELE).
    :param hits_at: Cut-offs of Hits@n.
    """

    def __init__(self, dataset, gci_name="gci0", relation=None, heads=None, tails=None,
                 batch_size=16, device="cpu", tie_method="average", hits_at=(1, 3, 10, 100)):
        if gci_name not in ("gci0", "gci2"):
            raise ValueError("gci_name must be 'gci0' or 'gci2'.")
        self.dataset = dataset
        self.gci_name = gci_name
        self.relation = relation
        self.tie_method = tie_method
        self.hits_at = tuple(hits_at)

        self.class_to_id = {c: i for i, c in enumerate(dataset.classes.as_str)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}
        self.relation_to_id = {r: i for i, r in enumerate(dataset.object_properties.as_str)}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}

        if heads is None or tails is None:
            eval_heads, eval_tails = dataset.evaluation_classes
            heads = eval_heads.as_str if heads is None else heads
            tails = eval_tails.as_str if tails is None else tails
        heads_t = th.tensor(sorted({self._cid(h) for h in heads}), dtype=th.long)
        tails_t = th.tensor(sorted({self._cid(t) for t in tails}), dtype=th.long)
        BaseRankingEvaluator.__init__(self, heads_t, tails_t, batch_size, device)
        self._tail_pos = {int(t): i for i, t in enumerate(tails_t.tolist())}
        self._head_set = set(heads_t.tolist())

    def _cid(self, c):
        if isinstance(c, (int, np.integer)):
            return int(c)
        return self.class_to_id[str(c)]

    # ---- test-set construction ------------------------------------------------------

    def create_tuples(self, ontology):
        """Index tuples of the axioms of ``ontology`` of the evaluated type, with named
        classes on both sides (and the evaluated role for ``gci2``).

        :rtype: :class:`torch.Tensor` of shape ``(n, 2)`` or ``(n, 3)``
        """
        from org.semanticweb.owlapi.model import AxiomType, OWLClass, OWLObjectSomeValuesFrom
        rows = []
        for ax in ontology.getAxioms(AxiomType.SUBCLASS_OF):
            sub, sup = ax.getSubClass(), ax.getSuperClass()
            if not isinstance(sub, OWLClass):
                continue
            a = str(sub.toStringID())
            if a not in self.class_to_id:
                continue
            if self.gci_name == "gci0":
                if isinstance(sup, OWLClass):
                    b = str(sup.toStringID())
                    if b in self.class_to_id:
                        rows.append((self.class_to_id[a], self.class_to_id[b]))
            elif isinstance(sup, OWLObjectSomeValuesFrom):
                filler = sup.getFiller()
                prop = sup.getProperty()
                if not isinstance(filler, OWLClass) or prop.isAnonymous():
                    continue
                r = str(prop.asOWLObjectProperty().toStringID())
                if self.relation is not None and r != self.relation:
                    continue
                b = str(filler.toStringID())
                if b in self.class_to_id and r in self.relation_to_id:
                    rows.append((self.class_to_id[a], self.relation_to_id[r],
                                 self.class_to_id[b]))
        width = 2 if self.gci_name == "gci0" else 3
        if not rows:
            return th.empty((0, width), dtype=th.long)
        return th.tensor(sorted(set(rows)), dtype=th.long)

    @staticmethod
    def remove_entailed(test_data, closure, gci_name):
        """Drop the test axioms that ``closure`` entails (DELE, Section 6.1.1: "we remove
        entailed axioms from the test set ... based on the precomputed deductive closure of
        the train ontology").

        :rtype: :class:`torch.Tensor`
        """
        if len(test_data) == 0:
            return test_data
        return test_data[~closure.contains(gci_name, test_data)]

    def get_scores(self, model, batch):
        return model(batch, self.gci_name)

    # ---- ranking ----------------------------------------------------------------------

    def _filter_mask(self, head, rel, filter_index, closure):
        mask = th.zeros(len(self.tails), dtype=th.bool, device=self.device)
        key = (head, rel) if self.gci_name == "gci2" else head
        for t in filter_index.get(key, ()):
            pos = self._tail_pos.get(t)
            if pos is not None:
                mask[pos] = True
        if closure is not None:
            entailed = closure.entailed_tails(self.gci_name, head, rel)
            if entailed is None:
                mask[:] = True
            else:
                for t in entailed:
                    pos = self._tail_pos.get(t)
                    if pos is not None:
                        mask[pos] = True
        return mask

    @th.no_grad()
    def compute_ranking_metrics(self, evaluation_model, test_data, filter_data=None,
                                mode="head_centric", filter_closure=None):
        """Rank every test axiom against all candidate tails.

        :param evaluation_model: Callable ``(data, gci_name) -> scores`` (e.g. an
            :class:`~mowl.nn.ELModule`); lower scores are better.
        :param test_data: Test axioms, shape ``(n, 2)`` (gci0) or ``(n, 3)`` (gci2).
        :param filter_data: Axioms to filter from the ranking (any known positives), same
            layout.
        :param filter_closure: A :class:`~mowl.training.DeductiveClosure` in the dataset's
            index space; every candidate it entails is filtered.
        :param mode: Only ``"head_centric"`` (DELE ranks superclasses / fillers).
        :return: NF metrics (``hits@k``, ``macro_mr``, ``micro_mr``, ``mrr``,
            ``macro_auc``, ``micro_auc``), the same with prefix ``f_`` for the filtered
            ranking, ``nf_minus_f_macro_mr``, ``nf_minus_f_micro_mr`` and ``num_test``.
        :rtype: dict
        """
        if mode != "head_centric":
            raise ValueError("DeductiveRankingEvaluator supports mode='head_centric' only.")
        if hasattr(evaluation_model, "eval"):
            evaluation_model.eval()

        keep = []
        for row in test_data.tolist():
            if row[0] in self._head_set and row[-1] in self._tail_pos:
                keep.append(row)
        if len(keep) < len(test_data):
            logger.warning("%d test axioms have a subclass or superclass outside the "
                           "candidate sets and are skipped.", len(test_data) - len(keep))
        width = test_data.shape[1] if test_data.dim() == 2 else (
            2 if self.gci_name == "gci0" else 3)
        test = th.tensor(keep, dtype=th.long).reshape(-1, width).to(self.device)

        filter_index = defaultdict(set)
        if filter_data is not None:
            for row in filter_data.tolist():
                key = (row[0], row[1]) if self.gci_name == "gci2" else row[0]
                filter_index[key].add(row[-1])

        ranks, franks, heads = [], [], []
        mask_cache = {}
        for start in range(0, len(test), self.batch_size):
            batch = test[start:start + self.batch_size]
            scores, _ = self.get_expanded_scores(evaluation_model, batch, "head_centric")
            for i, row in enumerate(batch.tolist()):
                head, tail = row[0], row[-1]
                rel = row[1] if self.gci_name == "gci2" else None
                preds = scores[i].flatten()
                target = self._tail_pos[tail]
                ranks.append(tie_rank(preds, target, tie_method=self.tie_method))
                key = (head, rel)
                if key not in mask_cache:
                    mask_cache[key] = self._filter_mask(head, rel, filter_index,
                                                        filter_closure)
                valid = ~mask_cache[key]
                franks.append(tie_rank(preds, target, valid=valid,
                                       tie_method=self.tie_method))
                heads.append(head)

        n = len(self.tails)
        metrics = {"num_test": len(ranks)}
        metrics.update(summarise_ranks(ranks, heads, n, self.hits_at))
        metrics.update(summarise_ranks(franks, heads, n, self.hits_at, prefix="f_"))
        if ranks:
            metrics["nf_minus_f_macro_mr"] = metrics["macro_mr"] - metrics["f_macro_mr"]
            metrics["nf_minus_f_micro_mr"] = metrics["micro_mr"] - metrics["f_micro_mr"]
        return metrics

    def evaluate(self, evaluation_model, testing_ontology, filter_ontologies=None,
                 mode="head_centric", filter_closure=None):
        """Like :meth:`RankingEvaluator.evaluate`; ``testing_ontology`` may also be an index
        tensor, and a ``filter_closure`` may be given."""
        if th.is_tensor(testing_ontology):
            test_data = testing_ontology
        else:
            test_data = self.create_tuples(testing_ontology)
        filter_data = None
        if filter_ontologies is not None:
            filter_data = th.cat([self.create_tuples(o) for o in filter_ontologies], dim=0)
        return self.compute_ranking_metrics(evaluation_model, test_data,
                                            filter_data=filter_data, mode=mode,
                                            filter_closure=filter_closure)


def _union_ontology(*ontologies):
    from java.util import HashSet
    from mowl.owlapi import OWLAPIAdapter
    adapter = OWLAPIAdapter()
    ont = adapter.create_ontology("http://mowl.borg/evaluation/deductive/union")
    axioms = HashSet()
    for o in ontologies:
        if o is not None:
            axioms.addAll(o.getAxioms())
    adapter.owl_manager.addAxioms(ont, axioms)
    return ont


class DELEEvaluation(EvaluationRegime):
    """The evaluation regime of DELE (Section 6.2, Tables 3-6).

    Steps, for a model trained on ``dataset``:

    1. Test axioms of type ``gci_name`` (with role ``relation`` for ``gci2``) are read from
       ``dataset.testing``. With ``remove_entailed_test=True`` (default) those entailed by
       the deductive closure of the training ontology are removed (DELE, Sections 6.1.1,
       6.1.2). With ``test_set="closure"`` the test set is instead the closure axioms of the
       training ontology that are not asserted in it (NeSy 2024, Appendix Fig. 12).
    2. Every test axiom is ranked against the candidate tails (``tails``, default the
       dataset's evaluation classes), ties getting the average rank.
    3. NF metrics use the plain ranking. F metrics remove from the ranking every candidate
       entailed by the closure of the ``filter_with`` ontologies (default ``"train+test"``,
       as in Tables 3, 5, 6; ``"train"`` filters with the training closure plus the
       asserted training axioms, as in NeSy 2024 Table 1). The test axiom itself is never
       removed.

    The closures are computed with :class:`mowl.training.DeductiveClosure` (``reasoner``
    ``"canonical"`` or ``"elk"``).

    :param gci_name: ``"gci0"`` (subsumption prediction) or ``"gci2"`` (e.g. protein
        function or interaction prediction).
    :param relation: Role IRI for ``gci2``.
    :param heads: Candidate subclasses (IRIs); default ``dataset.evaluation_classes[0]``.
    :param tails: Candidate superclasses/fillers (IRIs); default
        ``dataset.evaluation_classes[1]``.
    :param remove_entailed_test: Drop test axioms entailed by the training ontology.
    :param test_set: ``"test"`` (default) or ``"closure"``.
    :param filter_with: ``"train+test"`` (default), ``"train"`` or ``None`` (F = NF).
    :param reasoner: ``"canonical"`` or ``"elk"``.
    :param batch_size: Test axioms scored at once.
    :param device: Torch device.
    :param cache_dir: Cache directory for the closures' hierarchies.
    """

    key = "dele-evaluation"
    metrics = ("hits@10", "f_hits@10", "hits@100", "f_hits@100", "macro_mr", "f_macro_mr",
               "nf_minus_f_macro_mr", "micro_mr", "f_micro_mr", "nf_minus_f_micro_mr",
               "macro_auc", "f_macro_auc", "micro_auc", "f_micro_auc")
    subsets = ("all",)

    def __init__(self, gci_name="gci0", relation=None, heads=None, tails=None,
                 remove_entailed_test=True, test_set="test", filter_with="train+test",
                 reasoner="canonical", batch_size=16, device="cpu", cache_dir=None):
        if test_set not in ("test", "closure"):
            raise ValueError("test_set must be 'test' or 'closure'.")
        if filter_with not in ("train+test", "train", None):
            raise ValueError("filter_with must be 'train+test', 'train' or None.")
        self.gci_name = gci_name
        self.relation = relation
        self.heads = heads
        self.tails = tails
        self.remove_entailed_test = remove_entailed_test
        self.test_set = test_set
        self.filter_with = filter_with
        self.reasoner = reasoner
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self.last_test_data = None

    def evaluator(self, dataset):
        """The :class:`DeductiveRankingEvaluator` used for ``dataset``."""
        return DeductiveRankingEvaluator(dataset, self.gci_name, relation=self.relation,
                                         heads=self.heads, tails=self.tails,
                                         batch_size=self.batch_size, device=self.device)

    def _closure(self, ontology, ev):
        from mowl.training.closure import DeductiveClosure
        return DeductiveClosure.from_ontology(
            ontology, class_index_dict=ev.class_to_id,
            object_property_index_dict=ev.relation_to_id, reasoner=self.reasoner,
            cache_dir=self.cache_dir)

    def test_data(self, dataset, ev=None, train_closure=None):
        """The test axioms after steps 1 of the class description, as an index tensor."""
        ev = ev or self.evaluator(dataset)
        if self.test_set == "closure" or self.remove_entailed_test:
            train_closure = train_closure or self._closure(dataset.ontology, ev)
        if self.test_set == "closure":
            rel = None
            if self.gci_name == "gci2" and self.relation is not None:
                rel = [ev.relation_to_id[self.relation]]
            entailed = train_closure.axioms(self.gci_name, heads=ev.heads.tolist(),
                                            relations=rel, tails=ev.tails.tolist())
            asserted = {tuple(r) for r in ev.create_tuples(dataset.ontology).tolist()}
            rows = [r for r in entailed.tolist() if tuple(r) not in asserted]
            width = 2 if self.gci_name == "gci0" else 3
            return th.tensor(rows, dtype=th.long).reshape(-1, width)
        test = ev.create_tuples(dataset.testing)
        if self.remove_entailed_test:
            test = ev.remove_entailed(test, train_closure, self.gci_name)
        return test

    def evaluate(self, model, dataset):
        """Evaluate ``model`` on ``dataset``.

        :return: ``{"all": metrics}`` with the metrics of
            :meth:`DeductiveRankingEvaluator.compute_ranking_metrics`.
        """
        ev = self.evaluator(dataset)
        train_closure = None
        if self.test_set == "closure" or self.remove_entailed_test or \
                self.filter_with == "train":
            train_closure = self._closure(dataset.ontology, ev)
        test = self.test_data(dataset, ev, train_closure)
        self.last_test_data = test

        filter_closure, filter_data = None, None
        if self.filter_with == "train+test":
            filter_closure = self._closure(_union_ontology(dataset.ontology, dataset.testing),
                                           ev)
        elif self.filter_with == "train":
            filter_closure = train_closure
            filter_data = ev.create_tuples(dataset.ontology)

        scorer = model.module if hasattr(model, "module") else model
        metrics = ev.compute_ranking_metrics(scorer, test, filter_data=filter_data,
                                             filter_closure=filter_closure)
        return {"all": metrics}
