"""Hits@k of assertion queries: ``C(?)`` and ``r(a, ?)`` over the individuals.

The evaluation of the code of FaithEL (Lacerda, Ozaki and Guimarães, RuleML+RR 2024,
``utils.py`` of https://github.com/victorlacerdab/FaithEL_method), written against the model
scoring interface of :mod:`mowl.evaluation.model_scoring`:

* *concept queries*: for every class that occurs in a held-out class assertion, all
  individuals are ranked by the model's cost of ``C(a)``;
* *role queries*: for every pair ``(a, r)`` that occurs in held-out role assertions
  ``r(a, b)``, all individuals are ranked as the filler ``b``.

A query is a hit at ``k`` if **any** of its held-out answers is among the ``k`` best
candidates (the code counts one hit per query, not per answer). Candidates are the
individuals of the model; equal costs are ordered by individual index. The code does not
filter known answers (``filtered=False``); with ``filtered=True`` the answers that are
training or test assertions of the same query are removed from the ranking, except the
best-ranked held-out answer of the query, the usual filtered setting.

World assumption: closed world over stated axioms. A candidate that is neither a training
nor a held-out assertion counts as wrong, although it may be entailed or true.
"""

import numpy as np
import torch as th

from mowl.evaluation.axiom_ranking import el_rows
from mowl.evaluation.regime import EvaluationRegime


class AssertionQueryHitsRegime(EvaluationRegime):
    """Hits@k and mean reciprocal rank (of the best held-out answer) of ``C(?)`` and
    ``r(a, ?)`` queries.

    The held-out assertions are the class and object property assertions of
    ``dataset.testing``; the model must score ``class_assertion`` and
    ``object_property_assertion``.

    :param hits: Values of :math:`k`.
    :param filtered: Remove other known answers of the query from the ranking.
    :param batch_size: Candidate rows scored at a time.
    """

    key = "assertion-query-hits"
    subsets = ("concept", "role")

    def __init__(self, hits=(1, 3, 10, 100), filtered=False, batch_size=65536):
        super().__init__()
        self.hits = tuple(int(k) for k in hits)
        self.filtered = bool(filtered)
        self.batch_size = int(batch_size)
        self.metrics = tuple(f"hits@{k}" for k in self.hits) + ("mrr",)
        self.info = {}

    @staticmethod
    def _answers(rows, key_columns, answer_column):
        out = {}
        for row in rows.tolist():
            out.setdefault(tuple(row[i] for i in key_columns), set()).add(row[answer_column])
        return out

    @th.no_grad()
    def _costs(self, model, scorer, form, query, num):
        """Costs of all ``num`` individuals as the free position of ``query``."""
        device = getattr(model, "device", "cpu")
        free = th.arange(num)
        if form == "class_assertion":
            rows = th.stack([free, th.full((num,), query[0])], dim=1)
        else:
            rows = th.stack([th.full((num,), query[0]), th.full((num,), query[1]), free],
                            dim=1)
        out = [scorer(model, form, rows[i:i + self.batch_size].to(device)).detach().cpu()
               for i in range(0, num, self.batch_size)]
        return th.cat(out).numpy().astype(float)

    def _summary(self, ranks):
        if not ranks:
            return {**{f"hits@{k}": float("nan") for k in self.hits}, "mrr": float("nan")}
        ranks = np.asarray(ranks, dtype=float)
        return {**{f"hits@{k}": float((ranks <= k).mean()) for k in self.hits},
                "mrr": float((1.0 / ranks).mean())}

    def _ranks(self, model, scorer, form, test, known, num):
        ranks = []
        for query, answers in test.items():
            costs = self._costs(model, scorer, form, query, num)
            if self.filtered:
                drop = known.get(query, set()) | answers
                costs = costs.copy()
                best = min(answers, key=lambda a: (costs[a], a))
                for a in drop:
                    if a != best:
                        costs[a] = np.inf
            order = np.lexsort((np.arange(num), costs))      # stable on the index
            position = np.empty(num, dtype=np.int64)
            position[order] = np.arange(1, num + 1)
            ranks.append(int(min(position[a] for a in answers)))
        return ranks

    def evaluate(self, model, dataset):
        from mowl.evaluation.el_scoring import resolve_scorer
        scorer, name = resolve_scorer(model, "model")
        module = getattr(model, "module", None)
        if module is not None and hasattr(module, "eval"):
            module.eval()
        num = len(model.individual_index_dict)
        test, skipped = el_rows(model, dataset.testing)
        train, _ = el_rows(model, dataset.ontology)
        empty = {"class_assertion": th.zeros((0, 2), dtype=th.long),
                 "object_property_assertion": th.zeros((0, 3), dtype=th.long)}
        for form, default in empty.items():
            test.setdefault(form, default)
            train.setdefault(form, default)
        concept_test = self._answers(test["class_assertion"], (1,), 0)
        concept_known = self._answers(train["class_assertion"], (1,), 0)
        role_test = self._answers(test["object_property_assertion"], (0, 1), 2)
        role_known = self._answers(train["object_property_assertion"], (0, 1), 2)
        concept = self._ranks(model, scorer, "class_assertion", concept_test, concept_known,
                              num)
        role = self._ranks(model, scorer, "object_property_assertion", role_test,
                           role_known, num)
        self.info = {"scoring": name, "individuals": num, "concept_queries": len(concept),
                     "role_queries": len(role),
                     "test_class_assertions": int(len(test["class_assertion"])),
                     "test_role_assertions": int(len(test["object_property_assertion"])),
                     "skipped_axioms": int(skipped), "filtered": self.filtered}
        return {"concept": self._summary(concept), "role": self._summary(role)}
