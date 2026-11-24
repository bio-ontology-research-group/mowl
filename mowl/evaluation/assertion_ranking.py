"""Filtered link prediction on role assertions :math:`R(a, b)` between individuals.

The standard knowledge-base completion protocol (Bordes et al., 2013), as used by BoxLitE
(Lourenço et al., KR 2026, Section 6 and Appendix G.4): every evaluation assertion
:math:`R(a, b)` is ranked twice, as the object of the query :math:`R(a, ?)` and as the
subject of :math:`R(?, b)`, among all individuals of the training signature. In the
filtered setting, candidates :math:`c` such that the corrupted assertion is known to be
true (in the training, validation or test assertions, depending on ``filter_by``) are
removed from the ranking, except the evaluated assertion itself. The mean reciprocal rank
and Hits@k are averaged over all (assertion, direction) pairs.

The model must provide ``model.kb`` (a :class:`mowl.datasets.dllite.DLLiteKB`, the
training signature and ABox) and ``model.role_scores(role, subject=None, obj=None)``
returning one score per individual (higher is better). Validation and test assertions
are read from ``dataset.validation`` / ``dataset.testing`` with
:func:`mowl.datasets.dllite.extract_dllite` in the training signature.
"""

from collections import defaultdict

import numpy as np

from mowl.evaluation.regime import EvaluationRegime

#: Tie policies: rank among equal scores.
TIES = ("ordinal", "optimistic", "pessimistic", "average")


class RoleAssertionRankingRegime(EvaluationRegime):
    """Filtered head and tail ranking of role assertions.

    :param split: ``"test"`` or ``"valid"``.
    :param filter_by: Assertion sets whose members are removed from the candidates, any
        of ``"train"``, ``"valid"``, ``"test"``. Default: all three for ``split="test"``
        and ``("train", "valid")`` for ``split="valid"`` (Appendix G.4).
    :param ties: ``"ordinal"`` (equal scores ordered by individual index, as the stable
        sort of the BoxLitE reference code over IRI-sorted individuals), ``"optimistic"``,
        ``"pessimistic"`` or ``"average"``.
    :param hits: Values of :math:`k` for Hits@k.
    """

    key = "kg-filtered-link-prediction"

    def __init__(self, split="test", filter_by=None, ties="ordinal", hits=(1, 3, 5, 10)):
        if split not in ("test", "valid"):
            raise ValueError("split must be 'test' or 'valid'.")
        if ties not in TIES:
            raise ValueError(f"ties must be one of {TIES}.")
        if filter_by is None:
            filter_by = ("train", "valid", "test") if split == "test" else ("train", "valid")
        unknown = set(filter_by) - {"train", "valid", "test"}
        if unknown:
            raise ValueError(f"Unknown filter sets {sorted(unknown)}.")
        self.split, self.filter_by, self.ties = split, tuple(filter_by), ties
        self.hits = tuple(int(k) for k in hits)
        self.metrics = ("mrr",) + tuple(f"hits@{k}" for k in self.hits)
        self.info = {}

    @staticmethod
    def assertions(model, dataset):
        """``{"train", "valid", "test"}`` role assertion arrays in the model's signature,
        and the extraction reports of the validation and test ontologies."""
        from mowl.datasets.dllite import extract_dllite
        out = {"train": model.kb.role_assertions}
        reports = {}
        for name, attr in (("valid", "validation"), ("test", "testing")):
            ont = getattr(dataset, attr, None) if dataset is not None else None
            if ont is None:
                out[name] = np.zeros((0, 3), dtype=np.int64)
                continue
            kb = extract_dllite(ont, signature=model.kb)
            out[name] = kb.role_assertions
            reports[name] = {"skipped": kb.skipped, "concept_assertions":
                             len(kb.concept_assertions)}
        return out, reports

    def rank(self, scores, target, filtered):
        """Rank of candidate ``target`` given ``scores`` over all candidates and a boolean
        mask of filtered candidates (the target itself is never filtered)."""
        s = scores[target]
        keep = ~filtered
        keep[target] = False
        better = int(np.count_nonzero(keep & (scores > s)))
        equal = keep & (scores == s)
        if self.ties == "optimistic":
            return better + 1
        if self.ties == "pessimistic":
            return better + int(np.count_nonzero(equal)) + 1
        if self.ties == "average":
            return better + int(np.count_nonzero(equal)) / 2 + 1
        return better + int(np.count_nonzero(equal[:target])) + 1

    def ranks(self, model, target, known):
        """Head and tail ranks of every row ``[R, a, b]`` of ``target``."""
        n = len(model.kb.individuals)
        known_obj, known_sub = defaultdict(set), defaultdict(set)
        for r, a, b in known:
            known_obj[(r, a)].add(b)
            known_sub[(r, b)].add(a)
        tails, heads = defaultdict(list), defaultdict(list)
        for r, a, b in target:
            tails[(r, a)].append(b)
            heads[(r, b)].append(a)
        out = []
        for direction, groups, known_map in (("tail", tails, known_obj),
                                             ("head", heads, known_sub)):
            for (r, x), answers in groups.items():
                scores = np.asarray(
                    model.role_scores(r, subject=x) if direction == "tail"
                    else model.role_scores(r, obj=x), dtype=float)
                mask = np.zeros(n, dtype=bool)
                mask[list(known_map.get((r, x), ()))] = True
                for y in answers:
                    out.append(self.rank(scores, y, mask))
        return np.asarray(out, dtype=float)

    def evaluate(self, model, dataset):
        sets, reports = self.assertions(model, dataset)
        target = sets[self.split]
        known = {tuple(int(v) for v in row) for name in self.filter_by for row in sets[name]}
        ranks = self.ranks(model, [tuple(int(v) for v in row) for row in target], known)
        if len(ranks) == 0:
            raise ValueError(f"No {self.split} role assertions to evaluate.")
        metrics = {"mrr": float(np.mean(1.0 / ranks))}
        for k in self.hits:
            metrics[f"hits@{k}"] = float(np.mean(ranks <= k))
        self.info = {"split": self.split, "assertions": int(len(target)),
                     "ranked": int(len(ranks)), "filter_by": list(self.filter_by),
                     "ties": self.ties, "extraction": reports}
        return {"all": metrics}
