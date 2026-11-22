"""Exact answers by array operations equal the set evaluation; sampling is unchanged."""
import time
from collections import defaultdict

import numpy as np
import pytest

import mowl
mowl.init_jvm("1g")

from mowl.models.query import sampling  # noqa: E402
from mowl.models.query.graph import QueryGraph, role_relation  # noqa: E402
from mowl.nn.query.expression import (AnswerSet, anchor, answers, intersect, negate,  # noqa: E402
                                      project, union)


class SetGraph:
    """The graph as it was: Python dictionaries of sets, answers by set operations."""

    def __init__(self, triples, num_entities):
        self.num_entities = num_entities
        self.out, self.inc = defaultdict(set), defaultdict(set)
        for h, r, t in triples:
            self.out[(h, r)].add(t)
            self.inc[t].add((h, r))
        self.universe = set(range(num_entities))

    def successors(self, e, r):
        return self.out.get((e, r), set())


def random_graph(n=60, relations=4, edges=400, seed=0):
    rng = np.random.default_rng(seed)
    triples = set()
    for _ in range(edges):
        h, t, r = int(rng.integers(n)), int(rng.integers(n)), int(rng.integers(relations))
        triples.add((h, r, t))
        triples.add((t, r ^ 1, h))
    triples = sorted(triples)
    return QueryGraph(triples, n, relations), SetGraph(triples, n)


def test_array_answers_equal_set_answers_on_random_queries():
    g, old = random_graph()
    rng = np.random.default_rng(1)

    def random_query(depth):
        k = int(rng.integers(5 if depth else 1))
        if k == 0 or depth == 0:
            return anchor(int(rng.integers(g.num_entities)))
        if k == 1:
            return project(int(rng.integers(g.num_relations)), random_query(depth - 1))
        if k == 2:
            return intersect(random_query(depth - 1), random_query(depth - 1))
        if k == 3:
            return union(random_query(depth - 1), random_query(depth - 1))
        return negate(random_query(depth - 1))

    checked = 0
    for _ in range(300):
        q = random_query(3)
        fast = answers(q, g, g.universe)
        assert isinstance(fast, AnswerSet)
        expected = answers(q, old, old.universe)
        assert fast == expected and expected == fast
        assert len(fast) == len(expected) and set(fast) == expected
        checked += len(expected) > 0
    assert checked > 50


def test_sampled_queries_and_negatives_are_the_same_for_a_fixed_seed():
    n = 40
    rng = np.random.default_rng(3)
    triples = set()
    for i in range(1, n):
        p = int(rng.integers(i))
        triples |= {(i, 0, p), (p, 1, i)}
    for _ in range(40):
        h, t, r = int(rng.integers(n)), int(rng.integers(n)), int(rng.integers(2))
        triples |= {(h, role_relation(r), t), (t, role_relation(r, True), h)}
    triples = sorted(triples)
    fast, old = QueryGraph(triples, n, 6), SetGraph(triples, n)
    templates = sampling.structures(True)
    pool = fast.entities_with_incoming()
    assert pool == sorted(old.inc)
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    done = 0
    for name, template in templates.items():
        for _ in range(15):
            s1 = sampling.sample_query(template, fast, a, pool)
            s2 = sampling.sample_query(template, old, b, pool)
            assert (s1 is None) == (s2 is None), name
            if s1 is None:
                continue
            assert s1[0] == s2[0] and s1[2] == s2[2] and s1[1] == s2[1], name
            n1 = sampling.negative_entities(s1[1], 8, n, a)
            n2 = sampling.negative_entities(s2[1], 8, n, b)
            assert list(n1) == list(n2)
            done += 1
    assert done > 50


def test_the_cost_of_a_sampled_step_does_not_grow_with_the_ontology():
    """A taxonomy with closure: the exact answers of queries starting at an anchor near the
    root are large, and a step still costs about the same at 8 times the size."""
    def step_time(n):
        rng = np.random.default_rng(0)
        parents = {i: int(rng.integers(i)) for i in range(1, n)}
        triples = set()
        for i in range(n):
            j = i
            while True:
                triples |= {(i, 0, j), (j, 1, i)}
                if j == 0:
                    break
                j = parents[j]
        g = QueryGraph(sorted(triples), n, 2)
        templates = {k: v for k, v in sampling.structures(False).items()
                     if k in ("1p", "2i", "3i")}
        pool = g.entities_with_incoming()
        r = np.random.default_rng(1)
        t = time.perf_counter()
        for template in templates.values():
            for _ in range(20):
                sampling.sample_query(template, g, r, pool)
        return time.perf_counter() - t

    small, big = step_time(1000), step_time(8000)
    assert big < 6 * small + 0.05, (small, big)
