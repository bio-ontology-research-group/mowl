"""Two builds of one ontology in one process give the same rows, in the same order.

The ontology reader and the normaliser yield axioms in the order of a Java hash set, which
differs from one build to the next; negatives are drawn per row, so equal seeds gave unequal
training runs. The rows of every form of the EL and ALC datasets, and the edges of the
projectors whose Java code yields hash-set order, are now in a documented order.
"""
from unittest import TestCase

import mowl
mowl.init_jvm("1g")

import torch as th

from mowl.datasets import ALCDataset, ELDataset
from mowl.projection import DL2VecProjector, GDAProjector, OWL2VecStarProjector
from tests.baselines.pairs_boxrel import tiny_ontology
from tests.baselines.pairs_logic import alc_dataset

BUILDS = 4


def rows_of(dataset):
    return {k: v.data.tolist() for k, v in dataset.get_gci_datasets().items()}


class ELRows(TestCase):

    def builds(self, **kwargs):
        return [ELDataset(tiny_ontology().ontology, load_role_axioms=True, **kwargs)
                for _ in range(BUILDS)]

    def test_every_form_has_the_same_rows_on_every_build(self):
        built = [rows_of(d) for d in self.builds()]
        self.assertGreater(len(built[0]), 5)
        for other in built[1:]:
            self.assertEqual(other, built[0])

    def test_the_index_dictionaries_are_the_same_on_every_build(self):
        built = self.builds()
        for d in built:
            d.load()
        for d in built[1:]:
            for name in ("class_index_dict", "object_property_index_dict"):
                self.assertEqual(list(getattr(d, name).items()),
                                 list(getattr(built[0], name).items()))
            self.assertEqual(list(d._individual_index_dict.items()),
                             list(built[0]._individual_index_dict.items()))

    def test_the_global_random_state_is_not_used_or_moved(self):
        import random
        random.seed(1)
        first = rows_of(self.builds()[0])
        state = random.getstate()
        random.seed(99)
        second = rows_of(self.builds()[0])
        self.assertEqual(first, second)
        random.seed(99)
        random.random()
        random.setstate(state)
        self.assertEqual(random.getstate(), state)

    def test_without_shuffle_the_rows_are_sorted(self):
        for rows in rows_of(self.builds(shuffle=False)[0]).values():
            self.assertEqual(rows, sorted(rows))

    def test_the_shuffle_is_a_permutation_of_the_sorted_rows_and_follows_its_seed(self):
        plain = rows_of(self.builds(shuffle=False)[0])
        seeded = rows_of(self.builds(shuffle_seed=3)[0])
        self.assertEqual(seeded, rows_of(self.builds(shuffle_seed=3)[0]))
        for form, rows in seeded.items():
            self.assertEqual(sorted(rows), plain[form])
        self.assertTrue(any(seeded[f] != plain[f] for f in seeded if len(seeded[f]) > 2))

    def test_shuffle_seed_none_follows_the_global_seed(self):
        import random
        out = []
        for _ in range(2):
            random.seed(5)
            out.append(rows_of(self.builds(shuffle_seed=None)[0]))
        self.assertEqual(out[0], out[1])


class ALCRows(TestCase):

    def datasets(self):
        out = []
        for _ in range(BUILDS):
            data = alc_dataset()
            alc = ALCDataset(data.ontology, data)
            tables, rest = alc.get_datasets()
            out.append(({str(k): v.tensors[0].tolist() for k, v in tables.items()},
                        [str(a) for a in rest], [str(k) for k in alc.grouped_axioms]))
        return out

    def test_patterns_and_rows_are_the_same_on_every_build(self):
        built = self.datasets()
        self.assertGreater(len(built[0][0]), 3)
        for other in built[1:]:
            self.assertEqual(other, built[0])
        # the patterns are listed sorted
        self.assertEqual(built[0][2], sorted(built[0][2]))
        for rows in built[0][0].values():
            self.assertEqual(rows, sorted(rows))


class Edges(TestCase):

    def test_projections_list_the_same_edges_in_the_same_order(self):
        for make in (lambda: DL2VecProjector(True), lambda: OWL2VecStarProjector(True),
                     lambda: GDAProjector()):
            built = []
            for _ in range(BUILDS):
                edges = make().project(tiny_ontology().ontology)
                built.append([(e.src, e.rel, e.dst) for e in edges])
            self.assertGreater(len(built[0]), 3)
            for other in built[1:]:
                self.assertEqual(other, built[0])
            self.assertEqual(built[0], sorted(built[0]))
