from unittest import TestCase
from mowl.projection import projector_factory, TaxonomyProjector, TaxonomyWithRelationsProjector, \
    DL2VecProjector, OWL2VecStarProjector


class TestProjectionFactory(TestCase):

    def test_create_taxonomy_projector(self):
        taxonomy_projector = projector_factory("taxonomy")
        self.assertIsInstance(taxonomy_projector, TaxonomyProjector)

    def test_create_taxonomy_with_rels_projector(self):
        taxonomy_with_rels_projector = projector_factory("taxonomy_rels", taxonomy=True)
        self.assertIsInstance(taxonomy_with_rels_projector, TaxonomyWithRelationsProjector)

    def test_create_dl2vec_projector(self):
        dl2vec_projector = projector_factory("dl2vec")
        self.assertIsInstance(dl2vec_projector, DL2VecProjector)

    def test_create_owl2vecstar_projector(self):
        owl2vecstar_projector = projector_factory("owl2vecstar")
        self.assertIsInstance(owl2vecstar_projector, OWL2VecStarProjector)
