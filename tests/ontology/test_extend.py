from mowl.ontology.extend import insert_annotations
from unittest import TestCase
from tests.datasetFactory import FamilyDataset
from mowl.datasets import PathDataset
from mowl.projection import TaxonomyWithRelationsProjector


class TestExtend(TestCase):

    def test_parameter_type_checking_method_insert_annotations(self):
        """This should test the type checking for parameters of the method insert_anootations"""

        # ontology_file str
        self.assertRaisesRegex(TypeError, "Parameter ontology_file must be of type str",
                               insert_annotations, 1, [],)

        # annotations list
        self.assertRaisesRegex(TypeError, "Parameter annotations must be of type list",
                               insert_annotations, "ontology_file", 1)

        # out_file str optional
        self.assertRaisesRegex(TypeError, "Optional parameter out_file must be of type str",
                               insert_annotations, "ontology_file", [], out_file=1)

        # verbose bool optional
        self.assertRaisesRegex(TypeError, "Optional parameter verbose must be of type bool",
                               insert_annotations, "ontology_file", [], verbose=1)

    def test_correctness_method_insert_annotations(self):
        """This should test the correctness of the method insert_annotations"""
        _ = FamilyDataset()

        root = "tests/ontology/"
        annotation_data_1 = (root + "fixtures/family.tsv", "http://should_have", True)
        annotations = [annotation_data_1]  # There  could be more than 1 annotations file.
        insert_annotations("family/ontology.owl",
                           annotations, out_file="family/ontology_extended.owl")

        # Check if the ontology has been extended
        new_dataset = PathDataset("family/ontology_extended.owl")
        projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                   relations=["http://should_have"])
        edges = projector.project(new_dataset.ontology)
        edges = set([edge.astuple() for edge in edges])

        true_edges = set([("http://Person", "http://should_have", "http://Life"),
                          ("http://Child", "http://should_have", "http://Father"),
                          ("http://Child", "http://should_have", "http://Mother"),
                          ])

        self.assertEqual(edges, true_edges)

    def test_correctness_method_insert_annotations_undirected(self):
        """This should test the correctness of the method insert_annotations with undirected \
edges"""
        _ = FamilyDataset()

        root = "tests/ontology/"
        annotation_data_1 = (root + "fixtures/family.tsv", "http://should_have", False)
        annotations = [annotation_data_1]  # There  could be more than 1 annotations file.
        insert_annotations("family/ontology.owl",
                           annotations, out_file="family/ontology_extended.owl")

        # Check if the ontology has been extended
        new_dataset = PathDataset("family/ontology_extended.owl")
        projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                   relations=["http://should_have"])
        edges = projector.project(new_dataset.ontology)
        edges = set([edge.astuple() for edge in edges])

        true_edges = set([("http://Person", "http://should_have", "http://Life"),
                          ("http://Life", "http://should_have", "http://Person"),
                          ("http://Child", "http://should_have", "http://Father"),
                          ("http://Child", "http://should_have", "http://Mother"),
                          ("http://Father", "http://should_have", "http://Child"),
                          ("http://Mother", "http://should_have", "http://Child"),
                          ])

        self.assertEqual(edges, true_edges)

    def test_correctness_method_insert_annotations_many(self):
        """This should test the correctness of the method insert_annotations with many \
annotation files"""
        _ = FamilyDataset()

        root = "tests/ontology/"
        annotation_data_1 = (root + "fixtures/family.tsv", "http://should_have", True)
        annotation_data_2 = (root + "fixtures/family2.tsv", "http://should_be", True)
        annotations = [annotation_data_1, annotation_data_2]
        insert_annotations("family/ontology.owl",
                           annotations, out_file="family/ontology_extended.owl")

        # Check if the ontology has been extended
        new_dataset = PathDataset("family/ontology_extended.owl")
        projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                   relations=["http://should_have",
                                                              "http://should_be"])
        edges = projector.project(new_dataset.ontology)
        edges = set([edge.astuple() for edge in edges])

        true_edges = set([("http://Person", "http://should_have", "http://Life"),
                          ("http://Child", "http://should_have", "http://Father"),
                          ("http://Child", "http://should_have", "http://Mother"),
                          ("http://Person", "http://should_be", "http://Polite"),
                          ])

        self.assertEqual(edges, true_edges)
