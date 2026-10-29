from inspect import classify_class_attrs
from unittest import TestCase

from tests.datasetFactory import FamilyDataset
from mowl.datasets import ELDataset
from mowl.owlapi.defaults import BOT, TOP


class TestElDataset(TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset_family = FamilyDataset()

        self.male = "http://Male"
        self.female = "http://Female"
        self.parent = "http://Parent"
        self.person = "http://Person"
        self.mother = "http://Mother"
        self.father = "http://Father"
        self.has_child = "http://hasChild"

    def test_param_types(self):
        """This should check if the parameters of ELDataset are of the correct type"""

        with self.assertRaisesRegex(TypeError, "Parameter ontology must be of type \
org.semanticweb.owlapi.model.OWLOntology."):
            ELDataset("ontology")

        with self.assertRaisesRegex(TypeError, "Optional parameter class_index_dict must be of \
type dict"):
            ELDataset(self.dataset_family.ontology, class_index_dict="class_index_dict")

        with self.assertRaisesRegex(TypeError, "Optional parameter object_property_index_dict \
must be of type dict"):
            ELDataset(self.dataset_family.ontology,
                      object_property_index_dict="object_property_index_dict")

        with self.assertRaisesRegex(TypeError, "Optional parameter extended must be of type bool"):
            ELDataset(self.dataset_family.ontology, extended="extended")

        with self.assertRaisesRegex(TypeError, "Optional parameter device must be of type str"):
            ELDataset(self.dataset_family.ontology, device=1)

        with self.assertRaisesRegex(TypeError, "Optional parameter ontology_path must be of type str"):
            ELDataset(self.dataset_family.ontology, ontology_path=123)

        with self.assertRaisesRegex(TypeError, "Optional parameter use_cache must be of type bool"):
            ELDataset(self.dataset_family.ontology, use_cache="yes")

    def test_extended_parameter_false(self):
        """This should check if the extended parameter works as expected when set to false"""

        family_classes = self.dataset_family.classes.as_str
        family_object_properties = self.dataset_family.object_properties.as_str

        class_index_dict = {class_name: i for i, class_name in enumerate(family_classes)}
        object_property_index_dict = {object_property_name: i for i, object_property_name in
                                      enumerate(family_object_properties)}

        dataset = ELDataset(self.dataset_family.ontology,
                            class_index_dict=class_index_dict,
                            object_property_index_dict=object_property_index_dict,
                            extended=False)

        gcis = dataset.get_gci_datasets()

        # Testing not existance of GCI_BOT datasets
        self.assertNotIn("gci0_bot", gcis)
        self.assertNotIn("gci1_bot", gcis)
        self.assertNotIn("gci3_bot", gcis)

        # Testing correct contents of GCI datasets

        true_gci0 = set()
        true_gci0.add((class_index_dict[self.male], class_index_dict[self.person]))
        true_gci0.add((class_index_dict[self.female], class_index_dict[self.person]))
        true_gci0.add((class_index_dict[self.father], class_index_dict[self.male]))
        true_gci0.add((class_index_dict[self.mother], class_index_dict[self.female]))
        true_gci0.add((class_index_dict[self.mother], class_index_dict[self.parent]))
        true_gci0.add((class_index_dict[self.father], class_index_dict[self.parent]))
        true_gci0.add((class_index_dict[self.parent], class_index_dict[self.person]))

        true_gci1 = set()
        true_gci1.add((class_index_dict[self.female], class_index_dict[self.male],
                      class_index_dict[BOT]))
        true_gci1.add((class_index_dict[self.female], class_index_dict[self.parent],
                      class_index_dict[self.mother]))
        true_gci1.add((class_index_dict[self.male], class_index_dict[self.parent],
                      class_index_dict[self.father]))

        true_gci2 = set()
        true_gci2.add((class_index_dict[self.parent], object_property_index_dict[self.has_child],
                      class_index_dict[TOP]))

        true_gci3 = set()
        true_gci3.add((object_property_index_dict[self.has_child], class_index_dict[self.person],
                      class_index_dict[self.parent]))


class TestElDatasetUnindexedAxioms(TestCase):
    """Axioms mentioning entities missing from given index dictionaries are skipped with a
    single warning instead of raising KeyError."""

    def test_unindexed_auxiliary_classes_are_skipped(self):
        from mowl.owlapi import OWLAPIAdapter
        ad = OWLAPIAdapter()
        ont = ad.create_ontology("http://test.org/unindexed")
        A, B, C, D = [ad.create_class(f"http://test.org/unindexed#{x}") for x in "ABCD"]
        r = ad.create_object_property("http://test.org/unindexed#r")
        ad.owl_manager.addAxiom(ont, ad.create_subclass_of(
            ad.create_object_intersection_of(B, ad.create_object_some_values_from(r, C)), A))
        ad.owl_manager.addAxiom(ont, ad.create_subclass_of(D, B))
        names = [str(c.toStringID()) for c in (A, B, C, D)]
        class_index = {n: i for i, n in enumerate(names)}
        prop_index = {str(r.toStringID()): 0}

        el_dataset = ELDataset(ont, class_index, prop_index)
        with self.assertLogs("mowl.datasets.el.el_dataset", level="WARNING") as log:
            datasets = el_dataset.get_gci_datasets()
        warnings = [rec for rec in log.records if "were skipped" in rec.getMessage()]
        self.assertEqual(len(warnings), 1)
        self.assertIn("2 normalized axioms", warnings[0].getMessage())
        self.assertEqual(el_dataset.num_skipped_axioms, 2)
        self.assertEqual(len(datasets["gci0"]), 1)  # D SubClassOf B is kept

    def test_no_axioms_skipped_without_index_dictionaries(self):
        el_dataset = ELDataset(FamilyDataset().ontology)
        el_dataset.get_gci_datasets()
        self.assertEqual(el_dataset.num_skipped_axioms, 0)
