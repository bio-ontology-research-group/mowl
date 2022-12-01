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
