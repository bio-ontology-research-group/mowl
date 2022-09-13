from mowl.ontology.extend import insert_annotations
from unittest import TestCase


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
