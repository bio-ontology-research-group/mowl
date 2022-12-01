import os

from mowl.ontology.create import create_from_triples
from unittest import TestCase


class TestCreate(TestCase):

    @classmethod
    def setUpClass(self):
        root = "tests/ontology/"
        self._1_columns = root + "fixtures/1_columns.tsv"
        self._2_columns = root + "fixtures/2_columns.tsv"
        self._2_columns_owl = root + "fixtures/2_columns.owl"
        self._3_columns = root + "fixtures/3_columns.tsv"
        self._3_columns_owl = root + "fixtures/3_columns.owl"
        self._4_columns = root + "fixtures/4_columns.tsv"

    @classmethod
    def tearDownClass(self):
        if os.path.exists(self._2_columns_owl):
            os.remove(self._2_columns_owl)
        if os.path.exists(self._3_columns_owl):
            os.remove(self._3_columns_owl)

    def test_attribute_type_checking_create_from_triples(self):
        """It should test the type checking for the method create_from_triples"""

        self.assertRaisesRegex(TypeError, "Parameter triples_file must be of type str",
                               create_from_triples, 1, "out_file")

        self.assertRaisesRegex(TypeError, "Parameter out_file must be of type str",
                               create_from_triples, "triples_file", 1)

        self.assertRaisesRegex(TypeError, "Optional parameter relation_name must be of type str",
                               create_from_triples, "triples_file", "out_file", relation_name=1)

        self.assertRaisesRegex(TypeError, "Optional parameter bidirectional must be of type bool",
                               create_from_triples, "triples_file", "out_file",
                               bidirectional="True")

        self.assertRaisesRegex(TypeError, "Optional parameter head_prefix must be of type str",
                               create_from_triples, "triples_file", "out_file", head_prefix=1)

        self.assertRaisesRegex(TypeError, "Optional parameter tail_prefix must be of type str",
                               create_from_triples, "triples_file", "out_file", tail_prefix=1)

    def test_file_with_2_columns(self):
        """This would test correct behaviour when a file with 2 columns is used"""

        with self.assertRaisesRegex(ValueError, "Found 2 elements in triple but the relation_name \
field is None"):

            print(os.getcwd())
            create_from_triples(self._2_columns, self._2_columns_owl)

    def test_file_with_incorrect_number_of_columns(self):
        """This would make sure that the file has either 2 or 3 columns"""

        with self.assertRaisesRegex(ValueError, "Expected number of elements in triple to be 2 or \
3. Got 1"):
            create_from_triples(self._1_columns, self._2_columns_owl)

        with self.assertRaisesRegex(ValueError, "Expected number of elements in triple to be 2 or \
3. Got 4"):
            create_from_triples(self._4_columns, self._2_columns_owl)
