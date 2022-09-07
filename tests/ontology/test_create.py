from unittest import TestCase

import os
import mowl
mowl.init_jvm("10g")
from mowl.ontology.create import create_from_triples

class TestCreate(TestCase):

    def test_attribute_type_checking_create_from_triples(self):
        """It should test the type checking for the method create_from_triples"""

        self.assertRaisesRegex(TypeError, "Parameter triples_file must be of type str", create_from_triples, 1, "out_file")

        self.assertRaisesRegex(TypeError, "Parameter out_file must be of type str", create_from_triples, "triples_file", 1)

        self.assertRaisesRegex(TypeError, "Optional parameter relation_name must be of type str", create_from_triples, "triples_file", "out_file", relation_name = 1)

        self.assertRaisesRegex(TypeError, "Optional parameter bidirectional must be of type bool", create_from_triples, "triples_file", "out_file", bidirectional = "True")

        self.assertRaisesRegex(TypeError, "Optional parameter head_prefix must be of type str", create_from_triples, "triples_file", "out_file", head_prefix = 1)

        self.assertRaisesRegex(TypeError, "Optional parameter tail_prefix must be of type str", create_from_triples, "triples_file", "out_file", tail_prefix = 1)
