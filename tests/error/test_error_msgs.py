from unittest import TestCase
import mowl
mowl.init_jvm("10g")

import mowl.error as err


class TestErrorMessages(TestCase):

    def test_owlapi_direct_access_suggestion(self):
        """This message should suggest to use Java imports for direct access to OWLAPI"""
        text = "For direct access to OWLAPI use Java imports."
        self.assertEqual(err.OWLAPI_DIRECT, text)
