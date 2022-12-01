from mowl.nn import ELModule
from unittest import TestCase


class TestELModule(TestCase):

    def test_handle_incorrect_gci_names_forward_method(self):
        """This checks if the forward method raises a ValueError if the GCI names are incorrect."""

        module = ELModule()

        with self.assertRaisesRegex(
            ValueError,
            "Parameter gci_name must be one of the following: gci0, gci1, gci2, gci3, gci0_bot, \
gci1_bot, gci3_bot."):
            _ = module([], "gci4")
