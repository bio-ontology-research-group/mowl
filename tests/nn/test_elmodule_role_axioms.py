import mowl
from mowl import init_jvm
init_jvm("1g")

from unittest import TestCase

import torch as th
from mowl.nn import ELModule


class TestElModuleRoleAxioms(TestCase):

    def test_loss_function_dispatch(self):
        module = ELModule()
        self.assertIn("role_inclusion", module.gci_names)
        self.assertIn("role_chain", module.gci_names)
        self.assertEqual(module.get_loss_function("role_inclusion"), module.role_inclusion_loss)
        self.assertEqual(module.get_loss_function("role_chain"), module.role_chain_loss)

    def test_base_module_is_not_role_axiom_capable(self):
        self.assertFalse(ELModule.role_axiom_capable)

    def test_unimplemented_losses_explain_themselves(self):
        module = ELModule()

        for gci_name, data in [("role_inclusion", th.tensor([[0, 1]])),
                               ("role_chain", th.tensor([[0, 1, 2]]))]:
            with self.assertRaises(NotImplementedError) as ctx:
                module.forward(data, gci_name)

            message = str(ctx.exception)
            self.assertIn(gci_name, message)
            self.assertIn("role_axiom_capable", message)
