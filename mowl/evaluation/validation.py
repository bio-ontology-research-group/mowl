"""Validation of a module on held-out axioms."""
import torch as th


def mean_module_cost(module, data, gci_name):
    """The mean loss score of ``module`` on the axioms ``data`` of one normal form, as a Python
    float, in evaluation mode and without gradients. A validation epoch compares it with the
    best one so far to decide whether to keep the state."""
    with th.no_grad():
        module.eval()
        return th.mean(module(data, gci_name)).detach().item()
