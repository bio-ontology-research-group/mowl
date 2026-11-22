"""Validation of a module on held-out axioms."""
import torch as th


def mean_module_cost(module, data, gci_name):
    """The mean loss score of ``module`` on the axioms ``data`` of one normal form, as a Python
    float, in evaluation mode and without gradients. A validation epoch compares it with the
    best one so far to decide whether to keep the state."""
    with th.no_grad():
        module.eval()
        return th.mean(module(data, gci_name)).detach().item()


def summed_mean_cost(module, rows_by_form):
    """The sum, over the axiom forms of ``rows_by_form`` (form name -> rows), of the mean loss
    score of ``module`` on the rows of that form, as a Python float and without gradients:
    how far a module is from satisfying a set of axioms, with every form weighted equally."""
    with th.no_grad():
        return float(sum(module(form, rows).mean() for form, rows in rows_by_form.items()))


def reached_parameters(module, loss, after=None):
    """The ids of the parameters of ``module`` that receive a gradient from ``loss()`` (a
    function returning a scalar tensor or ``None``). Gradients are cleared afterwards and the
    torch random generator is put back as it was, so the probe leaves no trace in a run.
    ``after`` is called before the generator is restored (for further evaluations that must
    not disturb the run either). Used to refuse a replaced part that no axiom reads."""
    state = th.random.get_rng_state()
    try:
        total = loss()
        if total is not None and total.requires_grad:
            total.backward()
        reached = {id(p) for p in module.parameters() if p.grad is not None}
        for p in module.parameters():
            p.grad = None
        if after is not None:
            after()
    finally:
        th.random.set_rng_state(state)
    return reached
