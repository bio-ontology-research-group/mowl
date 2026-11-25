"""Validation of a module on held-out axioms."""
import torch as th


def mean_module_cost(module, data, gci_name):
    """The mean loss score of ``module`` on the axioms ``data`` of one normal form, as a Python
    float, in evaluation mode and without gradients. A validation epoch compares it with the
    best one so far to decide whether to keep the state."""
    with th.no_grad():
        module.eval()
        return th.mean(module(data, gci_name)).detach().item()


def corrupted_pair_error(module, data, gci_name, candidates, num_negatives=10, seed=0,
                         column=None):
    """How often ``module`` fails to score an axiom better than a corruption of it, in
    [0, 1] (lower is better; 0.5 is chance): every row of ``data`` is compared with
    ``num_negatives`` copies whose entity in ``column`` (default: the last) is replaced by
    a random one of ``candidates``; a pair counts 1 when the corruption has the lower loss
    score and 0.5 when the two are equal. The corruptions depend on ``seed`` only, so the
    value is comparable between epochs.

    Unlike the mean loss of the true axioms (:func:`mean_module_cost`), this cannot be
    made small by a module that gives every axiom the same score: one that has not learned
    anything (boxes that all overlap, a loss that is zero everywhere) sits at 0.5.
    """
    candidates = th.as_tensor(candidates, dtype=th.long, device=data.device)
    column = data.shape[1] - 1 if column is None else column
    generator = th.Generator().manual_seed(int(seed))
    was_training = module.training
    module.eval()
    with th.no_grad():
        pos = module(data, gci_name).reshape(-1)
        wrong = th.zeros((), dtype=th.float64, device=pos.device)
        for _ in range(int(num_negatives)):
            picks = th.randint(len(candidates), (len(data),), generator=generator)
            rows = data.clone()
            rows[:, column] = candidates[picks.to(data.device)]
            neg = module(rows, gci_name).reshape(-1)
            wrong += (neg < pos).double().sum() + 0.5 * (neg == pos).double().sum()
    if was_training:
        module.train()
    return float(wrong / max(1, len(data) * int(num_negatives)))


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
