"""The validation helpers on a module that is not one of the methods they came from."""
import torch as th

from mowl.evaluation.validation import reached_parameters, summed_mean_cost


class _Toy(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = th.nn.Parameter(th.ones(3))
        self.unused = th.nn.Parameter(th.ones(3))

    def forward(self, form, rows):
        return (self.used[rows] * (2.0 if form == "b" else 1.0)).unsqueeze(-1)


def test_summed_mean_cost_adds_the_mean_of_every_form():
    rows = th.tensor([0, 1])
    assert summed_mean_cost(_Toy(), {"a": rows, "b": rows}) == 3.0


def test_reached_parameters_names_what_the_loss_reads_and_leaves_no_trace():
    toy = _Toy()
    th.manual_seed(3)
    before = th.random.get_rng_state().clone()
    called = []
    reached = reached_parameters(toy, lambda: (toy.used * th.rand(3)).sum(),
                                 after=lambda: called.append(1))
    assert reached == {id(toy.used)}
    assert toy.used.grad is None and called == [1]
    assert th.equal(th.random.get_rng_state(), before)
    assert reached_parameters(toy, lambda: None) == set()
