"""Every optimisation step of mOWL is taken by the shared loop (:mod:`mowl.training.loop`).

Outside the baselines, a call of ``.backward()`` or of an optimiser's ``.step()`` may only
appear in the files of the allow-list below, each with the reason it is there. A model, a
regime or a part that needs a step of its own computes the loss of the step
(:meth:`mowl.training.TrainingRegime.step_loss`) and lets the loop take it.
"""
import ast
import pathlib
import warnings

import pytest
import torch as th

import mowl

ROOT = pathlib.Path(mowl.__file__).parent

#: file (relative to ``mowl/``) -> why it may call ``backward`` / ``step``
ALLOWED = {
    "training/loop.py": "the shared training loop and its chunked accumulation "
                        "(accumulate_backward)",
    "evaluation/model_scoring.py": "inner solver: the pair-classifier head of an evaluation "
                                   "regime, fitted once on frozen embeddings",
    "evaluation/validation.py": "gradient probe (reached_parameters): backward without a "
                                "step, to refuse a part no axiom reads",
    "representation/library/solved_solver.py": "inner solver: GradientFit, the gradient "
                                               "alternative to the convex solver of BoxLitE",
}


def _is_scheduler(node):
    """``x.scheduler.step()`` / ``scheduler.step()``: a learning-rate schedule, not an
    optimisation step."""
    receiver = node.func.value
    text = ast.unparse(receiver)
    return "sched" in text.lower()


def _steps_and_backwards():
    found = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("baselines/"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if name == "backward" or (name == "step" and not _is_scheduler(node)):
                    found.setdefault(rel, []).append(node.lineno)
            # an optimiser's step replaced by a wrapper (a loop in disguise)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "step":
                        found.setdefault(rel, []).append(node.lineno)
    return found


def test_only_the_shared_loop_and_documented_inner_solvers_take_steps():
    found = _steps_and_backwards()
    outside = {f: lines for f, lines in found.items() if f not in ALLOWED}
    assert not outside, (
        f"optimisation steps outside the shared loop: {outside}; give the step to "
        "TrainingRegime.step_loss and train in mowl.training.loop")
    assert "training/loop.py" in found


@pytest.mark.parametrize("name", sorted(f for f in ALLOWED if f != "training/loop.py"))
def test_every_allowed_file_says_why(name):
    text = (ROOT / name).read_text()
    assert "inner solver" in text or "probe" in text, name


def test_the_regime_loops_run_in_the_shared_loop():
    """The canonical-mixture batches, the sampled objective and the triple batches are
    regimes of the shared loop, not loops."""
    from mowl.representation.library.mixture_training import (AssertionBatches,
                                                              IndividualBatches)
    from mowl.training import TrainingRegime
    from mowl.training.sampled_objective import SampledObjectiveTraining
    from mowl.training.triples import TripleBatches
    for cls in (IndividualBatches, AssertionBatches, SampledObjectiveTraining, TripleBatches):
        assert issubclass(cls, TrainingRegime)
        assert "step_loss" in cls.__dict__ or any("step_loss" in b.__dict__
                                                  for b in cls.__mro__[1:-2])


# -- the step hook of the shared loop, on a toy model --------------------------------------
class _Toy:
    """A model of the shared loop with one form ``"x"``: the cost of a row is the squared
    distance of a learned vector to the row."""

    def __init__(self, rows, lr=0.1):
        from mowl.training.loop import RegimeTrained, Rows

        class Model(RegimeTrained):
            pass
        model = Model()
        model.module = th.nn.Module()
        model.module.weight = th.nn.Parameter(th.zeros(2))
        model.module.forward = lambda r, form, neg=False: ((r - model.module.weight) ** 2).sum(1)
        model.device = "cpu"
        model.learning_rate = lr
        model.model_filepath = None
        model._datasets = {"x": Rows(rows)}
        type(model).training_datasets = property(lambda self: self._datasets)
        type(model).get_optimizer = lambda self: th.optim.SGD(self.module.parameters(),
                                                              lr=self.learning_rate)
        self.model = model


def _rows():
    return th.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5], [2.0, 2.0], [-1.0, 4.0]])


def test_micro_batch_accumulation_gives_the_gradient_of_the_whole_step():
    from mowl.training import MicroBatchAccumulation
    whole = _Toy(_rows()).model
    whole.training_regime = None
    whole.train(epochs=3)
    chunked = _Toy(_rows()).model
    chunked.training_regime = MicroBatchAccumulation(chunk_size=2)
    losses = chunked.train(epochs=3)
    assert th.allclose(whole.module.weight, chunked.module.weight, atol=1e-6)
    assert th.allclose(th.tensor(whole.training_losses), th.tensor(losses), atol=1e-5)


def test_a_step_with_nothing_to_supervise_is_not_taken_nor_counted():
    from mowl.training import Accumulated, TrainingRegime
    from mowl.training.loop import accumulate_backward

    class EveryOther(TrainingRegime):
        steps_per_epoch = 2

        def step_loss(self, step):
            if step % 2:
                return None
            m = self.model
            return Accumulated(accumulate_backward(m.module(m._datasets["x"][:], "x").mean()))
    toy = _Toy(_rows()).model
    toy.training_regime = EveryOther()
    losses = toy.train(epochs=2)
    assert len(losses) == 2 and losses[0] > losses[1] > 0   # mean over the taken step only


def test_a_step_loss_with_terms_reaches_on_epoch_end():
    from mowl.training import ComposedRegime, TrainingRegime
    seen = []

    class Terms(TrainingRegime):
        def step_loss(self, step):
            m = self.model
            loss = m.module(m._datasets["x"][:], "x").mean()
            return {"total": loss, "half": loss / 2}

    class Record(TrainingRegime):
        def on_epoch_end(self, epoch, losses):
            seen.append(dict(losses))
    toy = _Toy(_rows()).model
    toy.training_regime = ComposedRegime(Record(), Terms())
    toy.train(epochs=2)
    assert len(seen) == 2 and set(seen[0]) == {"total", "half"}
    assert seen[0]["half"] == pytest.approx(seen[0]["total"] / 2)


def test_idle_component_reseeding_resets_the_optimiser_of_the_run():
    """Defect (round 7): the regime cleared ``model.optimizer``, which a model of the shared
    loop such as MultiBoxEL does not have, so its re-seeding never reset the optimiser. The
    fix (default) reads the optimiser of the run; ``optimizer_from="model attribute"`` is the
    earlier behaviour."""
    from mowl.training import ComposedRegime, IdleComponentReseeding, TrainingRegime

    seen = {}

    class Moments(TrainingRegime):
        def on_epoch_end(self, epoch, losses):
            opt = self.model.training_optimizer
            seen.setdefault("before", []).append(len(opt.state))

    def run(**options):
        seen.clear()
        toy = _Toy(_rows()).model
        type(toy).reseed_components = lambda self, epoch: 1
        toy.get_optimizer = lambda: th.optim.Adam(toy.module.parameters(), lr=0.1)
        reseed = IdleComponentReseeding(every=1, **options)
        toy.training_regime = ComposedRegime(reseed, Moments())
        toy.train(epochs=2)
        return toy, list(seen["before"])
    toy, states = run()
    assert states == [0, 0] and not hasattr(toy, "optimizer")
    _, old = run(optimizer_from="model attribute")
    assert old == [1, 1]          # the state was never cleared
