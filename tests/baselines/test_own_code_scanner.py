"""The scanner of ``own_code.py`` itself: what it flags and what it lets through.

The strict test of ``test_equivalence.py`` is only as good as this scanner, so it is tested
on small synthetic code: positives (computation the check must find, wherever it hides) and
negatives (wiring and bookkeeping it must not flag).
"""
import sys
import textwrap

import pytest

from tests.baselines import own_code
from tests.baselines.own_code import computing_functions, own_computation, unit_of


def flagged(source):
    return computing_functions(textwrap.dedent(source))


# -- positives -------------------------------------------------------------------------------
@pytest.mark.parametrize("source", [
    # a call into a tensor namespace
    "def f(x):\n    return th.relu(x)\n",
    "def f(x):\n    return torch.sqrt(x)\n",
    "def f(x):\n    return F.softplus(x)\n",
    "def f(x):\n    return np.exp(x)\n",
    # arithmetic tensor methods
    "def f(x):\n    return x.sum()\n",
    "def f(x):\n    return x.mean(dim=0)\n",
    "def f(x):\n    return x.clamp(min=0)\n",
    "def f(x):\n    return x.norm(dim=-1)\n",
    "def f(x):\n    return x.abs().max()\n",
    "def f(x, y):\n    return x.matmul(y)\n",
    # arithmetic operators on tensors (the function touches tensors)
    "def f(x, y):\n    z = x.unsqueeze(0)\n    return z - y\n",
    "def f(pos, neg):\n    loss = pos.reshape(-1)\n    loss = loss + neg\n    return loss\n",
    "def f(x):\n    y = x.detach()\n    y += 1.0\n    return y\n",
    "def f(a, b):\n    t = th.zeros(3)\n    return a * b + t\n",
    # a schedule formula on Python floats (no tensor in sight) is still computation
    "def on_epoch_start(self, epoch, epochs):\n"
    "    self.tau = self.tau_start * (self.tau_end / self.tau_start) ** (epoch / (epochs - 1))\n",
    "def lr(base, epoch):\n    return base * 0.5\n",
    # a private optimisation loop
    "def f(loss, optimizer):\n    loss.backward()\n    optimizer.step()\n",
    # in a method of a class, and in a function nested in a method
    "class C:\n    def m(self, x):\n        return x.exp()\n",
    "class C:\n    def m(self, x):\n        def inner(y):\n            return th.log(y)\n"
    "        return inner(x)\n",
])
def test_computation_is_found(source):
    assert flagged(source), source


def test_the_name_of_the_function_is_qualified_with_its_class():
    found = flagged("def a(x):\n    return x.sum()\n\nclass C:\n    def b(self, x):\n"
                    "        return th.exp(x)\n")
    assert found == ["C.b", "a"]


# -- negatives -------------------------------------------------------------------------------
@pytest.mark.parametrize("source", [
    # plumbing: building, moving, saving, seeding
    "def f(n, device):\n    return th.zeros(n, device=device)\n",
    "def f(rows):\n    return th.tensor(rows, dtype=th.long)\n",
    "def f(path, state):\n    th.save(state, path)\n",
    "def f(seed):\n    th.manual_seed(seed)\n    return np.random.default_rng(seed)\n",
    # moving and reshaping tensors is wiring
    "def f(x, device):\n    return x.to(device).detach().cpu().view(-1)\n",
    # integer bookkeeping on sizes and indices
    "def f(x):\n    n = x.shape[0]\n    return n + 1\n",
    "def f(x):\n    return x.reshape(len(x) // 2, 2)\n",
    "def f(rows, column):\n    return rows[:, 1 - column]\n",
    "def f(module, i):\n    return module.num_classes + i\n",
    "def f(rows):\n    out = []\n    for c, r, d in rows.tolist():\n        out.append(c + d)\n"
    "    return out\n",
    "def f(x):\n    total = 0\n    for i in range(len(x)):\n        total += i\n    return x.to('cpu')\n",
    "def f(n_components, embed_dim):\n    return n_components * embed_dim\n",
    # strings and containers
    "def f(prefix, name):\n    return prefix + name\n",
    "def f(a, b):\n    return [a] + [b]\n",
    "def f(x):\n    return 'loss: ' + str(x)\n",
    "def f(name):\n    return f'{name}.weight'\n",
    # a bare sign flip
    "def f(cost):\n    return -cost\n",
    # arithmetic in a function that handles no tensors and no floats
    "def f(a, b):\n    return a + b\n",
    # wiring: parts handed to parts
    "def f(self, rows):\n    return self.measure.inside(self.classes(rows), self.classes(rows))\n",
    # optimiser construction is not an optimisation loop
    "def f(params):\n    return th.optim.Adam(params, lr=1e-3)\n",
    "def f(data):\n    return th.utils.data.DataLoader(data)\n",
])
def test_bookkeeping_and_wiring_pass(source):
    assert not flagged(source), (source, flagged(source))


# -- units: module-level functions of the per-method package -----------------------------------
def test_unit_of_names_the_per_method_package():
    assert unit_of("mowl.nn.el.octagon.module") == "mowl.nn.el.octagon"
    assert unit_of("mowl.models.multiboxel.common") == "mowl.models.multiboxel"
    # a module that is not in a package is its own unit
    assert unit_of("mowl.models.taxonomy_base") == "mowl.models.taxonomy_base"
    # general models are looked at one module at a time
    assert unit_of("mowl.models.composed.query") == "mowl.models.composed.query"
    assert unit_of("mowl.base_models.alcmodel") == "mowl.base_models.alcmodel"
    # code outside the per-method places is not grouped
    assert unit_of("mowl.geometric.cones") == "mowl.geometric.cones"


@pytest.fixture
def fake_method(tmp_path, monkeypatch):
    """A package ``fakelib.models.fake`` with a model class whose methods only wire and a
    module-level helper that computes, and a general model that runs its own loop."""
    root = tmp_path / "fakelib"
    for directory in (root, root / "models", root / "models" / "fake"):
        directory.mkdir()
        (directory / "__init__.py").write_text("")
    (root / "models" / "fake" / "model.py").write_text(textwrap.dedent('''
        from fakelib.models.fake.helpers import clean


        class Base:
            pass


        class Fake(Base):
            def score(self, rows):
                return clean(self.measure(rows))
        '''))
    (root / "models" / "fake" / "helpers.py").write_text(textwrap.dedent('''
        import torch as th


        def clean(x):
            return th.relu(x - 1.0).sum()


        def harmless(n):
            return n + 1
        '''))
    (root / "models" / "general.py").write_text(textwrap.dedent('''
        class General:
            def train(self, loss, optimizer):
                loss.backward()
                optimizer.step()

            def wire(self, rows):
                return self.part(rows)
        '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(own_code, "LIBRARY", "fakelib.")
    monkeypatch.setattr(own_code, "PER_METHOD_ROOTS", ("fakelib.nn.", "fakelib.models."))
    monkeypatch.setattr(own_code, "GENERAL_MODELS", ("fakelib.models.general",))
    monkeypatch.setattr(own_code, "_unit_cache", {})
    yield
    for name in [m for m in sys.modules if m.startswith("fakelib")]:
        del sys.modules[name]


def test_a_module_level_function_of_the_method_package_counts(fake_method):
    from fakelib.models.fake.model import Fake
    # the class itself only wires; the helper it calls, in its own package, computes
    assert own_computation(Fake) == ["fakelib.models.fake.helpers:clean"]


def test_a_general_model_may_wire_but_not_run_its_own_loop(fake_method):
    from fakelib.models.general import General
    assert own_computation(General) == ["fakelib.models.general:General.train"]
