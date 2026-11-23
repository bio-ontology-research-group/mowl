"""The scanner of ``own_code.py`` itself: what it flags and what it lets through.

The strict test of ``test_equivalence.py`` is only as good as this scanner, so it is tested
on small synthetic code: positives (computation the check must find, wherever it hides) and
negatives (wiring and bookkeeping it must not flag).
"""
import sys
import textwrap

import pytest

from tests.baselines import own_code
from tests.baselines.own_code import (class_int_attributes, computing_functions,
                                      own_computation, unit_of)


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
    # tensor evidence per value: a value from a part or a tensor call, in a function that calls
    # no tensor namespace (the five ``get_regularization_loss`` anchor terms, QueryGeometry.logit)
    "class C:\n    def get_regularization_loss(self):\n"
    "        loss = super().get_regularization_loss()\n"
    "        if self.init_boxes is not None and self.anchor_weight > 0:\n"
    "            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)\n"
    "        return loss\n",
    "class C:\n    def logit(self, state, ids):\n"
    "        return self.gamma - self.distance(state, ids)\n",
    "class C:\n    def conditional_errors(self):\n        out = []\n"
    "        for c in self.conditionals:\n"
    "            p = self.probability(c.conclusion, c.premise)\n"
    "            out.append(max(c.lower - p, p - c.upper, 0.0))\n        return out\n",
    "class C:\n    def f(self, rows):\n        a = self.classes(rows)\n        b = a * 2\n"
    "        return b + 1\n",
    "class C:\n    def f(self, x, y):\n        return self.part(x).norm() - y\n",
    # names are no evidence of an integer: ``offset + pos``, ``d``, ``n`` in a function that
    # handles tensors
    "def f(x, offset, pos):\n    y = x.to('cpu')\n    return y[offset + pos]\n",
    "def f(x, d):\n    y = x.detach()\n    return y[:, 2 * d]\n",
    "def f(self, rows, n):\n    r = self.part(rows)\n    return r[n - 1]\n",
    "def f(self, num_classes, i):\n    x = self.part(i)\n    return x[num_classes + i]\n",
    # an integer name does not make the attribute of another object an integer
    "def f(self, rows, other):\n    x = self.part(rows)\n    return x[other.size + 1]\n",
    # a loop over a tensor yields tensors
    "def f(self, rows):\n    out = 0\n    for r in self.part(rows):\n"
    "        out = out + r\n    return out\n",
    "def f(self, x, a):\n    for u, v in zip(self.part(x), a):\n        w = u + v\n"
    "    return w\n",
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
    "def f(x):\n    total = 0\n    for i in range(len(x)):\n        total += i\n"
    "    return x.to('cpu')\n",
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
    # integer evidence: annotations, integer defaults, range/len/shape/int, never the name
    "def f(self, x, offset: int, pos: int):\n    return self.part(x)[offset + pos]\n",
    "def f(self, rows, seed=0):\n    return self.part(rows, seed + 1)\n",
    "def f(self, x, k=3, tag='a'):\n    return self.part(x, k * 2, tag + 'b')\n",
    "def f(self, rows):\n    out = []\n    for i in range(len(rows)):\n"
    "        out.append(self.part(rows[i + 1:]))\n    return out\n",
    "def f(self, x):\n    n = int(self.size_of(x))\n    return self.part(x, n + 1)\n",
    "def f(self, x):\n    h = x.shape[1] // 2\n    return self.part(x, h * 2)\n",
    "def f(self, x, target_col: int):\n    return x[:, 1 - target_col]\n",
    "def f(self, x):\n    for k, row in enumerate(range(4)):\n        self.part(x, k + 1)\n",
    # container and text on one side, also with a part in the function
    "def f(self, rows):\n    regimes = super().regimes()\n"
    "    return regimes + [self.part(rows)]\n",
    "def f(self, name):\n    self.part(name)\n    return name + '.weight'\n",
    # set and list handling without arithmetic on parts
    "def f(self, edges, closure, a):\n    excluded = set(closure.get(a, ())) | {a}\n"
    "    sibs = {s for s in edges}\n    return self.part(sorted(sibs - excluded))\n",
    "def f(self, sources, data):\n    corpus = []\n    for s in sources:\n"
    "        corpus.extend(s.sentences(data))\n    return corpus\n",
    "def f(self, signature, columns):\n    return self.part(signature, next(columns))\n",
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
            def __init__(self, size: int):
                self.size = size


        class Fake(Base):
            def score(self, rows):
                return clean(self.measure(rows, 2 * self.size))
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


# -- integer attributes of the object ----------------------------------------------------------
def test_an_attribute_assigned_from_integer_evidence_is_an_integer():
    source = """
class C:
    steps = 3

    def __init__(self, dim: int, seed=0, rows=()):
        self.dim = dim
        self.seed = seed
        self.n = len(rows)
        self.weight = float(dim)

    def ints(self, x):
        return self.part(x, 2 * self.dim, self.seed + 1, self.n - 1, self.steps * 2)

    def other(self, x):
        return self.part(x, self.weight * 2)

    def tensor(self, x):
        return self.part(x, self.part2(x) + 1)
"""
    assert flagged(source) == ["C.other", "C.tensor"]


def test_the_integer_attributes_of_the_whole_object_count(fake_method):
    from fakelib.models.fake import model as fake_model
    from fakelib.models.fake.model import Fake
    assert own_computation(Fake) == ["fakelib.models.fake.helpers:clean"]
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(fake_model.Base))
    assert class_int_attributes(tree.body[0]) == {"self.size"}
