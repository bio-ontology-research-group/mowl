"""Code written for the released mOWL 2.1.0 keeps working, or breaks only where the CHANGELOG
says so ("Upgrading from 2.1.0").

The reference is the released code itself:

* ``released/api_v2_1_0.json``: the public surface of v2.1.0 (every module, public class and
  function, constructor and method signature with the kind of every parameter), dumped from
  ``git archive v2.1.0`` by importing it;
* ``released/v2_1_0_el_modules.pt``: state dicts of ``ELEmModule``, ``ELBEModule``,
  ``BoxELModule`` and ``BoxSquaredELModule`` made by the v2.1.0 code, with every per-axiom
  loss on fixed rows and its shape (``make_v2_1_0_fixtures.py`` makes it).

The names that came back as deprecated wrappers are checked against the direct
implementations of ``mowl.baselines`` (the earlier code) and against the v2.1.0 numbers.
"""
import ast
import importlib
import inspect
import json
import os
import warnings

import pytest
import torch as th

DATA = os.path.join(os.path.dirname(__file__), "released")
POS = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")

#: Names of 2.1.0 that are gone on purpose (listed in CHANGELOG, "Upgrading from 2.1.0").
REMOVED = {
    # tensor-level helpers of the loss modules: the formulas are measures of the parts now
    "mowl.nn.el.elem.losses": {"ball_inclusion_score", "gci2_score", "gci2_score_neg"},
    "mowl.nn.el.elbe.losses": {"box_inclusion_score", "gci2_score", "gci2_score_neg"},
    "mowl.nn.el.boxel.losses": {"Box", "inclusion_loss", "intersection", "volumes"},
    "mowl.nn.el.boxsquaredel.losses": {
        "box_distance", "box_intersection", "inclusion_score", "gci0_score", "gci0_bot_score",
        "gci1_score", "gci1_bot_score", "gci2_score", "gci2_score_neg", "gci3_score",
        "gci3_bot_score"},
}

#: Signatures that a static comparison flags but that accept every call of 2.1.0.
ACCEPTED = {
    # 2.1.0 took (*args, **kwargs) and required the dataset through the base class
    "SyntacticPlusW2VModel": "new required parameter 'dataset'",
    # forward(*args, stage=...) takes the released forward(axiom, x, e_emb, stage) too
    # (test_falcon_forward_takes_the_stage_by_position)
    "FALCONModule.forward": "'stage' was positional",
}


def _api():
    with open(os.path.join(DATA, "api_v2_1_0.json")) as f:
        return json.load(f)


def _sig(obj):
    try:
        s = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [[p.name, p.kind.name, p.default is not inspect.Parameter.empty]
            for p in s.parameters.values()]


def _breaks(old, new):
    """What a caller of the signature ``old`` finds broken in ``new``."""
    if old is None or new is None:
        return []
    out = []
    names = {p[0]: p for p in new}
    takes_kwargs = any(p[1] == "VAR_KEYWORD" for p in new)
    old_pos = [p[0] for p in old if p[1] in POS]
    new_pos = [p[0] for p in new if p[1] in POS]
    for name, kind, _ in old:
        if kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue
        if name not in names:
            if not takes_kwargs:
                out.append(f"parameter {name!r} removed")
            continue
        if kind in POS:
            i = old_pos.index(name)
            if names[name][1] == "KEYWORD_ONLY":
                out.append(f"{name!r} was positional #{i}, now keyword-only")
            elif i >= len(new_pos) or new_pos[i] != name:
                out.append(f"positional #{i} was {name!r}")
    known = {p[0] for p in old}
    for name, kind, default in new:
        if name not in known and not default and kind not in ("VAR_POSITIONAL",
                                                              "VAR_KEYWORD"):
            out.append(f"new required parameter {name!r}")
    return out


def _accepted(qualified, problem):
    return any(qualified.endswith(k) and problem.startswith(v) for k, v in ACCEPTED.items())


# -- 1. the public surface ---------------------------------------------------------------------

def test_every_public_name_of_2_1_0_is_still_importable():
    missing = []
    for module, names in sorted(_api().items()):
        m = importlib.import_module(module)
        for name in names:
            if name in REMOVED.get(module, ()):
                assert not hasattr(m, name), f"{module}.{name} is back: update REMOVED"
                continue
            if not hasattr(m, name):
                missing.append(f"{module}.{name}")
    assert not missing, missing


def test_every_call_of_2_1_0_binds_as_it_did():
    """Constructor, function and method signatures: no released parameter is gone, none is
    at another position (a positional call would bind it to something else), nothing new is
    required. Every public method of a released class is still there."""
    problems = []
    for module, names in sorted(_api().items()):
        m = importlib.import_module(module)
        for name, d in names.items():
            obj = getattr(m, name, None)
            if obj is None:
                continue
            q = f"{module}.{name}"
            if d["kind"] == "function":
                problems += [f"{q}(): {b}" for b in _breaks(d["sig"], _sig(obj))]
                continue
            problems += [f"{q}: {b}" for b in _breaks(d["init"], _sig(obj.__init__))
                         if not _accepted(name, b)]
            for method, old in d["methods"].items():
                if not hasattr(obj, method):
                    problems.append(f"{q}.{method} is gone")
                    continue
                raw = inspect.getattr_static(obj, method)
                new = _sig(getattr(obj, method))
                if inspect.isfunction(raw) and new:
                    new = new[1:]
                problems += [f"{q}.{method}: {b}" for b in _breaks(old, new)
                             if not _accepted(f"{name}.{method}", b)]
            for prop in d["properties"]:
                if not hasattr(obj, prop):
                    problems.append(f"{q}.{prop} (property) is gone")
    assert not problems, "\n".join(problems)


def test_falcon_model_takes_the_released_positional_order(family_dataset):
    from mowl.models import FALCONModel
    model = FALCONModel(family_dataset, 6, 2, 0.3, 0.3, "product", "max", 2, 0.01, 16)
    assert model.learning_rate == 0.01 and model.batch_size == 16
    assert (model.role_pairs, model.thing, model.nothing) == ("all", "learned", "empty")


def test_per_axiom_losses_take_the_released_argument_names():
    import mowl.nn as N
    rows = th.tensor([[0, 1], [2, 3]])
    for cls in (N.ELEmModule, N.ELBEModule, N.BoxELModule, N.BoxSquaredELModule):
        module = cls(4, 2, 2, embed_dim=3)
        by_name = module.gci0_loss(data=rows)
        assert th.equal(by_name, module.gci0_loss(rows))
    module = N.ELEmModule(4, 2, 2, embed_dim=3)
    crc = th.tensor([[0, 1, 2]])
    # ELEmModule.gci2_loss(data, neg, idxs_for_negs): the third argument is accepted, unused
    assert th.equal(module.gci2_loss(crc, False, None), module.gci2_loss(crc))


def test_falcon_forward_takes_the_stage_by_position():
    from mowl.nn.alc.falcon.module import FALCONModule
    from mowl.owlapi import OWLAPIAdapter
    heads = {(1, 0): [0], (0, 0): [2]}
    tails = {(0, 0): [1], (2, 0): [0]}
    module = FALCONModule(2, 3, 1, heads, tails, embed_dim=4, num_negs=2)
    adapter = OWLAPIAdapter()
    a, b = adapter.create_individual("http://a"), adapter.create_individual("http://b")
    r = adapter.create_object_property("http://r")
    axiom = adapter.data_factory.getOWLObjectPropertyAssertionAxiom(r, a, b)
    x = th.tensor([[0, 0, 1]])
    e_emb = module.e_embedding.weight
    th.manual_seed(0)
    by_keyword = module(axiom, x, e_emb, stage="test")
    th.manual_seed(0)
    by_position = module(axiom, x, e_emb, "test")
    assert th.equal(by_keyword, by_position)


# -- 2. names that came back as deprecated wrappers over the parts ----------------------------

WRAPPER_FILES = ["mowl/nn/el/released_losses.py", "mowl/nn/el/elem/losses.py",
                 "mowl/nn/el/elbe/losses.py", "mowl/nn/el/boxel/losses.py",
                 "mowl/nn/el/boxsquaredel/losses.py", "mowl/nn/alc/falcon/module.py",
                 "mowl/nn/el/boxel/module.py", "mowl/nn/el/boxsquaredel/module.py",
                 "mowl/models/composed/sequence.py"]


def test_the_wrappers_do_not_use_the_baselines():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for path in WRAPPER_FILES:
        with open(os.path.join(root, path)) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module or ""] if isinstance(node, ast.ImportFrom) else \
                    [a.name for a in node.names]
                assert not any("baselines" in n for n in names), (path, names)


@pytest.fixture(scope="module")
def released():
    return th.load(os.path.join(DATA, "v2_1_0_el_modules.pt"), weights_only=False)


#: argument name of the released loss functions -> key of the released state dict
TABLES = {
    "elem": ("ELEmModule", {"class_embed": "class_embed", "class_rad": "class_rad",
                            "rel_embed": "rel_embed", "ind_embed": "ind_embed",
                            "ind_rad": "ind_rad"}),
    "elbe": ("ELBEModule", {"class_embed": "class_embed", "class_offset": "class_offset",
                            "rel_embed": "rel_embed", "ind_embed": "ind_embed",
                            "ind_offset": "ind_offset", "ind_off_set": "ind_offset"}),
    "boxel": ("BoxELModule", {"min_embed": "min_embedding", "delta_embed": "delta_embedding",
                              "relation_embed": "relation_embedding",
                              "scaling_embed": "scaling_embedding",
                              "ind_embed": "ind_embedding",
                              "ind_delta_embed": "ind_delta_embedding"}),
    "boxsquaredel": ("BoxSquaredELModule", {
        "class_center": "class_center", "class_offset": "class_offset",
        "head_center": "head_center", "head_offset": "head_offset",
        "tail_center": "tail_center", "tail_offset": "tail_offset", "bump": "bump_classes",
        "ind_center": "ind_center", "ind_offset": "ind_offset",
        "bump_individuals": "bump_individuals"}),
}
SCALARS = {"margin": 0.1, "temperature": 1.0, "gamma": 0, "delta": 2, "reg_factor": 0.05}
#: The options of the direct implementations that give the released numbers.
DIRECT_OPTIONS = {"radius_term": False, "positive_scale": False,
                  "role_chain_composition": "as_released"}
#: Released numbers that are a defect, fixed (CHANGELOG): ELBE's gci1 loss broadcast a column
#: against a flat vector to a (B, B) matrix; BoxEL's gci3 box took its upper corner as a
#: log-width (``gci3_preimage="as_released"`` restores it for the module and the model, the
#: deprecated function keeps the released numbers).
FIXED = {("elbe", "gci1")}


def _tables(state, mapping, names):
    out = {}
    for name in names:
        key = mapping.get(name)
        if key is None:
            continue
        weight = state.get(key + ".weight")
        out[name] = None if weight is None else th.nn.Embedding.from_pretrained(
            weight.clone(), freeze=False)
    return out


def _call(function, state, mapping, data, neg):
    parameters = inspect.signature(function).parameters
    kwargs = _tables(state, mapping, parameters)
    kwargs.update({k: v for k, v in SCALARS.items() if k in parameters})
    if "data" in parameters:
        kwargs["data"] = data
    if "neg" in parameters:
        kwargs["neg"] = neg
    return kwargs


@pytest.mark.parametrize("method", sorted(TABLES))
def test_the_released_loss_functions_give_the_released_numbers(method, released):
    module_name, mapping = TABLES[method]
    entry = released[module_name]
    losses = importlib.import_module(f"mowl.nn.el.{method}.losses")
    direct = importlib.import_module(f"mowl.baselines.nn.el.{method}.losses")
    compared = 0
    for key, value in entry["losses"].items():
        form, neg = key.split("/")[0], key.endswith("/neg")
        function = getattr(losses, f"{form}_loss")
        named = function
        if all(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in inspect.signature(function).parameters.values()):
            # gci0_bot_loss(*args, **kwargs) of BoxEL is gci0_loss (and gci3_bot gci3)
            named = getattr(losses, form.replace("_bot", "") + "_loss")
        kwargs = _call(named, entry["state_dict"], mapping, value["rows"], neg)
        if "neg" not in inspect.signature(named).parameters and neg:
            continue                         # (the released function had no ``neg``)
        with pytest.warns(DeprecationWarning):
            out = function(**kwargs)
        if (method, form) in FIXED:
            assert out.shape == (len(value["rows"]), 1) and value["shape"] != out.shape
        else:
            assert out.shape == value["value"].shape, (key, out.shape, value["shape"])
            assert th.allclose(out, value["value"], atol=1e-6), key
        # the direct implementation, with the options that give the released numbers
        reference = getattr(direct, named.__name__, None)
        if reference is not None and not (method == "boxel" and form.startswith("gci3")):
            accepted = inspect.signature(reference).parameters
            extra = {k: v for k, v in DIRECT_OPTIONS.items() if k in accepted}
            options = {k: v for k, v in kwargs.items() if k in accepted}
            with th.no_grad():
                expected = reference(**options, **extra)
            assert th.allclose(out.reshape(expected.shape), expected, atol=1e-6), key
        compared += 1
    assert compared >= 9
    regularization = getattr(losses, "regularization_loss", None) or \
        getattr(losses, "reg_loss", None)
    if entry["regularization"] is not None and regularization is not None:
        state = entry["state_dict"]
        with pytest.warns(DeprecationWarning):
            if method == "boxsquaredel":
                tables = _tables(state, mapping, ("bump", "bump_individuals"))
                term = regularization(tables["bump"], 0.05) + \
                    regularization(tables["bump_individuals"], 0.05)
            else:
                term = regularization(**_call(regularization, state, mapping, None, False))
        assert th.allclose(th.as_tensor(term), entry["regularization"], atol=1e-6)


def test_the_released_loss_functions_pass_gradients_to_the_tables():
    from mowl.nn.el.elem import losses
    centres, radii = th.nn.Embedding(4, 3), th.nn.Embedding(4, 1)
    with pytest.warns(DeprecationWarning):
        losses.gci0_loss(th.tensor([[0, 1], [2, 3]]), centres, radii, 0.1).sum().backward()
    assert centres.weight.grad is not None and radii.weight.grad is not None


def test_falcon_module_sample_negatives_and_cc_loss():
    from mowl.baselines.nn.alc.falcon.module import FALCONModule as Direct
    from mowl.nn.alc.falcon.module import FALCONModule
    heads = {(0, 0): [1], (1, 0): [0, 2], (2, 0): [3]}
    for measure in ("max", "pmean2"):
        th.manual_seed(0)
        new = FALCONModule(3, 5, 1, heads, heads, embed_dim=4, num_negs=3, max_measure=measure)
        th.manual_seed(0)
        old = Direct(3, 5, 1, heads, heads, embed_dim=4, num_negs=3, max_measure=measure)
        fs = th.rand(4, 6)
        with pytest.warns(DeprecationWarning):
            assert th.allclose(new.get_cc_loss(fs), old.get_cc_loss(fs), atol=0, rtol=0)
        e, r = th.tensor([[0], [1], [2]]), th.tensor([[0], [0], [0]])
        th.manual_seed(1)
        with pytest.warns(DeprecationWarning):
            drawn = new.sample_negatives(e, r, heads)
        th.manual_seed(1)
        assert th.equal(drawn, old.sample_negatives(e, r, heads))
    assert new.residuum == "notCorD" and new.adapter is not None
    import mowl.nn.alc
    assert mowl.nn.alc.FALCONModule is FALCONModule


def test_box_modules_init_tables_as_released():
    from mowl.baselines.nn.el.boxel.module import BoxELModule as DirectBoxEL
    from mowl.baselines.nn.el.boxsquaredel.module import BoxSquaredELModule as DirectBox2EL
    from mowl.nn import BoxELModule, BoxSquaredELModule
    for new, old, method, args in (
            (BoxELModule(3, 1), DirectBoxEL(3, 1), "init_entity_embedding", (5, 4, (0.1, 0.3))),
            (BoxSquaredELModule(3, 1), DirectBox2EL(3, 1), "init_embeddings", (5, 4)),
            (BoxSquaredELModule(3, 1), DirectBox2EL(3, 1), "init_embeddings", (5, 4, 0, 2))):
        th.manual_seed(3)
        with pytest.warns(DeprecationWarning):
            a = getattr(new, method)(*args)
        th.manual_seed(3)
        b = getattr(old, method)(*args)
        assert th.equal(a.weight, b.weight)


def test_evaluation_model_of_the_word2vec_models(family_dataset):
    from gensim.models import Word2Vec

    from mowl.baselines.models.graph_random_walk.random_walk_w2v_model import \
        EvaluationModel as Direct
    from mowl.models.graph_random_walk.random_walk_w2v_model import EvaluationModel
    from mowl.models.syntactic.w2v_model import EvaluationModel as Syntactic
    assert Syntactic is EvaluationModel
    classes = family_dataset.classes.as_str
    sentences = [[c, classes[(i + 1) % len(classes)]] for i, c in enumerate(classes)] * 5
    w2v = Word2Vec(sentences, vector_size=4, min_count=1, seed=0, workers=1)
    with pytest.warns(DeprecationWarning):
        new = EvaluationModel(w2v, family_dataset, 4, "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = Direct(w2v, family_dataset, 4, "cpu")
    pairs = th.cartesian_prod(th.arange(len(classes)), th.arange(len(classes)))
    assert th.allclose(new(pairs), old(pairs).float(), atol=1e-6)
    triples = th.stack([pairs[:, 0], th.zeros(len(pairs), dtype=th.long), pairs[:, 1]], 1)
    assert th.allclose(new(triples), old(triples).float(), atol=1e-6)
    assert th.allclose(new.embeddings.weight, old.embeddings.weight.float())


# -- 3. state dicts of 2.1.0 ------------------------------------------------------------------

#: The options that give the numbers of 2.1.0 (each documented in the CHANGELOG).
RESTORE = {"ELEmModule": dict(gci1_radius_term=False), "ELBEModule": {},
           "BoxELModule": dict(positive_scale=False, role_chain_composition="as_released",
                               gci3_preimage="as_released"),
           "BoxSquaredELModule": {}}


@pytest.mark.parametrize("name", sorted(RESTORE))
def test_state_dicts_of_2_1_0_load_and_give_the_released_losses(name, released):
    import mowl.nn as N
    entry = released[name]
    classes, relations, individuals, dim = released["sizes"]
    # the module with its defaults loads the released state dict (strictly) ...
    plain = getattr(N, name)(classes, relations, individuals, embed_dim=dim)
    plain.load_state_dict(entry["state_dict"])
    # ... and saves every table under its released name too
    assert set(entry["state_dict"]) <= set(plain.state_dict())
    # with the options of 2.1.0, every loss is the released one, in the released shape
    module = getattr(N, name)(classes, relations, individuals, embed_dim=dim, **RESTORE[name])
    module.load_state_dict(entry["state_dict"])
    module.per_axiom_column = False
    module.strict_negatives = False
    method = {"ELEmModule": "elem", "ELBEModule": "elbe", "BoxELModule": "boxel",
              "BoxSquaredELModule": "boxsquaredel"}[name]
    for key, value in entry["losses"].items():
        form, neg = key.split("/")[0], key.endswith("/neg")
        with th.no_grad():
            out = module.get_loss_function(form)(value["rows"], neg=neg)
        if (method, form) in FIXED:            # one loss per row now, (B, B) in 2.1.0
            assert out.shape == (len(value["rows"]), 1) and value["shape"] != out.shape
            continue
        assert out.shape == value["value"].shape, (key, out.shape, value["shape"])
        assert th.allclose(out, value["value"], atol=1e-6), key
    if entry["regularization"] is not None:
        with th.no_grad():
            assert th.allclose(th.as_tensor(module.regularization_loss()),
                               entry["regularization"], atol=1e-6)


@pytest.mark.parametrize("name", sorted(RESTORE))
def test_state_dicts_of_the_direct_implementation_load(name):
    """The direct implementations (``mowl.baselines``, the earlier hand-written modules)
    save state dicts under the released names; they load, and the losses agree."""
    import mowl.nn as N
    direct = importlib.import_module(
        {"ELEmModule": "mowl.baselines.nn.el.elem.module",
         "ELBEModule": "mowl.baselines.nn.el.elbe.module",
         "BoxELModule": "mowl.baselines.nn.el.boxel.module",
         "BoxSquaredELModule": "mowl.baselines.nn.el.boxsquaredel.module"}[name])
    th.manual_seed(0)
    old = getattr(direct, name)(6, 2, 3, embed_dim=4)
    new = getattr(N, name)(6, 2, 3, embed_dim=4)
    new.load_state_dict(old.state_dict())
    new.per_axiom_column = False
    rows = th.tensor([[0, 1], [2, 3], [4, 5]])
    with th.no_grad():
        a, b = old.gci0_loss(rows), new.gci0_loss(rows)
    assert th.allclose(a.reshape(-1), b.reshape(-1), atol=1e-6)


# -- 4. behaviour of the released graph model -------------------------------------------------

def _graph_model(dataset, **attributes):
    import pykeen.models

    from mowl.models import GraphPlusPyKEENModel
    from mowl.projection import DL2VecProjector
    model = GraphPlusPyKEENModel(dataset)
    model.set_projector(DL2VecProjector(False))
    model.set_kge_method(pykeen.models.TransE, embedding_dim=4, random_seed=0)
    model.optimizer, model.lr, model.batch_size = th.optim.Adam, 0.01, 8
    for k, v in attributes.items():
        setattr(model, k, v)
    return model


def test_graph_plus_pykeen_draws_its_parameters_again_by_default(family_dataset):
    """As released (PyKEEN's loop): every ``train()`` starts from freshly drawn parameters;
    ``reinitialise = False`` continues from the current ones."""
    model = _graph_model(family_dataset)
    assert model.reinitialise is True and model.batching == "pykeen"
    table = model.kge_method.entity_representations[0]._embeddings.weight
    with th.no_grad():
        table.fill_(0.25)
    model.train(epochs=0)
    assert not th.equal(table, th.full_like(table, 0.25))
    model = _graph_model(family_dataset, reinitialise=False)
    table = model.kge_method.entity_representations[0]._embeddings.weight
    with th.no_grad():
        table.fill_(0.25)
    model.train(epochs=0)
    assert th.equal(table, th.full_like(table, 0.25))


def test_graph_plus_pykeen_refuses_a_regime_that_pykeen_batching_would_ignore(family_dataset):
    from mowl.training import TrainingRegime

    class Loss(TrainingRegime):
        def loss(self, form, positive, negative):
            return (negative - positive).relu().mean()

    model = _graph_model(family_dataset)
    model.training_regime = Loss()
    with pytest.raises(TypeError, match="batching"):
        model.train(epochs=1)
    model.batching = "mowl"
    model.train(epochs=1)


# -- 5. PyKEEN's automatic memory optimisation (sub-batches) ----------------------------------

def test_slcwa_batches_split_a_batch_as_pykeen_does():
    """PyKEEN splits a batch that does not fit on the GPU into sub-batches and accumulates
    their gradients (automatic memory optimisation, applied by the released ``KGEModel`` and
    ``GraphPlusPyKEENModel``); ``SLCWABatches`` does the same. Forced here to sub-batches of
    2 on both sides (on a CPU the probe keeps the whole batch)."""
    from types import SimpleNamespace

    from pykeen.models import TransE
    from pykeen.training import SLCWATrainingLoop

    from mowl.projection import Edge
    from mowl.representation.library.other_graph import PyKEENGeometry
    from mowl.training.triples import SLCWABatches, positive_triples, train_on_triples
    edges = [Edge("a", "r", "b"), Edge("b", "r", "c"), Edge("c", "s", "a"),
             Edge("a", "s", "d"), Edge("d", "r", "c"), Edge("e", "r", "a"),
             Edge("b", "s", "e")]
    factory = Edge.as_pykeen(edges)

    def run(ours):
        th.manual_seed(0)
        kge = TransE(triples_factory=factory, embedding_dim=4, random_seed=0)
        th.manual_seed(1)
        if not ours:
            loop = SLCWATrainingLoop(model=kge, triples_factory=factory,
                                     optimizer=th.optim.Adam(kge.get_grad_params(), lr=0.01),
                                     automatic_memory_optimization=False)
            loop.train(triples_factory=factory, num_epochs=3, batch_size=5, sub_batch_size=2,
                       use_tqdm=False)
            return kge
        view = SimpleNamespace(
            kge_method=kge, triples_factory=factory, class_index_dict=factory.entity_to_id,
            individual_index_dict={}, object_property_index_dict=factory.relation_to_id,
            SUBCLASS_RELATION="http://subclassof", create_inverse_triples=False,
            num_negs_per_pos=None, projector=None)
        batches = SLCWABatches(kge, factory, 5, sub_batch_size=2,
                               automatic_memory_optimization=False)
        part = batches._batch_loss
        batches._batch_loss = lambda batch, start, stop: (
            parts.append((start, stop)), part(batch, start, stop))[1]
        train_on_triples(PyKEENGeometry(view), positive_triples(factory.mapped_triples),
                         factory.num_entities, factory.num_relations, th.optim.Adam, 0.01, 5,
                         3, batches=batches)
        return kge

    parts = []
    a, b = run(False), run(True)
    for x, y in zip(a.parameters(), b.parameters()):
        assert th.allclose(x, y, atol=1e-7, rtol=0)
    # every batch of 5 was split into sub-batches of at most 2, in every epoch
    assert len(parts) >= 9 and all(stop - start <= 2 for start, stop in parts)


# -- 6. walk and sentence models: the defaults differ only where documented ------------------

def test_word2vec_models_differ_from_2_1_0_only_on_a_second_train(family_dataset, tmp_path):
    """With Python's hash (``hashfxn=hash``, as released), the first ``train`` of a new model
    is the released one with and without ``restart``; only a second ``train`` differs: the
    released code (the direct implementation, and ``restart=True``) draws the vectors again,
    the default continues from them."""
    from mowl.baselines.models.syntactic.w2v_model import SyntacticPlusW2VModel as Direct
    from mowl.models import SyntacticPlusW2VModel

    def model(cls, name, **options):
        m = cls(family_dataset, corpus_filepath=str(tmp_path / name))
        m.set_w2v_model(vector_size=4, min_count=1, workers=1, seed=0, hashfxn=hash,
                        **options)
        m.generate_corpus(save=True, with_annotations=False)
        return m

    def same(a, b):
        a, b = a.class_embeddings, b.class_embeddings
        return sorted(a) == sorted(b) and all((a[k] == b[k]).all() for k in a)

    direct = model(Direct, "direct")
    released = model(SyntacticPlusW2VModel, "released", restart=True)
    default = model(SyntacticPlusW2VModel, "default")
    for m in (direct, released, default):
        m.train(epochs=1)
    assert same(direct, released) and same(direct, default)
    for m in (direct, released, default):
        m.train(epochs=1)
    assert same(direct, released)
    assert not same(direct, default)


def test_evaluate_predictions_is_deprecated_and_as_released():
    from mowl.evaluation.predictions import evaluate_predictions
    with pytest.warns(DeprecationWarning):
        metrics = evaluate_predictions({"a", "b"}, {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.3},
                                       [1, 2])
    # The released function ranks the scores after cutting them at the best threshold, so
    # every cut-off beyond the first place falls among ties and depends on how the sort orders
    # them (it differs between processors).  Only the tie-free values are checked.
    assert metrics["auc"] == 1.0 and metrics["prec@1"] == 1.0 and metrics["rec@1"] == 0.5
