"""Tests of the method and regime catalogue (:mod:`mowl.catalog`).

Most tests only read card data and do not need the JVM. The tests that import
implementation classes (``card.load()``) need it, so the JVM is started at module import,
as in the other test modules.
"""

import importlib
import inspect
import sys

import pytest

import mowl
mowl.init_jvm("2g")

from mowl.catalog import (Family, Guarantee, GuaranteeKind, MethodCard,  # noqa: E402
                          Origin, Reference, RegimeCard, RegimeKind, Status,
                          catalogue_to_rst, get_method, get_regime, list_methods,
                          list_regimes, regimes_to_rst)
from mowl.catalog import registry  # noqa: E402

REF = Reference("test2026", "Test, T. (2026). A test reference.")


def _method(**overrides):
    kwargs = dict(key="testmethod", name="Test method", family=Family.GEOMETRIC,
                  region="box", summary="A test.", logic="EL", concepts="box",
                  roles="vector", references=(REF,),
                  not_implemented_reason="Only for testing.")
    kwargs.update(overrides)
    return MethodCard(**kwargs)


def _regime(**overrides):
    kwargs = dict(key="test-regime", name="Test regime", kind=RegimeKind.TRAINING,
                  summary="A test.", components=("negative sampling",),
                  applies_to="EL models", references=(REF,),
                  not_implemented_reason="Only for testing.")
    kwargs.update(overrides)
    return RegimeCard(**kwargs)


# ---------------------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------------------

def test_registry_loads_and_is_not_empty():
    cards = list_methods()
    assert cards
    assert all(isinstance(c, MethodCard) for c in cards)


def test_method_keys_unique():
    # Discovery itself raises on duplicates; check the result as well.
    discovered = registry._discover()
    keys = [c.key for c in list_methods()]
    assert len(keys) == len(set(keys)) == len(discovered)


def test_get_method_and_unknown_key():
    assert get_method("elbe").key == "elbe"
    with pytest.raises(KeyError):
        get_method("no-such-method")


@pytest.mark.parametrize("card", list_methods(), ids=lambda c: c.key)
def test_every_guarantee_has_source_and_condition(card):
    for g in card.guarantees:
        assert isinstance(g, Guarantee)
        assert g.source.strip(), f"{card.key}: guarantee without source"
        assert g.condition.strip(), f"{card.key}: guarantee without condition"
        assert g.statement.strip(), f"{card.key}: empty guarantee statement"


@pytest.mark.parametrize("card", list_methods(), ids=lambda c: c.key)
def test_card_consistency(card):
    assert card.references
    if card.implemented:
        assert card.implementation
        assert card.origin is not Origin.NONE
    else:
        assert card.not_implemented_reason.strip()
        assert not card.implementation
        with pytest.raises(NotImplementedError):
            card.load()


@pytest.mark.parametrize("card", list_methods(implemented=True), ids=lambda c: c.key)
def test_implementation_paths_resolve(card):
    cls = card.load()
    assert inspect.isclass(cls), f"{card.key}: {card.implementation} is not a class"
    if card.module:
        module_name, _, attr = card.module.rpartition(".")
        assert inspect.isclass(getattr(importlib.import_module(module_name), attr))


def test_every_exported_model_has_a_card():
    import mowl.models as models

    exported = getattr(models, "__all__", None)
    if exported is None:
        exported = [n for n, obj in vars(models).items()
                    if inspect.isclass(obj) and obj.__module__.startswith("mowl.models")]
    implemented = set()
    for card in list_methods(implemented=True):
        implemented.add(card.load())
    missing = []
    for name in exported:
        if name.endswith("PPI") or name.endswith("GDA"):
            continue
        obj = getattr(models, name)
        if inspect.isclass(obj) and obj not in implemented:
            missing.append(name)
    assert not missing, f"Exported models without a catalogue card: {sorted(missing)}"


def test_catalogue_renders_with_every_card_name():
    rst = catalogue_to_rst()
    for card in list_methods():
        assert card.name in rst, card.key
        assert f".. _method-{card.key}:" in rst


def test_list_methods_filters():
    all_cards = list_methods()
    geometric = list_methods(family=Family.GEOMETRIC)
    assert geometric and all(c.family is Family.GEOMETRIC for c in geometric)
    assert list_methods(family="geometric") == geometric

    catalogued = list_methods(status=Status.CATALOGUED)
    assert catalogued and all(c.status is Status.CATALOGUED for c in catalogued)
    assert list_methods(status="catalogued") == catalogued

    implemented = list_methods(implemented=True)
    not_implemented = list_methods(implemented=False)
    assert all(c.implemented for c in implemented)
    assert not any(c.implemented for c in not_implemented)
    assert len(implemented) + len(not_implemented) == len(all_cards)

    alc = list_methods(logic="alc")
    assert alc and all("alc" in c.logic.lower() for c in alc)

    combined = list_methods(family="geometric", implemented=True)
    assert set(combined) == set(geometric) & set(implemented)

    with pytest.raises(ValueError):
        list_methods(family="not-a-family")


def test_list_methods_is_sorted():
    cards = list_methods()
    assert cards == sorted(cards, key=lambda c: (c.family.value, c.region, c.key))


# ---------------------------------------------------------------------------------------
# MethodCard / Guarantee validation
# ---------------------------------------------------------------------------------------

def test_valid_test_card():
    card = _method()
    assert not card.implemented


def test_guarantee_without_source_rejected():
    with pytest.raises(ValueError):
        Guarantee(GuaranteeKind.SOUNDNESS, "zero loss gives a model", source="  ")


def test_catalogued_card_without_reason_rejected():
    with pytest.raises(ValueError):
        _method(not_implemented_reason="")


def test_catalogued_card_with_implementation_rejected():
    with pytest.raises(ValueError):
        _method(implementation="mowl.models.elbe.model.ELBE")


def test_implemented_card_without_implementation_rejected():
    with pytest.raises(ValueError):
        _method(status=Status.LEARNABLE, origin=Origin.REIMPLEMENTATION,
                not_implemented_reason="")


def test_implemented_card_without_origin_rejected():
    with pytest.raises(ValueError):
        _method(status=Status.LEARNABLE, implementation="mowl.models.elbe.model.ELBE",
                not_implemented_reason="")


def test_card_without_reference_rejected():
    with pytest.raises(ValueError):
        _method(references=())


@pytest.mark.parametrize("key", ["", "Upper", "with space"])
def test_bad_key_rejected(key):
    with pytest.raises(ValueError):
        _method(key=key)


def test_wrong_types_rejected():
    with pytest.raises(TypeError):
        _method(family="geometric")
    with pytest.raises(TypeError):
        _method(status="catalogued")
    with pytest.raises(TypeError):
        _method(guarantees=("not a guarantee",))
    with pytest.raises(TypeError):
        _method(references=("not a reference",))


# ---------------------------------------------------------------------------------------
# Regime registry
# ---------------------------------------------------------------------------------------

@pytest.fixture
def fake_regimes(monkeypatch, tmp_path):
    """Point ``mowl.catalog.regimes.__path__`` at a temporary directory.

    Returns a function that writes the given ``{module_name: source}`` files there.
    Module names must start with ``fake_`` (or ``_`` for private modules); imported fake
    modules are removed from ``sys.modules`` afterwards.
    """
    package = importlib.import_module("mowl.catalog.regimes")
    monkeypatch.setattr(package, "__path__", [str(tmp_path)])

    def write(files):
        for name, text in files.items():
            assert name.startswith(("fake_", "_"))
            (tmp_path / f"{name}.py").write_text(text)

    yield write
    for name in [n for n in sys.modules if n.startswith("mowl.catalog.regimes.fake_")]:
        del sys.modules[name]


_REGIME_SRC = '''
from mowl.catalog import Reference, RegimeCard, RegimeKind
REGIME = RegimeCard(key="{key}", name="Fake", kind=RegimeKind.EVALUATION,
                    summary="s", components=("metrics",), applies_to="any",
                    references=(Reference("x", "X"),), not_implemented_reason="test")
'''


def test_regime_discovery_with_zero_modules(fake_regimes):
    fake_regimes({})
    assert registry._discover_regimes() == {}


def test_regime_discovery_with_modules(fake_regimes):
    fake_regimes({"fake_regime_a": _REGIME_SRC.format(key="fake-a"),
                  "fake_regime_b": _REGIME_SRC.format(key="fake-b"),
                  "_private": "X = 1\n"})
    found = registry._discover_regimes()
    assert sorted(found) == ["fake-a", "fake-b"]


def test_regime_discovery_rejects_duplicates(fake_regimes):
    fake_regimes({"fake_dup_a": _REGIME_SRC.format(key="fake-dup"),
                  "fake_dup_b": _REGIME_SRC.format(key="fake-dup")})
    with pytest.raises(ValueError):
        registry._discover_regimes()


def test_regime_discovery_rejects_module_without_regime(fake_regimes):
    fake_regimes({"fake_empty": "X = 1\n"})
    with pytest.raises(ImportError):
        registry._discover_regimes()


def test_real_regime_registry_loads_with_unique_keys():
    regimes = list_regimes()
    keys = [r.key for r in regimes]
    assert len(keys) == len(set(keys)) == len(registry._discover_regimes())
    assert all(isinstance(r, RegimeCard) for r in regimes)
    for r in regimes:
        assert get_regime(r.key) is r
    with pytest.raises(KeyError):
        get_regime("no-such-regime")


def test_list_regimes_filters():
    regimes = list_regimes()
    training = list_regimes(kind=RegimeKind.TRAINING)
    evaluation = list_regimes(kind="evaluation")
    assert all(r.kind is RegimeKind.TRAINING for r in training)
    assert all(r.kind is RegimeKind.EVALUATION for r in evaluation)
    assert len(training) + len(evaluation) == len(regimes)
    implemented = list_regimes(implemented=True)
    assert all(r.implemented for r in implemented)
    for r in regimes:
        for key in r.introduced_with + r.used_by:
            assert r in list_regimes(method=key)


def test_regime_card_validation():
    assert not _regime().implemented
    with pytest.raises(ValueError):
        _regime(references=())
    with pytest.raises(ValueError):
        _regime(not_implemented_reason="")
    with pytest.raises(ValueError):
        _regime(key="Bad Key")
    with pytest.raises(TypeError):
        _regime(kind="training")
    with pytest.raises(TypeError):
        _regime(properties=("not a guarantee",))
    with pytest.raises(NotImplementedError):
        _regime().load()


@pytest.mark.parametrize("regime", list_regimes(implemented=True), ids=lambda r: r.key)
def test_regime_implementation_paths_resolve(regime):
    obj = regime.load()
    assert inspect.isclass(obj) or callable(obj), regime.implementation


def test_method_regime_references_exist():
    known = {r.key for r in list_regimes()}
    unknown = []
    for card in list_methods():
        for field_name in ("training", "evaluation"):
            for key in getattr(card, field_name, ()):
                if key not in known:
                    unknown.append(f"{card.key}.{field_name} -> {key}")
    assert not unknown, "Unknown regime keys: " + "; ".join(unknown)


def test_method_regime_kinds_match():
    wrong = []
    for card in list_methods():
        for key in getattr(card, "training", ()):
            if key in {r.key for r in list_regimes()} and \
                    get_regime(key).kind is not RegimeKind.TRAINING:
                wrong.append(f"{card.key}.training -> {key}")
        for key in getattr(card, "evaluation", ()):
            if key in {r.key for r in list_regimes()} and \
                    get_regime(key).kind is not RegimeKind.EVALUATION:
                wrong.append(f"{card.key}.evaluation -> {key}")
    assert not wrong, "Regime of the wrong kind: " + "; ".join(wrong)


def test_regime_method_references_exist():
    known = {c.key for c in list_methods()}
    unknown = []
    for r in list_regimes():
        for field_name in ("introduced_with", "used_by"):
            for key in getattr(r, field_name):
                if key not in known:
                    unknown.append(f"{r.key}.{field_name} -> {key}")
    assert not unknown, "Unknown method keys in regimes: " + "; ".join(unknown)


def test_regimes_render():
    rst = regimes_to_rst()
    assert "Training and evaluation regimes" in rst
    for r in list_regimes():
        assert r.name in rst, r.key
