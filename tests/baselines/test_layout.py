"""The library is the combinations of parts; direct implementations are only a reference."""
import ast
import os

import pytest

import mowl
from tests.baselines.harness import all_exempt, all_pairs

ROOT = os.path.dirname(mowl.__file__)


def _python_files():
    for folder, _, files in os.walk(ROOT):
        if os.path.join(ROOT, "baselines") in folder + os.sep or "__pycache__" in folder:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(folder, name)


def test_nothing_outside_the_baselines_imports_them():
    offenders = []
    for path in _python_files():
        if os.path.relpath(path, ROOT).startswith("baselines"):
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any(n == "mowl.baselines" or n.startswith("mowl.baselines.") for n in names):
                offenders.append(f"{os.path.relpath(path, ROOT)}:{node.lineno}")
    assert not offenders, offenders


def _catalogue_keys():
    from mowl.catalog.registry import list_methods
    return sorted(card.key for card in list_methods(implemented=True))


@pytest.mark.parametrize("key", _catalogue_keys())
def test_every_implemented_method_is_compared_or_exempt(key):
    compared = {p.method for p in all_pairs()}
    exempt = all_exempt()
    assert key in compared or key in exempt, (
        f"{key}: no comparison in tests/baselines/pairs_*.py and no stated exemption")
    assert not (key in compared and key in exempt), f"{key}: both compared and exempt"


def test_exemptions_give_a_reason():
    for key, reason in all_exempt().items():
        assert isinstance(reason, str) and len(reason) > 30, key


# ----------------------------------------------------------------------------------------
# The baselines own their computation: they do not import it from the library.
# ----------------------------------------------------------------------------------------
BASELINES = os.path.join(ROOT, "baselines")

#: Packages of the library whose code computes (parts, regimes, scores, methods).
LIBRARY_COMPUTING = ("mowl.representation", "mowl.training", "mowl.evaluation", "mowl.nn",
                     "mowl.models", "mowl.geometric", "mowl.projection")

#: Names a baseline may import from those packages because they do not compute: base classes
#: that only dispatch, exception classes, a reader of the axioms into index rows.
ALLOWED_NAMES = {
    ("mowl.evaluation", "el_scoring"): "the evaluation's table of published scores, in "
                                       "which the benchmark plugin registers the direct "
                                       "modules (both implementations are ranked alike)",
    ("mowl.representation", "adapters"): "the table of region readers, in which the "
                                         "benchmark plugin registers the direct modules for "
                                         "the evaluations defined on regions",
    ("mowl.representation", "methods"): "the builders that read the trained tables of a "
                                        "direct module as balls or boxes for those "
                                        "evaluations (after training; no part trains it)",
    ("mowl.evaluation.axiom_ranking", "rank_queries"):
        "the validation measure a regime selects the model by (the evaluation both implementations are measured with)",
    ("mowl.evaluation.el_scoring", "resolve_scorer"):
        "the validation measure a regime selects the model by (the evaluation both implementations are measured with)",
    ("mowl.evaluation.sampled_f1", "validation_best_f1"):
        "the validation measure a regime selects the model by (the evaluation both implementations are measured with)",
    **{("mowl.geometric.normal_forms", name): "normal-form axiom record read by the "
                                              "completion reasoner (data structure)"
       for name in ("NF1", "NF2", "NF3", "NF4", "NF6", "NF7")},
    ("mowl.geometric.completion", "Completion"): "the EL completion reasoner (deductive "
                                                 "closure of the axioms, not an embedding)",
    ("mowl.evaluation.model_scoring", "UnsupportedAxiomForm"): "exception class",
    ("mowl.evaluation.axiom_ranking", "el_rows"): "reads the validation axioms into index "
                                                   "rows (data extraction)",
    ("mowl.projection", "Edge"): "a (head, relation, tail) triple: data structure",
    ("mowl.projection.base", "ProjectionModel"): "base class of the projectors: holds the edges",
    ("mowl.projection", "TaxonomyProjector"): "reads the subclass axioms into edges (data "
                                              "extraction)",
    ("mowl.projection", "OWL2VecStarProjector"): "reads the ontology into edges (data "
                                                  "extraction)",
    ("mowl.projection", "projector_factory"): "builds a projector that reads the ontology "
                                              "into edges (data extraction; the evaluation "
                                              "filter of the original EL model base)",
}


def _baseline_files():
    for folder, _, files in os.walk(BASELINES):
        if "__pycache__" in folder:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(folder, name)


def _library_imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            targets = [(node.module, a.name) for a in node.names]
        elif isinstance(node, ast.Import):
            targets = [(a.name, None) for a in node.names]
        else:
            continue
        for module, name in targets:
            if any(module == p or module.startswith(p + ".") for p in LIBRARY_COMPUTING):
                yield node.lineno, module, name


def test_baselines_do_not_import_library_computation():
    offenders = []
    for path in _baseline_files():
        rel = os.path.relpath(path, BASELINES).replace(os.sep, "/")
        for line, module, name in _library_imports(path):
            if (module, name) not in ALLOWED_NAMES:
                offenders.append(f"{rel}:{line} imports {module}"
                                 + (f".{name}" if name else ""))
    assert not offenders, offenders


def test_the_allow_list_has_no_stale_entries():
    used = set()
    for path in _baseline_files():
        for _, module, name in _library_imports(path):
            used.add((module, name))
    assert set(ALLOWED_NAMES) <= used, sorted(set(ALLOWED_NAMES) - used)
    assert all(ALLOWED_NAMES.values())


def test_the_library_paths_that_users_know_still_import():
    import importlib
    for old in ("mowl.nn.lm.hyperbolic", "mowl.nn.order.manifolds",
                "mowl.nn.graph.kge_interactions", "mowl.nn.el.octagon.geometry",
                "mowl.nn.el.multibox_utils", "mowl.nn.numerics", "mowl.nn.query.base"):
        importlib.import_module(old)


def test_a_baseline_has_its_own_copy_of_every_geometry_it_computes_with():
    from mowl.baselines.nn.lm import hyperbolic as base_hb
    from mowl.nn.lm import hyperbolic as lib_hb
    from mowl.baselines.nn.order import manifolds as base_m
    from mowl.nn.order import manifolds as lib_m
    assert base_hb is not lib_hb and base_hb.distance is not lib_hb.distance
    assert base_m.PoincareBall is not lib_m.PoincareBall
