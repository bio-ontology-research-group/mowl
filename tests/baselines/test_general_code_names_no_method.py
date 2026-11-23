"""No function of the general packages branches on which method it is serving.

``test_the_method_users_get_is_built_from_parts`` (``own_code.py``) checks that the named
classes do not compute. This is the other half, checked separately and not part of that rule:
the general packages (``mowl/representation``, ``mowl/training``, ``mowl/evaluation``) must
not contain per-method code paths either. A function there must not

* compare a value with the key of a method (``key == "elbe"``, ``name in ("elem", "boxel")``,
  ``.startswith("elbe")``), the name of a named method class (``"EmELPPModule"``) or put one in
  a dictionary lookup that decides between behaviours;
* test a value against a concrete named-method class (``isinstance(x, ELBEModule)``), or by
  the name of its class (``type(x).__name__ == ...``, ``cls.__name__ in TABLE``,
  ``any(c.__name__ == ... for c in type(x).__mro__)``).

A part that behaves differently for two methods takes an option and the named combination sets
it. The one place whose job is to name combinations is the registry of methods: the functions
decorated with ``register_method`` (and ``register_dataset_inputs``, which names the inputs a
method's builder takes) build the combination a key stands for, so they and only they may
mention keys (``ALLOWED``). The module-level tables that *are* registries are listed there too.
"""
import ast
import os

import pytest

import mowl
from mowl.representation import METHODS, REPRESENTATIONS

ROOT = os.path.join(os.path.dirname(mowl.__file__))
SCANNED = ("representation", "training", "evaluation")
PER_METHOD = ("nn", "models")
#: Keys of the registry that are not the key of one method.
NOT_METHOD_KEYS = {"composed", "composed-alc", "composed_graph_kge", "composed_sequences",
                   "composed_subspace"}
#: Decorators whose functions name combinations.
BUILDER_DECORATORS = {"register_method", "register_dataset_inputs"}
#: ``(file relative to mowl/, function)``: functions that are allowed to name a method, each
#: with the reason. Nothing else.
ALLOWED = {}


def _files(*packages):
    for package in packages:
        for dirpath, _, names in os.walk(os.path.join(ROOT, package)):
            if "baselines" in dirpath.split(os.sep):
                continue
            for name in names:
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)


def method_keys():
    return (set(METHODS) | set(REPRESENTATIONS)) - NOT_METHOD_KEYS


def named_method_classes():
    """Classes defined in the per-method packages (``mowl/nn``, ``mowl/models``) other than the
    general models, by name; a name that a general package also defines is left out."""
    general = set()
    for path in _files(*SCANNED):
        general |= {n.name for n in ast.walk(ast.parse(open(path).read()))
                    if isinstance(n, ast.ClassDef)}
    names = set()
    for path in _files(*PER_METHOD):
        rel = os.path.relpath(path, ROOT)
        if rel.startswith(os.path.join("models", "composed")) \
                or rel.startswith(os.path.join("models", "constructed")):
            continue
        names |= {n.name for n in ast.walk(ast.parse(open(path).read()))
                  if isinstance(n, ast.ClassDef) and (n.name.endswith(("Module", "Model"))
                                                      or n.name[0].isupper())}
    return {n for n in names - general if len(n) > 3}


def _decorated_as_builder(fn):
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in BUILDER_DECORATORS:
            return True
    return False


def _strings(node):
    """The string constants of an expression (a constant, a tuple, list or set of them)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [s for e in node.elts for s in _strings(e)]
    return []


def _is_name_of_class(node):
    """``X.__name__`` or ``type(x).__name__`` / ``x.__class__.__name__``."""
    return isinstance(node, ast.Attribute) and node.attr == "__name__"


def _is_table(node):
    """A module-level constant (``CENTRE_SCORERS``, ``module.TABLE``): ALL_CAPS."""
    label = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
    return label.isupper() and len(label) > 2


def findings(path, keys, classes):
    tree = ast.parse(open(path).read())
    rel = os.path.relpath(path, ROOT)
    out = []

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _decorated_as_builder(child):
                    visit(child, child.name)
                continue
            if isinstance(child, ast.Compare):
                operands = [child.left, *child.comparators]
                strings = [s for o in operands for s in _strings(o)]
                hit = [s for s in strings if s in keys or s in classes]
                if hit:
                    out.append((rel, function, child.lineno, f"compared with {sorted(set(hit))}"))
                if any(_is_name_of_class(o) for o in operands) and any(
                        _strings(o) or _is_table(o) for o in operands):
                    # (a class name against another class name, to tell two classes apart in
                    # a message, is not a branch on a method)
                    out.append((rel, function, child.lineno, "compares a class name"))
            elif isinstance(child, ast.Subscript) and _is_name_of_class(child.slice):
                out.append((rel, function, child.lineno, "looks up a table by a class name"))
            elif isinstance(child, ast.Call):
                f = child.func
                if isinstance(f, ast.Attribute) and f.attr == "get" and child.args \
                        and _is_name_of_class(child.args[0]):
                    out.append((rel, function, child.lineno, "looks up a table by a class name"))
                if isinstance(f, ast.Name) and f.id == "isinstance" and len(child.args) == 2:
                    cls = child.args[1]
                    names = [e for e in (cls.elts if isinstance(cls, ast.Tuple) else [cls])]
                    for e in names:
                        label = e.attr if isinstance(e, ast.Attribute) else getattr(e, "id", "")
                        if label in classes:
                            out.append((rel, function, child.lineno,
                                        f"isinstance of the named-method class {label}"))
                elif isinstance(f, ast.Attribute) and f.attr in ("startswith", "endswith"):
                    hit = [s for a in child.args for s in _strings(a) if s in keys]
                    if hit:
                        out.append((rel, function, child.lineno,
                                    f"{f.attr} of the key {hit}"))
            visit(child, function)
    visit(tree, "<module>")
    return out


def test_no_function_of_the_general_packages_branches_on_a_method():
    keys, classes = method_keys(), named_method_classes()
    found = []
    for path in _files(*SCANNED):
        for item in findings(path, keys, classes):
            if (item[0], item[1]) not in ALLOWED:
                found.append(item)
    assert not found, "\n" + "\n".join(f"{a}:{c} in {b}: {d}" for a, b, c, d in found)


def test_the_scanner_finds_what_it_is_meant_to_find(tmp_path):
    source = (
        "def a(key):\n    return key == 'elbe'\n"
        "def b(x):\n    return isinstance(x, ELBEModule)\n"
        "def c(x):\n    return type(x).__name__ == 'Anything'\n"
        "def d(x):\n    return x in ('elem', 'boxel')\n"
        "def e(x):\n    return isinstance(x, Region)\n"
        "def f(x):\n    return x == 'box'\n"
        "def g(x):\n    return TABLE[type(x).__name__]\n"
        "def h(x, y):\n    return type(x).__name__ == type(y).__name__\n")
    path = tmp_path / "sample.py"
    path.write_text(source)
    global ROOT
    old, ROOT = ROOT, str(tmp_path)
    try:
        lines = sorted(item[2] for item in findings(str(path), {"elbe", "elem"}, {"ELBEModule"}))
    finally:
        ROOT = old
    assert lines == [2, 4, 6, 8, 14]


def test_the_allow_list_has_only_reasons():
    assert all(isinstance(k, tuple) and v for k, v in ALLOWED.items())
