"""Which code of a named method computes with tensors itself.

A named method is a combination of parts. Its module class and its model class may wire
parts together, name them, hold options and hand rows to parts, but the arithmetic belongs
to the parts. This reads the source and reports the functions that compute.

What is looked at, for a class ``C`` (the module or the model of a method) and every class in
its MRO that is not a general part:

* the methods the class defines;
* the module-level functions of the *unit* the class comes from: the per-method package
  (``mowl/nn/<family>/<method>/``, ``mowl/models/<method>/``) when there is one, otherwise the
  module file. A computation does not become a part by moving it from a method into a function
  of the same package.
* general models (``mowl.base_models``, ``mowl.models.composed``, ``mowl.models.constructed``)
  are looked at too, one module at a time: they may wire, hold options and call parts, but not
  compute for one method nor run a private optimisation loop.

Not looked at: the general part packages (``GENERAL``: representation, training, evaluation, ...).

What counts as computing (per function, so that a helper cannot hide it):

1. a call into a tensor namespace (``torch.``, ``th.``, ``F.``, ``np.``) other than plumbing
   (building, moving, saving, seeding: ``PLUMBING``);
2. a call of an arithmetic tensor method (``.sum``, ``.norm``, ``.clamp``, ``.exp``, ... see
   ``TENSOR_METHODS``);
3. an arithmetic operator (``+ - * / // ** @``, augmented assignment; a bare sign flip ``-x`` is not counted) with an operand
   that is not statically known to be bookkeeping, in a function that touches tensors (it calls a
   tensor namespace, a tensor method or one of ``TENSOR_EVIDENCE``) -- or anywhere when the
   operator is ``/`` or ``**`` or one operand is a float (``tau_start * (tau_end / tau_start) **
   x`` is a schedule formula even though it only handles Python floats). Operands are bookkeeping
   when they are
   literals of integers, strings, containers or booleans; calls of ``len``, ``int``, ``range``,
   ``str``, ``round``, ``max``/``min`` of bookkeeping; ``.shape``, ``.size()``, ``.numel()``,
   ``.dim()``, ``.item()``; names that were assigned such an expression in the same function or
   that are named as sizes and indices (``n``, ``dim``, ``num_*``, ``*_size``, ``*_count``,
   ``*_idx``, ``epoch``, ...); and sums of those. This is a heuristic: it errs towards flagging
   in functions that handle tensors and towards allowing integer sizes, index arithmetic, string
   and set operations in functions that do not;
4. an optimisation loop: ``.backward(`` or ``.step(`` on an optimiser outside ``mowl/training``.
"""
import ast
import importlib
import inspect
import pkgutil
import re
import textwrap

TENSOR_NAMESPACES = {"torch", "th", "F", "np", "numpy"}
# Building, moving and bookkeeping are wiring, not computation.
PLUMBING = {
    "default_rng", "RandomState", "seed",
    "tensor", "as_tensor", "from_numpy", "arange", "zeros", "ones", "empty", "full",
    "zeros_like", "ones_like", "device", "no_grad", "manual_seed", "save", "load",
    "is_tensor", "Generator", "array", "asarray", "randperm", "get_default_dtype",
    "set_grad_enabled", "enable_grad", "inference_mode", "long", "float", "bool",
    "int64", "float32", "float64", "dtype", "Size", "cuda", "isfinite", "isnan",
}
# Tensor methods that compute (as opposed to ``.to``, ``.view``, ``.detach``, ``.cpu``).
TENSOR_METHODS = set(
    "sum mean clamp clamp_min clamp_max exp log log1p sqrt pow norm relu sigmoid tanh softmax "
    "log_softmax abs max min matmul mm bmm prod cumsum logsumexp square neg sub mul div "
    "sub_ add_ mul_ div_ clamp_ amax amin softplus maximum minimum where masked_fill topk "
    "argmax argmin gather scatter_add einsum std var transpose diag outer dot cos sin atan2 "
    "acosh asinh atanh cosh sinh floor ceil round sign reciprocal rsqrt lgamma digamma".split())
# Calls that show a function handles tensors (used for the operator rule).
TENSOR_EVIDENCE = {"unsqueeze", "squeeze", "reshape", "view", "detach", "to", "cpu", "expand",
                   "repeat", "index_select", "flatten", "permute", "contiguous", "clone",
                   "requires_grad_", "float", "long", "item", "tolist", "numpy"}
GENERAL = ("mowl.representation", "mowl.training", "mowl.evaluation",
           "mowl.projection", "mowl.walking", "mowl.corpus", "mowl.reasoning",
           "mowl.datasets", "mowl.owlapi", "mowl.ontology", "mowl.utils")
# General models: looked at, one module at a time.
GENERAL_MODELS = ("mowl.base_models", "mowl.models.composed", "mowl.models.constructed")
# Where per-method packages live. Other non-general packages (``mowl.geometric``: exact cones,
# attributed logics; named for what they are, not for a method) are looked at one module at a
# time, like the general models.
PER_METHOD_ROOTS = ("mowl.nn.", "mowl.models.")
LIBRARY = "mowl."
ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.MatMult, ast.FloorDiv)
_SIZE_NAME = re.compile(
    r"^(n|m|k|i|j|d|dim|dims|idx|index|indices|pos|epoch|epochs|step|steps|offset|rank|seed|col|column|columns|"
    r"size|count|length|num|n_\w+|num_\w+|\w+_(col|cols|column|dim|dims|size|sizes|count|counts|idx|index|len|"
    r"length|rank|epoch|epochs|step|steps|offset|n))$")
_SCALAR_CALLS = {"len", "int", "range", "str", "round", "repr", "ord", "hash", "id", "format",
                 "list", "tuple", "set", "dict", "sorted", "enumerate", "zip", "bool"}
_SCALAR_METHODS = {"item", "tolist", "numel", "dim", "size", "ndimension", "nelement", "format",
                   "join", "keys", "values", "items", "index", "count", "split"}


def _owner(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return node


def _bookkeeping(node, known):
    """Statically known not to be a tensor (see the module docstring)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)) \
            and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
        return True  # index arithmetic: ``h // 2``, ``i % 3``
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, float) and not isinstance(node.value, complex)
    if isinstance(node, (ast.JoinedStr, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp,
                         ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return True
    if isinstance(node, ast.Name):
        return node.id in known or bool(_SIZE_NAME.match(node.id))
    if isinstance(node, ast.Attribute):
        return node.attr in ("shape", "ndim", "device") or bool(_SIZE_NAME.match(node.attr))
    if isinstance(node, ast.Subscript):
        return _bookkeeping(node.value, known)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            if f.id in ("max", "min", "abs", "sum", "pow"):
                return bool(node.args) and all(_bookkeeping(a, known) for a in node.args)
            return f.id in _SCALAR_CALLS
        if isinstance(f, ast.Attribute):
            return f.attr in _SCALAR_METHODS
        return False
    if isinstance(node, ast.BinOp):
        return _bookkeeping(node.left, known) and _bookkeeping(node.right, known)
    if isinstance(node, ast.UnaryOp):
        return _bookkeeping(node.operand, known)
    if isinstance(node, ast.IfExp):
        return _bookkeeping(node.body, known) and _bookkeeping(node.orelse, known)
    return False


def _targets(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from _targets(element)


def _collect_known(function):
    """Names bound only to bookkeeping values in ``function``: assigned from such an expression
    (also by unpacking) or looped over such an iterable, e.g. ``range`` or ``rows.tolist()``
    (two passes, so that chains work)."""
    known = set()
    for _ in range(2):
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _bookkeeping(node.value, known):
                for target in node.targets:
                    known.update(_targets(target))
            elif isinstance(node, (ast.For, ast.comprehension)) \
                    and _bookkeeping(node.iter, known):
                known.update(_targets(node.target))
    return known


def _is_scalar_text(node):
    """String formatting and container concatenation with a literal on one side."""
    for side in (node.left, node.right):
        if isinstance(side, (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Set,
                             ast.Dict)) and not (isinstance(side, ast.Constant)
                                                 and isinstance(side.value, (int, float))):
            return True
    return False


def _touches_tensors(function):
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = _owner(node.func)
            if isinstance(owner, ast.Name) and owner.id in TENSOR_NAMESPACES:
                return True
            if node.func.attr in TENSOR_METHODS or node.func.attr in TENSOR_EVIDENCE:
                return True
    return False


def _float_or_schedule(node):
    """``/``, ``**`` and anything with a float operand."""
    if isinstance(node.op, (ast.Div, ast.Pow)):
        return True
    for side in (node.left, node.right) if isinstance(node, ast.BinOp) else (node.target, node.value):
        if isinstance(side, ast.Constant) and isinstance(side.value, float):
            return True
        if isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "float":
            return True
    return False


def computation(function):
    """The sites in a function node that compute, as ``(line, kind, text)``."""
    known = _collect_known(function)
    touches = _touches_tensors(function)
    sites = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            chain = ast.unparse(node.func)
            if ".optim." in chain or ".utils.data." in chain:
                continue
            owner = _owner(node.func)
            name = node.func.attr
            if isinstance(owner, ast.Name) and owner.id in TENSOR_NAMESPACES \
                    and "(" not in chain:
                if name not in PLUMBING:
                    sites.append((node.lineno, "namespace", chain))
            elif name in TENSOR_METHODS:
                receiver = node.func.value
                if not (isinstance(receiver, ast.Name)
                        and receiver.id in ("math", "os", "random", "statistics")):
                    sites.append((node.lineno, "method", chain))
            elif name == "backward" or (name == "step" and "opt" in chain.lower()):
                sites.append((node.lineno, "optimisation", chain))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ARITHMETIC):
            if not (_bookkeeping(node.left, known) and _bookkeeping(node.right, known)) \
                    and not _is_scalar_text(node) and (touches or _float_or_schedule(node)):
                sites.append((node.lineno, "operator", ast.unparse(node)[:60]))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ARITHMETIC):
            if not (_bookkeeping(node.target, known) and _bookkeeping(node.value, known)) \
                    and not _is_scalar_text(ast.BinOp(node.target, node.op, node.value)) \
                    and (touches or _float_or_schedule(node)):
                sites.append((node.lineno, "operator", ast.unparse(node)[:60]))
    return sites


def computing_functions(source):
    """Names of the functions and methods in ``source`` that compute (module level, class
    methods; nested functions belong to the function that contains them)."""
    tree = ast.parse(textwrap.dedent(source))
    found = []

    def visit(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if computation(node):
                    found.append(prefix + node.name)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
    visit(tree.body, "")
    return sorted(found)


def unit_of(module_name):
    """The per-method unit a module belongs to: the per-method package, or the module itself."""
    parts = module_name.split(".")
    if module_name.startswith(GENERAL_MODELS) or not module_name.startswith(PER_METHOD_ROOTS):
        return module_name
    depth = 4 if module_name.startswith("mowl.nn.") else 3
    if len(parts) <= depth:
        return module_name
    candidate = ".".join(parts[:depth])
    try:
        module = importlib.import_module(candidate)
    except ImportError:
        return module_name
    return candidate if hasattr(module, "__path__") else module_name


def _modules_of(unit):
    module = importlib.import_module(unit)
    yield module
    if hasattr(module, "__path__"):
        for info in pkgutil.walk_packages(module.__path__, unit + "."):
            try:
                yield importlib.import_module(info.name)
            except ImportError:
                continue


_unit_cache = {}


def unit_functions(unit):
    """``{qualified name: sites}`` of the module-level functions of every module of ``unit``."""
    if unit not in _unit_cache:
        found = {}
        for module in _modules_of(unit):
            try:
                tree = ast.parse(inspect.getsource(module))
            except (OSError, TypeError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if computation(node):
                        found["%s:%s" % (module.__name__, node.name)] = node.lineno
        _unit_cache[unit] = found
    return _unit_cache[unit]


def named_units(key):
    """The per-method packages a Pair key names (``octagon/abox`` -> ``mowl.nn.el.octagon``,
    ``mowl.models.octagon``), whether or not a class from them is on the MRO."""
    base = key.split("/")[0].replace("-", "_")
    found = []
    for root in ("mowl.models", "mowl.nn.el", "mowl.nn.alc", "mowl.nn.graph", "mowl.nn.lm",
                 "mowl.nn.order", "mowl.nn.query"):
        name = "%s.%s" % (root, base)
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        found.append(name)
    return found


def own_computation(cls, units=()):
    """Names (``module:Class.method`` or ``module:function``) of the methods defined by ``cls``
    and its per-method ancestors, and of the module-level functions of their per-method package
    (and of the extra ``units``), that compute with tensors."""
    found = set()
    units = set(units)
    for klass in cls.__mro__:
        where = klass.__module__
        if not where.startswith(LIBRARY) or where.startswith(GENERAL):
            continue
        if where.startswith(PER_METHOD_ROOTS + GENERAL_MODELS):
            units.add(unit_of(where))
        for name, value in vars(klass).items():
            value = getattr(value, "fget", value)
            value = getattr(value, "__func__", value)
            if not inspect.isfunction(value):
                continue
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
            except (OSError, TypeError, SyntaxError):
                continue
            if computation(tree.body[0]):
                found.add("%s:%s.%s" % (where, klass.__name__, name))
    for unit in units:
        found.update(unit_functions(unit))
    return sorted(found)


def foreign_members(module):
    """Objects a module is made of whose class is per-method code."""
    members = list(vars(module).values())
    if hasattr(module, "modules"):
        members += [m for m in module.modules() if m is not module]
    found = set()
    for member in members:
        where = type(member).__module__
        if where.startswith("mowl.") and not where.startswith(GENERAL) \
                and not where.startswith(GENERAL_MODELS):
            found.add("%s.%s" % (where, type(member).__name__))
    return sorted(found)


# -- TEMPORARY: strict-test cases whose files another helper owns in round 6 -------------------
#: Pair key -> why the strict check fails for it now and who removes the entry. Strict
#: expected failures: when the owner's work is merged and the case passes, the test run fails
#: ("XPASS(strict)") and the entry must be deleted. Nothing else is expected to fail.
TEMPORARY_EXPECTED_FAILURES = {}
for _key in ("dfalc", "dfalc/rule"):
    TEMPORARY_EXPECTED_FAILURES[_key] = (
        "TEMPORARY(r6-alc): hierarchical_loss in mowl/nn/alc/dfalc/module.py and the objective "
        "and loop in mowl/base_models/alcmodel.py / mowl/models/composed/alc.py; r6-alc owns "
        "them this round. Remove this entry at merge.")
for _key in ("falcon", "falcon/minmax", "falcon/self-pairs", "falcon/learned-nothing",
             "alcone", "alcone/old-constants"):
    TEMPORARY_EXPECTED_FAILURES[_key] = (
        "TEMPORARY(r6-alc): the FALCON objective in mowl/base_models/alcmodel.py "
        "(EmbeddingALCModel) and the private loop in mowl/models/composed/alc.py; r6-alc owns "
        "them this round. Remove this entry at merge.")
TEMPORARY_EXPECTED_FAILURES["e2r"] = (
    "TEMPORARY(r6-base): E2R-only scoring in mowl/models/composed/subspace.py "
    "(SubspaceAxiomModel.concept_indicator / axiom_loss); r6-base owns it this round. Remove "
    "this entry at merge.")
