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
3. an arithmetic operator (``+ - * / // ** @``, augmented assignment; a bare sign flip ``-x``
   is not counted) in any of three situations:

   a. an operand is *tensor evidence*: a call of a tensor namespace or tensor method, a call of
      a part (``self.<part>(...)``, ``self.<part>.<method>(...)``), a name assigned from such an
      expression (also through ``x = ...; y = x ...`` chains), or an attribute, item, sign flip
      or arithmetic of one. Evidence is per value, so ``loss + self.weight * self.regulariser(x)``
      is found even in a method that calls no ``torch``;
   b. the function handles tensors anywhere (it calls a tensor namespace, a tensor method, one
      of ``TENSOR_EVIDENCE`` or a part) and an operand is not *integer evidence*;
   c. the operator is ``/`` or ``**``, or one operand is a float (``tau_start * (tau_end /
      tau_start) ** x`` is a schedule formula even though it only handles Python floats).

   *Integer evidence* is a fact about the value, never about its name. Operands count as
   bookkeeping when they are literals of integers, strings, containers or booleans; calls of
   ``len``, ``int``, ``range``, ``str``, ``round``, ``max``/``min`` of bookkeeping; ``.shape``,
   ``.size()``, ``.numel()``, ``.dim()``, ``.item()``; names assigned such an expression in the
   same function, looped over ``range``/``enumerate``/``zip`` of such, parameters annotated
   ``int``/``str``/``bool`` or with an integer/string/boolean literal default, and ``self.<a>``
   where the class (or a base class) assigns ``self.<a>`` from such a value or defines ``<a>``
   as an integer constant;
   ``x // k`` and ``x % k`` with an integer literal ``k``; and sums, products and differences of
   those. A name such as ``offset``, ``pos``, ``d``, ``n`` or ``num_classes`` is *not* evidence:
   ``offset + pos`` in a function that handles tensors is flagged unless both were derived from
   ``len``/``shape``/``range``/``int``. The check still lets through, in functions without any
   tensor evidence, arithmetic on operands of unknown type (string joining, set differences,
   index arithmetic on attributes), because it cannot tell them from tensors there; a value that
   comes from a part (called on ``self``) or a tensor call is never let through. Parts reached
   through another object (``model.module(x)``) are not recognised as parts.
4. an optimisation loop: ``.backward(`` or ``.step(`` on an optimiser outside ``mowl/training``.
"""
import ast
import importlib
import inspect
import pkgutil
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
_SCALAR_CALLS = {"len", "int", "range", "str", "round", "repr", "ord", "hash", "id", "format",
                 "list", "tuple", "set", "dict", "sorted", "enumerate", "zip", "bool"}
_SCALAR_METHODS = {"item", "tolist", "numel", "dim", "size", "ndimension", "nelement", "format",
                   "join", "keys", "values", "items", "index", "count", "split"}


def _owner(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return node


def _bookkeeping(node, known):
    """Statically known not to be a tensor (see the module docstring). ``known`` holds names and
    ``self.<attribute>`` spellings with integer/string evidence."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)) \
            and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
        return True  # index arithmetic: ``h // 2``, ``i % 3``
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, float) and not isinstance(node.value, complex)
    if isinstance(node, (ast.JoinedStr, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp,
                         ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return True
    if isinstance(node, ast.Name):
        return node.id in known
    if isinstance(node, ast.Attribute):
        return node.attr in ("shape", "ndim", "device") or ast.unparse(node) in known
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


_ITERATORS = {"enumerate", "zip", "sorted", "list", "tuple", "set", "reversed", "dict"}


def _iterates_bookkeeping(node, known):
    """A loop over this yields bookkeeping: ``range``, ``enumerate``/``zip``/``sorted`` of such,
    or a value that is bookkeeping itself (``rows.tolist()``)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range":
            return True
        if node.func.id in _ITERATORS:
            return all(_iterates_bookkeeping(a, known) for a in node.args)
    return _bookkeeping(node, known)


def _plain(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, str, bool)) \
        and node.value is not None


def class_int_attributes(tree):
    """``{"self.a", ...}`` for the attributes a class (AST) assigns from integer/string
    evidence: ``self.a = <bookkeeping>`` in any method, and ``a = <integer literal>`` in the
    class body."""
    found = set()
    for node in tree.body if isinstance(tree, ast.ClassDef) else ():
        if isinstance(node, ast.Assign) and _plain(node.value):
            found.update("self." + t for t in _targets(node.targets[0]))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            known = _collect_known(node)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and _bookkeeping(sub.value, known):
                    for target in sub.targets:
                        if isinstance(target, ast.Attribute) \
                                and ast.unparse(target.value) == "self":
                            found.add(ast.unparse(target))
    return found


def _collect_known(function, attributes=()):
    """Names that are integers, strings or containers *by evidence*: parameters annotated
    ``int``/``str``/``bool``, names assigned a bookkeeping expression (also by unpacking), and
    loop variables over ``range`` or over bookkeeping (two passes, so that chains work). A name
    is never bookkeeping because of how it is spelled."""
    known = set(attributes)
    for arg in ast.walk(function):
        if isinstance(arg, ast.arg) and isinstance(arg.annotation, ast.Name) \
                and arg.annotation.id in ("int", "str", "bool"):
            known.add(arg.arg)
    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = function.args
        positional = args.posonlyargs + args.args
        for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
            if _plain(default):
                known.add(arg.arg)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None and _plain(default):
                known.add(arg.arg)
    for _ in range(2):
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _bookkeeping(node.value, known):
                for target in node.targets:
                    known.update(_targets(target))
            elif isinstance(node, (ast.For, ast.comprehension)) \
                    and _iterates_bookkeeping(node.iter, known):
                known.update(_targets(node.target))
    return known


def _is_tensor_call(node):
    """A call that returns tensor-like values: a tensor namespace, a tensor method, a tensor
    helper, or a part (anything called on ``self``) that is not a known scalar method."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    name = node.func.attr
    if name in _SCALAR_METHODS:
        return False
    owner = _owner(node.func)
    if isinstance(owner, ast.Name) and (owner.id in TENSOR_NAMESPACES or owner.id == "self"):
        return True
    return name in TENSOR_METHODS or name in TENSOR_EVIDENCE


def _tensorish(node, tensors):
    """Tensor evidence for a value: it comes from a tensor call or a part, from a name that
    was assigned one, or is an attribute, item, sign flip or arithmetic of one."""
    if isinstance(node, ast.Call):
        return _is_tensor_call(node) or (
            isinstance(node.func, ast.Name) and node.func.id in ("max", "min", "abs", "sum")
            and any(_tensorish(a, tensors) for a in node.args))
    if isinstance(node, ast.Name):
        return node.id in tensors
    if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        return _tensorish(node.value, tensors)
    if isinstance(node, ast.UnaryOp):
        return _tensorish(node.operand, tensors)
    if isinstance(node, ast.BinOp):
        return _tensorish(node.left, tensors) or _tensorish(node.right, tensors)
    if isinstance(node, ast.IfExp):
        return _tensorish(node.body, tensors) or _tensorish(node.orelse, tensors)
    return False


def _collect_tensors(function, known):
    """Names bound to a value with tensor evidence (assignment, unpacking, augmented
    assignment, ``for x in <tensor>``); two passes so that chains ``a = f(x); b = a * 2`` work."""
    tensors = set()
    for _ in range(3):
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and _tensorish(node.value, tensors):
                for target in node.targets:
                    tensors.update(t for t in _targets(target) if t not in known)
            elif isinstance(node, ast.AugAssign) and _tensorish(node.value, tensors):
                tensors.update(t for t in _targets(node.target) if t not in known)
            elif isinstance(node, (ast.For, ast.comprehension)) and _tensorish(node.iter, tensors):
                tensors.update(t for t in _targets(node.target) if t not in known)
    return tensors


def _is_scalar_text(node):
    """String formatting and container concatenation with a literal on one side."""
    for side in (node.left, node.right):
        if isinstance(side, (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Set,
                             ast.Dict)) and not (isinstance(side, ast.Constant)
                                                 and isinstance(side.value, (int, float))):
            return True
    return False


def _touches_tensors(function):
    """The function handles tensors somewhere: it calls a tensor namespace, a tensor method, a
    tensor helper (``TENSOR_EVIDENCE``) or a part."""
    return any(_is_tensor_call(node) for node in ast.walk(function))


def _float_or_schedule(node):
    """``/``, ``**`` and anything with a float operand."""
    if isinstance(node.op, (ast.Div, ast.Pow)):
        return True
    sides = (node.left, node.right) if isinstance(node, ast.BinOp) \
        else (node.target, node.value)
    for side in sides:
        if isinstance(side, ast.Constant) and isinstance(side.value, float):
            return True
        if isinstance(side, ast.Call) and isinstance(side.func, ast.Name) \
                and side.func.id == "float":
            return True
    return False


def _operator_computes(node, known, tensors, touches):
    """Whether an arithmetic operator node computes (rules 3a-3c of the module docstring)."""
    if isinstance(node, ast.BinOp):
        left, right = node.left, node.right
    else:
        left, right = node.target, node.value
    if _is_scalar_text(ast.BinOp(left, node.op, right)):
        return False  # ``regimes + [X]``, ``'a' + b``: a container or text on one side
    if _tensorish(left, tensors) or _tensorish(right, tensors):
        return True
    if _bookkeeping(left, known) and _bookkeeping(right, known):
        return False
    return touches or _float_or_schedule(node)


def computation(function, attributes=()):
    """The sites in a function node that compute, as ``(line, kind, text)``. ``attributes`` are
    ``self.<a>`` spellings known to be integers (``class_int_attributes``)."""
    known = _collect_known(function, attributes)
    tensors = _collect_tensors(function, known)
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
            if _operator_computes(node, known, tensors, touches):
                sites.append((node.lineno, "operator", ast.unparse(node)[:60]))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ARITHMETIC):
            if _operator_computes(node, known, tensors, touches):
                sites.append((node.lineno, "operator", ast.unparse(node)[:60]))
    return sites


def computing_functions(source):
    """Names of the functions and methods in ``source`` that compute (module level, class
    methods; nested functions belong to the function that contains them)."""
    tree = ast.parse(textwrap.dedent(source))
    found = []

    def visit(body, prefix, attributes=()):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if computation(node, attributes):
                    found.append(prefix + node.name)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".", class_int_attributes(node))
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


_attribute_cache = {}


def _class_attributes(klass):
    if klass not in _attribute_cache:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
            _attribute_cache[klass] = class_int_attributes(tree.body[0])
        except (OSError, TypeError, SyntaxError, IndexError):
            _attribute_cache[klass] = set()
    return _attribute_cache[klass]


def own_computation(cls, units=()):
    """Names (``module:Class.method`` or ``module:function``) of the methods defined by ``cls``
    and its per-method ancestors, and of the module-level functions of their per-method package
    (and of the extra ``units``), that compute with tensors."""
    found = set()
    units = set(units)
    attributes = set()  # integers the object holds, assigned by any class of its MRO
    for base in cls.__mro__:
        if base.__module__.startswith(LIBRARY):
            attributes |= _class_attributes(base)
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
            if computation(tree.body[0], attributes):
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
