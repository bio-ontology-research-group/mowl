"""Which methods of a class compute with tensors themselves.

A named method is a combination of parts. Its module class and its model class
may wire parts together, name them and hand rows to them, but the arithmetic
belongs to the parts. This reads the source of the methods a class defines
itself and reports those that call tensor functions.
"""
import ast
import inspect
import textwrap

TENSOR_NAMESPACES = {"torch", "th", "F", "np", "numpy"}
# Building, moving and bookkeeping are wiring, not computation.
PLUMBING = {
    "tensor", "as_tensor", "from_numpy", "arange", "zeros", "ones", "empty", "full",
    "zeros_like", "ones_like", "device", "no_grad", "manual_seed", "save", "load",
    "is_tensor", "Generator", "array", "asarray", "randperm", "get_default_dtype",
    "set_grad_enabled", "enable_grad", "inference_mode", "long", "float", "bool",
    "int64", "float32", "float64", "dtype", "Size", "cuda", "isfinite", "isnan",
}
GENERAL = ("mowl.representation", "mowl.training", "mowl.evaluation",
           "mowl.projection", "mowl.walking", "mowl.corpus", "mowl.reasoning",
           "mowl.datasets", "mowl.owlapi", "mowl.ontology", "mowl.utils",
           "mowl.base_models", "mowl.models.composed", "mowl.models.constructed")


def _computes(function):
    for node in ast.walk(function):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        while isinstance(owner, ast.Attribute):
            owner = owner.value
        chain = ast.unparse(node.func)
        if ".optim." in chain or ".utils.data." in chain:
            continue
        if isinstance(owner, ast.Name) and owner.id in TENSOR_NAMESPACES \
                and node.func.attr not in PLUMBING:
            return True
    return False


def own_computation(cls):
    """Names of the methods defined by ``cls`` and its per-method ancestors that
    compute with tensors. General library classes are not looked at."""
    found = []
    for klass in cls.__mro__:
        if not klass.__module__.startswith("mowl.") or klass.__module__.startswith(GENERAL):
            continue
        for name, value in vars(klass).items():
            value = getattr(value, "fget", value)
            value = getattr(value, "__func__", value)
            if not inspect.isfunction(value):
                continue
            try:
                tree = ast.parse(textwrap.dedent(inspect.getsource(value)))
            except (OSError, TypeError, SyntaxError):
                continue
            if _computes(tree):
                found.append("%s.%s" % (klass.__name__, name))
    return sorted(found)


def foreign_members(module):
    """Objects a module is made of whose class is per-method code."""
    members = list(vars(module).values())
    if hasattr(module, "modules"):
        members += [m for m in module.modules() if m is not module]
    found = set()
    for member in members:
        where = type(member).__module__
        if where.startswith("mowl.") and not where.startswith(GENERAL):
            found.add("%s.%s" % (where, type(member).__name__))
    return sorted(found)
