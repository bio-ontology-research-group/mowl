"""Declared operations of parts: what a measure or a relation model can compute beyond the
relations every part of its kind has.

An objective that needs a quantity only some parts define (the volume product of the BoxEL
code, the negative loss of Box2EL, the enhanced existential of TransBox) does not call a
method that one class happens to have. It asks the part for the operation by name,
:meth:`Operations.operation`, and the part either computes it or refuses with
:class:`~mowl.representation.ConstructorUnavailable` saying which operation is missing and
what it means.

Each kind of part declares its optional operations in ``OPERATIONS`` (``{name: what it
computes}``). Where the operation has a definition in terms of the common interface (the
regions' ``intersect``, ``volume``, ``contract``, the measure's ``meets``, the relation
model's ``preimage`` and ``image``) the base class gives it, marked with :func:`generic`, so
that *every* part of that kind has the operation; a part whose own formula is exact or
faster overrides it. :meth:`Operations.provides` says which of the two a part has
(``"own"``, ``"generic"``) or ``None``.
"""

from mowl.representation.regions import ConstructorNotImplemented, ConstructorUnavailable

__all__ = ["Operations", "generic", "unavailable", "operation_table"]


def generic(method):
    """Mark ``method`` as the generic definition of an operation (in terms of the common
    interface of the parts of its kind)."""
    method.generic = True
    return method


def unavailable(name):
    """A placeholder for an operation that has no generic definition: calling it raises
    :class:`ConstructorUnavailable`; :meth:`Operations.provides` returns ``None``."""
    def method(self, *args, **kwargs):
        raise ConstructorNotImplemented(self._missing(name))
    method.__name__ = name
    method.unavailable = True
    return method


class Operations:
    """Mixin of the parts that declare optional operations."""

    #: ``{name: what the operation computes}``; subclasses of a kind of part extend the
    #: declaration of their base (the dictionaries are merged along the class hierarchy).
    OPERATIONS = {}

    @classmethod
    def declared_operations(cls):
        """All operations declared along the class hierarchy, ``{name: meaning}``."""
        merged = {}
        for klass in reversed(cls.__mro__):
            merged.update(klass.__dict__.get("OPERATIONS", {}))
        return merged

    @classmethod
    def provides(cls, name):
        """``"own"`` if the part defines the operation ``name`` itself, ``"generic"`` if it
        has the generic definition of its kind, ``None`` if it has none."""
        if name not in cls.declared_operations():
            raise KeyError(f"{name!r} is not an operation of {cls.__name__}; declared: "
                           f"{sorted(cls.declared_operations())}.")
        for klass in cls.__mro__:
            if name in klass.__dict__:
                method = klass.__dict__[name]
                method = getattr(method, "__func__", method)
                method = getattr(method, "__wrapped__", method)
                if getattr(method, "unavailable", False):
                    return None
                return "generic" if getattr(method, "generic", False) else "own"
        return None

    def _missing(self, name, purpose=""):
        meaning = self.declared_operations().get(name, "")
        return (f"{type(self).__name__} does not provide the operation {name!r}"
                f"{f' ({meaning})' if meaning else ''}"
                f"{f', which {purpose} needs' if purpose else ''}.")

    def operation(self, name, purpose=""):
        """The operation ``name`` of this part, as a bound method; raises
        :class:`ConstructorUnavailable` naming the operation and ``purpose`` (what needs it)
        if the part has none."""
        if self.provides(name) is None:
            raise ConstructorNotImplemented(self._missing(name, purpose))
        return getattr(self, name)


def operation_table(parts):
    """``{part class name: {operation: "own" | "generic" | None}}`` of the part classes
    ``parts``, for the documentation."""
    return {cls.__name__: {name: cls.provides(name) for name in cls.declared_operations()}
            for cls in parts}

