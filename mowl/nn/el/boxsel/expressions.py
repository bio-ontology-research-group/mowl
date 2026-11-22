"""Concept expressions of BoxSEL conditionals (Zhu et al. 2024) as a structure and its leaves.

A class expression is written with names: an IRI string is a class, ``("and", e1, e2, ...)``
is a conjunction and ``("some", role, e)`` is :math:`\\exists r.e`. A geometry that reads
conditionals (:class:`~mowl.nn.el.boxsel.module.ConditionalExpressions`) takes the expression
without names, as a nested tuple (the *structure*: ``("atom",)``, ``("and", s1, s2, ...)``,
``("some", s)``) and the class and role indices in left-to-right order (the *leaves*).
"""

__all__ = ["compile_expression"]


def _lookup(index, name, what):
    if name not in index:
        raise KeyError(f"Unknown {what} {name!r} in a conditional.")
    return index[name]


def compile_expression(expr, class_index, role_index):
    """Split ``expr`` into ``(structure, [indices])`` (see the module docstring).

    :param class_index: Mapping from class IRIs to indices.
    :param role_index: Mapping from object property IRIs to indices.
    :raises KeyError: For a class or role that is not in the mapping.
    :raises ValueError: For anything that is not an expression.
    """
    if isinstance(expr, str):
        return ("atom",), [_lookup(class_index, expr, "class")]
    kind = expr[0] if isinstance(expr, (tuple, list)) and expr else None
    if kind == "and" and len(expr) >= 3:
        structure, leaves = ["and"], []
        for operand in expr[1:]:
            sub, indices = compile_expression(operand, class_index, role_index)
            structure.append(sub)
            leaves.extend(indices)
        return tuple(structure), leaves
    if kind == "some" and len(expr) == 3:
        role = _lookup(role_index, expr[1], "object property")
        sub, indices = compile_expression(expr[2], class_index, role_index)
        return ("some", sub), [role] + indices
    raise ValueError(f"Not a concept expression: {expr!r}. Use an IRI, ('and', e1, e2, ...) "
                     "or ('some', role_iri, e).")
