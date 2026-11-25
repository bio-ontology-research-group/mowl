"""The loss functions of the modules ``mowl.nn.el.{elem,elbe,boxel,boxsquaredel}.losses`` of
mOWL 2.1.0, computed by the parts.

In 2.1.0 every EL module called functions such as ``gci0_loss(data, class_embed, class_rad,
margin, neg=False)`` with its embedding tables. The formulas are now parts
(:mod:`mowl.representation`: class models, relation models, measures, objectives), and these
functions are kept, deprecated, for code that called them: each one builds the module of the
method with the options that give the released numbers, puts the tables it is given in place
of the module's own (so gradients reach them), and returns the module's loss in the shape the
released function had. The tensor-level helpers of those modules (``ball_inclusion_score``,
``Box``, ``box_distance``, ...) are not kept: they are measures of
:mod:`mowl.representation` now.
"""

import warnings

__all__ = ["released_loss", "released_regularization"]


def _deprecated(module, name):
    warnings.warn(f"{module}.{name} is deprecated: the loss is a part of mowl.representation "
                  f"(use the module's {name}, or the measure and objective of its parts).",
                  DeprecationWarning, stacklevel=3)


def _share(module, path, table):
    """Put ``table`` (an :class:`torch.nn.Embedding`) in place of the table at ``path`` of
    ``module`` and of every other name of that same table (aliases)."""
    owner, _, name = path.rpartition(".")
    old = module.get_submodule(path)
    for parent in list(module.modules()):
        for child_name, child in list(parent.named_children()):
            if child is old:
                setattr(parent, child_name, table)
    if module.get_submodule(path) is not table:
        setattr(module.get_submodule(owner) if owner else module, name, table)


#: Rows of a table the caller did not give (the loss of the form does not read it); set to
#: more than the largest index of the rows by :func:`_module`.
_ABSENT = [1]


def _size(table, default=None):
    if table is not None:
        return int(table.num_embeddings)
    return _ABSENT[0] if default is None else default


def _dim(tables):
    for table in tables.values():
        if table is not None and table.embedding_dim > 1:
            return int(table.embedding_dim)
    return 1


def _elem(tables, margin=0.1, reg_norm=1):
    from mowl.nn.el.elem.module import ELEmModule
    classes = tables.get("class_embed") or tables.get("class_rad")
    inds = tables.get("ind_embed") or tables.get("ind_rad")
    module = ELEmModule(_size(classes), _size(tables.get("rel_embed")),
                        _size(inds, 0) or None, embed_dim=_dim(tables), margin=margin,
                        reg_norm=reg_norm, gci1_radius_term=False)
    return module, {"class_embed": "class_model.centres", "class_rad": "class_model.radii",
                    "rel_embed": "relation_model.vectors",
                    "ind_embed": "individual_model.centres",
                    "ind_rad": "individual_model.radii"}


def _elbe(tables, margin=0.1):
    from mowl.nn.el.elbe.module import ELBEModule
    classes = tables.get("class_embed") or tables.get("class_offset")
    inds = tables.get("ind_embed") or tables.get("ind_offset")
    module = ELBEModule(_size(classes), _size(tables.get("rel_embed")), _size(inds, 0) or None,
                        embed_dim=_dim(tables), margin=margin)
    return module, {"class_embed": "class_model.centres", "class_offset": "class_model.offsets",
                    "rel_embed": "relation_model.vectors",
                    "ind_embed": "individual_model.centres",
                    "ind_offset": "individual_model.offsets"}


def _boxel(tables, temperature=1.0):
    from mowl.nn.el.boxel.module import BoxELModule
    classes = tables.get("min_embed") or tables.get("delta_embed")
    inds = tables.get("ind_embed") or tables.get("ind_delta_embed")
    relations = tables.get("relation_embed") or tables.get("scaling_embed")
    module = BoxELModule(_size(classes), _size(relations), _size(inds, 0) or None,
                         embed_dim=_dim(tables), temperature=temperature,
                         positive_scale=False, role_chain_composition="as_released",
                         gci3_preimage="as_released")
    return module, {"min_embed": "class_model.mins", "delta_embed": "class_model.deltas",
                    "relation_embed": "relation_model.vectors",
                    "scaling_embed": "relation_model.scales",
                    "ind_embed": "individual_model.mins",
                    "ind_delta_embed": "individual_model.deltas"}


def _box2el(tables, gamma=0, delta=2, reg_factor=0.05):
    from mowl.nn.el.boxsquaredel.module import BoxSquaredELModule
    classes = tables.get("class_center") or tables.get("class_offset") or tables.get("bump")
    inds = tables.get("ind_center") or tables.get("ind_offset") or \
        tables.get("bump_individuals")
    relations = next((tables[k] for k in ("head_center", "head_offset", "tail_center",
                                          "tail_offset") if tables.get(k) is not None), None)
    module = BoxSquaredELModule(_size(classes), _size(relations), _size(inds, 0) or None,
                                embed_dim=_dim(tables), gamma=gamma, delta=delta,
                                reg_factor=reg_factor)
    return module, {"class_center": "class_model.centres",
                    "class_offset": "class_model.offsets", "bump": "class_model.bumps",
                    "head_center": "relation_model.head_centres",
                    "head_offset": "relation_model.head_offsets",
                    "tail_center": "relation_model.tail_centres",
                    "tail_offset": "relation_model.tail_offsets",
                    "ind_center": "individual_model.centres",
                    "ind_offset": "individual_model.offsets",
                    "bump_individuals": "individual_model.bumps"}


_METHODS = {"elem": _elem, "elbe": _elbe, "boxel": _boxel, "boxsquaredel": _box2el}


def _module(method, tables, options, data=None):
    _ABSENT[0] = 1 if data is None or data.numel() == 0 else int(data.max()) + 1
    module, paths = _METHODS[method](tables, **options)
    for name, table in tables.items():
        if table is not None:
            _share(module, paths[name], table)
    module.per_axiom_column = False      # the shapes of the released functions
    module.strict_negatives = False      # they ignored ``neg`` where there is no negative
    return module


def released_loss(method, form, data, neg, tables, **options):
    """The loss of the ``form`` axioms ``data`` that the released function
    ``mowl.nn.el.<method>.losses.<form>_loss`` computed with the embedding tables ``tables``
    (``{released argument name: nn.Embedding}``) and ``options`` (``margin``,
    ``temperature``, ``gamma``, ``delta``), computed by the module of the method.

    .. deprecated:: 2.2.0
    """
    _deprecated(f"mowl.nn.el.{method}.losses", f"{form}_loss")
    module = _module(method, tables, options, data)
    return module.get_loss_function(form)(data, neg=neg)


def released_regularization(method, tables, **options):
    """The regularisation term of the module of ``method`` on the given tables (the released
    ``regularization_loss`` / ``reg_loss`` functions).

    .. deprecated:: 2.2.0
    """
    _deprecated(f"mowl.nn.el.{method}.losses", "regularization_loss")
    return _module(method, tables, options).regularization_loss()
