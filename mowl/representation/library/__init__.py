"""Parts and methods beyond the core: one module per group of methods.

Every module of this package is imported when :mod:`mowl.representation` is, and registers
what it defines (:func:`~mowl.representation.methods.register_class_model`,
:func:`~mowl.representation.methods.register_relation_model`,
:func:`~mowl.representation.methods.register_measure`,
:func:`~mowl.representation.methods.register_method`,
:func:`~mowl.representation.adapters.register_adapter`). Adding a method therefore means
adding a module here, and nothing else.
"""

import importlib
import pkgutil

for _name in sorted(info.name for info in pkgutil.iter_modules(__path__)):
    importlib.import_module(f"{__name__}.{_name}")
