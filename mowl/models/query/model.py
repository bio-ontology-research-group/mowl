"""Old import path of :mod:`mowl.models.composed.query`.

The module is a general part and lives there; this name is the same module object.
"""
import sys

import importlib

sys.modules[__name__] = importlib.import_module("mowl.models.composed.query")
