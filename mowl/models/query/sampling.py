"""Old import path of :mod:`mowl.training.query_structures`.

The module is a general part and lives there; this name is the same module object.
"""
import sys

import importlib

sys.modules[__name__] = importlib.import_module("mowl.training.query_structures")
