"""Old import path of :mod:`mowl.projection.query_graph`.

The module is a general part and lives there; this name is the same module object.
"""
import sys

import importlib

sys.modules[__name__] = importlib.import_module("mowl.projection.query_graph")
