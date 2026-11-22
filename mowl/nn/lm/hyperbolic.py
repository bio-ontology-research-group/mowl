"""Old import path of :mod:`mowl.representation.library.poincare_ball` (Poincare-ball operations).

The module is a general part and lives in :mod:`mowl.representation.library.poincare_ball`; this
name is the same module object, so every attribute (also private ones) is shared.
"""
import sys

from mowl.representation.library import poincare_ball as _module

sys.modules[__name__] = _module
