"""Old import path of :mod:`mowl.representation.library.text_encoders` (Text encoders (sentence-transformers wrapper, bag-of-words stand-in)).

The module is a general part and lives in :mod:`mowl.representation.library.text_encoders`; this
name is the same module object, so every attribute (also private ones) is shared.
"""
import sys

from mowl.representation.library import text_encoders as _module

sys.modules[__name__] = _module
