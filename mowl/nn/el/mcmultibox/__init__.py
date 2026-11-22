"""The volume-based MultiBoxEL: its module.

The maths of sampled volumes is part of the parts of the method
(:mod:`mowl.representation.library.multibox_parts`), which the module is built from, so the
module is loaded when it is asked for and not when the package is imported.
"""

__all__ = ["VolumeMultiBoxELModule"]


def __getattr__(name):
    if name == "VolumeMultiBoxELModule":
        from mowl.nn.el.mcmultibox.module import VolumeMultiBoxELModule
        return VolumeMultiBoxELModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
