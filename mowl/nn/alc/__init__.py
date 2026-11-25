from mowl.nn.alc.module import ALCModule

# The modules of the methods are put together from parts and loaded when they are first asked
# for (see :mod:`mowl.nn`): ``mowl.nn.alc.falcon``, ``mowl.nn.alc.dfalc``,
# ``mowl.nn.alc.alcone``, ``mowl.nn.alc.e2r``.
__all__ = ["ALCModule", "FALCONModule"]


def __getattr__(name):
    # ``mowl.nn.alc.FALCONModule`` (a name of 2.1.0), loaded on first use
    if name == "FALCONModule":
        from mowl.nn.alc.falcon.module import FALCONModule
        globals()[name] = FALCONModule
        return FALCONModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
