from .manifolds import RiemannianSGD

__all__ = ["OrderModule", "HyperbolicConesModule", "ShadowConesModule",
           "DiskEmbeddingsModule", "RiemannianSGD"]

_MODULES = ("OrderModule", "HyperbolicConesModule", "ShadowConesModule",
            "DiskEmbeddingsModule")


def __getattr__(name):
    # The modules are combinations of parts (mowl.representation), which import the
    # manifolds of this package: they are loaded on first use, not with the package.
    if name in _MODULES:
        from mowl.nn.order import module
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
