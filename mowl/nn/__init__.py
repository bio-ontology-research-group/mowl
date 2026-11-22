from .el.elmodule import ELModule
# The part-built modules (ELBEModule, ...) are made of the parts of mowl.representation, which
# in turn read the base classes of this package: each imports the other first.
import mowl.representation  # noqa: F401  (must stay after ELModule, before the modules)
from .el.elbe.module import ELBEModule
from .el.multielbe.module import MultiELBEModule
from .el.boxel.module import BoxELModule
from .el.boxsquaredel.module import BoxSquaredELModule
from .el.multiboxel.module import MultiBoxELModule
from .el.canonical_mixture.module import CanonicalMixtureModule
from .el.transbox.module import TransBoxModule
from .alc.module import ALCModule
from .el.elkem.module import ELKEmModule
from .el.mcmultibox.module import VolumeMultiBoxELModule
from .el.order_embeddings.module import OrderEmbeddingsModule
from .el.box_lattice.module import BoxLatticeModule
from .el.gumbel_box.module import GumbelBoxModule
from .graph.bcbox import BinaryCodeBoxModule
from .graph.cate import CatEModule
from .el.boxsel.module import BoxSELModule
from . import order  # noqa: F401  (its modules are loaded on first use, see below)


# Modules that are put together from parts (:mod:`mowl.representation`) are loaded when they
# are first asked for: the parts import :class:`ELModule`, so importing them here, while this
# package is being imported (as :mod:`mowl.representation.geometry` does), would be circular.
_PART_BUILT = {
    "GroundedELModule": "mowl.nn.el.grounded.module",
    "FALCONModule": "mowl.nn.alc.falcon.module",
    "ALConeModule": "mowl.nn.alc.alcone.module",
    "DFALCModule": "mowl.nn.alc.dfalc.module",
    "E2RModule": "mowl.nn.alc.e2r.module",
    "HyperbolicConesModule": "mowl.nn.order.module",
    "ShadowConesModule": "mowl.nn.order.module",
    "DiskEmbeddingsModule": "mowl.nn.order.module",
    "PrismModule": "mowl.nn.el.prism.module",
    "OctagonELModule": "mowl.nn.el.octagon.module",
    "EIKEModule": "mowl.nn.el.eike.module",
    "Concept2BoxModule": "mowl.nn.el.concept2box.module",
    "ELEmModule": "mowl.nn.el.elem.module",
    "EmELPPModule": "mowl.nn.el.emelpp.module",
    "EmELVarModule": "mowl.nn.el.emelpp.module",
    "RegDELBEModule": "mowl.nn.el.regd.module",
    "RegDELEmModule": "mowl.nn.el.regd.module",
}


def __getattr__(name):
    if name in _PART_BUILT:
        import importlib
        value = getattr(importlib.import_module(_PART_BUILT[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
