from .el.elmodule import ELModule
from .el.elem.module import ELEmModule
from .el.elbe.module import ELBEModule
from .el.multielbe.module import MultiELBEModule
from .el.boxel.module import BoxELModule
from .el.boxsquaredel.module import BoxSquaredELModule
from .el.multiboxel.module import MultiBoxELModule
from .el.prism.module import PrismModule
from .el.canonical_mixture.module import CanonicalMixtureModule
from .el.transbox.module import TransBoxModule
from .alc.module import ALCModule
from .el.elkem.module import ELKEmModule
from .el.emelpp.module import EmELPPModule, EmELVarModule
from .el.regd.module import RegDELBEModule, RegDELEmModule
from .el.mcmultibox.module import VolumeMultiBoxELModule
from .el.octagon.module import OctagonELModule
from .el.order_embeddings.module import OrderEmbeddingsModule
from .el.box_lattice.module import BoxLatticeModule
from .el.gumbel_box.module import GumbelBoxModule
from .graph.bcbox import BinaryCodeBoxModule
from .graph.cate import CatEModule
from .el.eike.module import EIKEModule
from .el.boxsel.module import BoxSELModule
from .el.concept2box.module import Concept2BoxModule
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
}


def __getattr__(name):
    if name in _PART_BUILT:
        import importlib
        value = getattr(importlib.import_module(_PART_BUILT[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
