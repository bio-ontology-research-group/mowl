from .el.elmodule import ELModule
from .el.elem.module import ELEmModule
from .el.elbe.module import ELBEModule
from .el.multielbe.module import MultiELBEModule
from .el.boxel.module import BoxELModule
from .el.boxsquaredel.module import BoxSquaredELModule
from .el.multiboxel.module import MultiBoxELModule
from .el.canonical_mixture.module import CanonicalMixtureModule
from .el.grounded.module import GroundedELModule
from .el.transbox.module import TransBoxModule
from .alc.module import ALCModule
from .alc.falcon.module import FALCONModule
from .el.elkem.module import ELKEmModule
from .el.emelpp.module import EmELPPModule, EmELVarModule
from .el.regd.module import RegDELBEModule, RegDELEmModule
from .el.mcmultibox.module import VolumeMultiBoxELModule
from .el.order_embeddings.module import OrderEmbeddingsModule
from .el.box_lattice.module import BoxLatticeModule
from .el.gumbel_box.module import GumbelBoxModule
from .graph.bcbox import BinaryCodeBoxModule
from .graph.cate import CatEModule
from .alc.alcone.module import ALConeModule
from .alc.dfalc.module import DFALCModule
from .el.boxsel.module import BoxSELModule
from .alc.e2r.module import E2RModule
from .order.module import HyperbolicConesModule, ShadowConesModule, DiskEmbeddingsModule


# Modules that are built from parts (:mod:`mowl.representation`) are imported on first use:
# the parts import this package (``ELModule``), so importing them here would be circular.
_PART_BUILT = {
    "PrismModule": "mowl.nn.el.prism.module",
    "OctagonELModule": "mowl.nn.el.octagon.module",
    "EIKEModule": "mowl.nn.el.eike.module",
    "Concept2BoxModule": "mowl.nn.el.concept2box.module",
}


def __getattr__(name):
    if name in _PART_BUILT:
        import importlib
        return getattr(importlib.import_module(_PART_BUILT[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
