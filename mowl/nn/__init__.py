from .el.elmodule import ELModule
from .el.elbe.module import ELBEModule
from .el.multielbe.module import MultiELBEModule
from .el.boxel.module import BoxELModule
from .el.boxsquaredel.module import BoxSquaredELModule
from .el.multiboxel.module import MultiBoxELModule
from .el.prism.module import PrismModule
from .el.canonical_mixture.module import CanonicalMixtureModule
from .el.grounded.module import GroundedELModule
from .el.transbox.module import TransBoxModule
from .alc.module import ALCModule
from .alc.falcon.module import FALCONModule
from .el.elkem.module import ELKEmModule
from .el.mcmultibox.module import VolumeMultiBoxELModule
from .el.octagon.module import OctagonELModule
from .el.order_embeddings.module import OrderEmbeddingsModule
from .el.box_lattice.module import BoxLatticeModule
from .el.gumbel_box.module import GumbelBoxModule
from .graph.bcbox import BinaryCodeBoxModule
from .graph.cate import CatEModule
from .alc.alcone.module import ALConeModule
from .alc.dfalc.module import DFALCModule
from .el.eike.module import EIKEModule
from .el.boxsel.module import BoxSELModule
from .alc.e2r.module import E2RModule
from .el.concept2box.module import Concept2BoxModule
from .order.module import HyperbolicConesModule, ShadowConesModule, DiskEmbeddingsModule


# The modules that are combinations of parts (ELEmbeddings, EmEL++, EmEL(var), RegD) are built
# from :mod:`mowl.representation`, which itself imports this package (for ``ELModule``); they
# are imported when first asked for.
_PART_BUILT = {
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
