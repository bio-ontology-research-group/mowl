from .el.elmodule import ELModule
from .el.elem.module import ELEmModule
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
from .el.emelpp.module import EmELPPModule, EmELVarModule
from .el.regd.module import RegDELBEModule, RegDELEmModule
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
from . import order  # noqa: F401  (its modules are loaded on first use, see below)


def __getattr__(name):
    # Modules that are combinations of parts (mowl.representation) are loaded on first use:
    # the representation imports from mowl.nn, so mowl.nn cannot import it at start.
    if name in ("HyperbolicConesModule", "ShadowConesModule", "DiskEmbeddingsModule"):
        from .order import module
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
