"""Direct implementations of the HiT and OnT modules: hand-written losses on a text encoder
(the reference of ``tests/baselines``). The encoders and the Poincare-ball maths are the
neutral ones of :mod:`mowl.nn.lm`."""
from .hit import HiTModule
from .ont import OnTModule

__all__ = ["HiTModule", "OnTModule"]
