"""Direct implementations of hyperbolic entailment cones, shadow cones and disk embeddings:
one hand-written loss module per paper (the reference of ``tests/baselines``). ``cones`` is
the baseline's own copy of the cone energies; the manifolds are the neutral ones of
:mod:`mowl.nn.order.manifolds`."""
from .module import HyperbolicConesModule, ShadowConesModule, DiskEmbeddingsModule

__all__ = ["HyperbolicConesModule", "ShadowConesModule", "DiskEmbeddingsModule"]
