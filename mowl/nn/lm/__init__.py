"""Language-model based ontology embeddings in hyperbolic space (HiT, OnT).

* :mod:`mowl.nn.lm.hyperbolic`: Poincaré-ball operations.
* :mod:`mowl.nn.lm.encoders`: text encoders (sentence-transformers wrapper, a small stub).
* :mod:`mowl.nn.lm.hit`: the HiT module, a combination of parts.
* :mod:`mowl.nn.lm.ont`: the OnT module, a combination of parts (a relation model per role).

``sentence-transformers`` is an optional dependency, imported only when a pre-trained
model is loaded. The modules are combinations of parts (:mod:`mowl.representation`), which
import the Poincaré-ball operations from this package: they are loaded on first use.
"""

from mowl.nn.lm.encoders import BagOfWordsEncoder, SentenceTransformerEncoder, as_encoder

__all__ = ["BagOfWordsEncoder", "SentenceTransformerEncoder", "as_encoder", "HiTModule",
           "OnTModule", "RoleTransformation"]


def __getattr__(name):
    if name == "HiTModule":
        from mowl.nn.lm.hit import HiTModule
        return HiTModule
    if name in ("OnTModule", "RoleTransformation"):
        from mowl.nn.lm import ont
        return getattr(ont, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
