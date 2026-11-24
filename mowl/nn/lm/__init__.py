"""Language-model based ontology embeddings in hyperbolic space (HiT, OnT).

* :mod:`mowl.nn.lm.hyperbolic`: Poincaré-ball operations.
* :mod:`mowl.nn.lm.encoders`: text encoders (sentence-transformers wrapper, a small stub).
* :mod:`mowl.nn.lm.hit`: HiT losses and module.
* :mod:`mowl.nn.lm.ont`: OnT role transformations, logical losses and module.

``sentence-transformers`` is an optional dependency, imported only when a pre-trained
model is loaded.
"""

from mowl.nn.lm.encoders import BagOfWordsEncoder, SentenceTransformerEncoder, as_encoder
from mowl.nn.lm.hit import HiTModule
from mowl.nn.lm.ont import OnTModule, RoleTransformation

__all__ = ["BagOfWordsEncoder", "SentenceTransformerEncoder", "as_encoder", "HiTModule",
           "OnTModule", "RoleTransformation"]
