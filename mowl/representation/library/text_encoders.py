"""Text encoders for the language-model based methods (HiT, OnT).

An *encoder* is a :class:`torch.nn.Module` whose ``forward`` takes a list of strings and
returns a differentiable tensor of shape ``(len(texts), dim)``, and that has an integer
attribute ``dim``. Two encoders are provided:

* :class:`SentenceTransformerEncoder` wraps a `sentence-transformers
  <https://www.sbert.net>`_ model (optional dependency, imported lazily). This is what HiT
  and OnT use: a pre-trained transformer with mean pooling and *without* the normalisation
  layer that some sentence-transformers models have (HiT, Section 3).
* :class:`BagOfWordsEncoder` is a tiny, dependency-free stand-in (hashed word embeddings,
  mean pooling, ``tanh``). It is meant for tests and documentation examples only; it has
  none of the pre-trained knowledge the methods rely on.
"""

import logging
import re
import zlib

import torch as th
from torch import nn

logger = logging.getLogger(__name__)

_INSTALL_HINT = ("The language-model based methods (HiT, OnT) need the optional package "
                 "'sentence-transformers' (pip install sentence-transformers).")


def _require_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return sentence_transformers


def _st_modules():
    """``(Transformer, Pooling, Normalize)`` of sentence-transformers 6 or of earlier
    versions (``sentence_transformers.models``)."""
    _require_sentence_transformers()
    try:
        from sentence_transformers.sentence_transformer.modules import (Normalize, Pooling,
                                                                        Transformer)
    except ImportError:
        from sentence_transformers.models import Normalize, Pooling, Transformer
    return Transformer, Pooling, Normalize


class BagOfWordsEncoder(nn.Module):
    """Hashed bag-of-words encoder: lower-cased words are hashed (CRC32) into
    ``num_buckets`` learnable vectors, averaged, and passed through ``tanh``, so outputs
    lie in :math:`[-1, 1]^d` like those of the encoders HiT assumes.

    For tests and examples; not a language model.

    :param dim: Output dimension (even, so that OnT's rotations apply).
    :param num_buckets: Number of hash buckets.
    :param scale: Standard deviation of the initial word vectors.
    """

    def __init__(self, dim=16, num_buckets=1024, scale=0.5):
        super().__init__()
        self.dim = dim
        self.num_buckets = num_buckets
        self.embedding = nn.EmbeddingBag(num_buckets, dim, mode="mean")
        nn.init.normal_(self.embedding.weight, std=scale)

    def _bucket(self, word):
        return zlib.crc32(word.encode("utf-8")) % self.num_buckets

    def forward(self, texts):
        ids, offsets = [], []
        for text in texts:
            offsets.append(len(ids))
            words = re.findall(r"\w+", text.lower()) or ["<empty>"]
            ids.extend(self._bucket(w) for w in words)
        device = self.embedding.weight.device
        out = self.embedding(th.tensor(ids, dtype=th.long, device=device),
                             th.tensor(offsets, dtype=th.long, device=device))
        return th.tanh(out)


class SentenceTransformerEncoder(nn.Module):
    """A sentence-transformers model used as a differentiable text encoder.

    :param model: A ``sentence_transformers.SentenceTransformer``. A ``Normalize``
        module, if present, is removed (HiT, Section 3: normalisation would fix all norms
        to one and prevent the hierarchical organisation by norm).
    """

    def __init__(self, model):
        super().__init__()
        st = _require_sentence_transformers()
        modules = list(model._modules.values())
        normalize = _st_modules()[2]
        if any(isinstance(m, normalize) for m in modules):
            logger.warning("Removing the Normalize layer of the sentence-transformers model "
                           "(HiT/OnT use unnormalised mean-pooled embeddings).")
            model = st.SentenceTransformer(
                modules=[m for m in modules if not isinstance(m, normalize)],
                device=str(next(model.parameters()).device))
        self.model = model
        get_dim = getattr(model, "get_embedding_dimension", None) or \
            model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        # sentence-transformers >= 6 renamed tokenize to preprocess
        self._preprocess = getattr(model, "preprocess", None) or model.tokenize

    @classmethod
    def from_pretrained(cls, name, max_seq_length=256, pooling_mode="mean", device=None,
                        revision=None):
        """Load a Hugging Face transformer with a mean-pooling layer and no normalisation,
        as HiT's ``HierarchyTransformer.from_pretrained`` does.

        :param name: Model name or path, e.g. ``"sentence-transformers/all-MiniLM-L12-v2"``.
        """
        import inspect
        st = _require_sentence_transformers()
        transformer_cls, pooling_cls, _ = _st_modules()
        kwargs = {"max_seq_length": max_seq_length}
        if revision:
            params = inspect.signature(transformer_cls.__init__).parameters
            key = "model_kwargs" if "model_kwargs" in params else "model_args"
            kwargs[key] = {"revision": revision}
        transformer = transformer_cls(name, **kwargs)
        dim = getattr(transformer, "get_embedding_dimension", None) or \
            transformer.get_word_embedding_dimension
        pooling = pooling_cls(dim(), pooling_mode=pooling_mode)
        return cls(st.SentenceTransformer(modules=[transformer, pooling], device=device))

    def forward(self, texts):
        features = self._preprocess(list(texts))
        device = next(self.model.parameters()).device
        features = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in features.items()}
        return self.model(features)["sentence_embedding"]

    def save(self, path):
        """Save the underlying sentence-transformers model to ``path``."""
        self.model.save(path)


def _is_sentence_transformer(encoder):
    """Whether ``encoder`` is a ``sentence_transformers.SentenceTransformer`` (False where the
    package is not installed: then no such object exists)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False
    return isinstance(encoder, SentenceTransformer)


def as_encoder(encoder, device=None):
    """Turn a model name, a sentence-transformers model or an encoder module into an
    encoder (see the module docstring).

    :param encoder: A string (Hugging Face model name or local path, loaded with
        :meth:`SentenceTransformerEncoder.from_pretrained`), a
        ``SentenceTransformer`` instance, or an :class:`torch.nn.Module` mapping a list
        of strings to a tensor.
    :raises ImportError: If a string or sentence-transformers model is given and
        sentence-transformers is not installed.
    """
    if isinstance(encoder, str):
        enc = SentenceTransformerEncoder.from_pretrained(encoder, device=device)
    elif _is_sentence_transformer(encoder):
        enc = SentenceTransformerEncoder(encoder)
    elif isinstance(encoder, nn.Module):
        enc = encoder
    else:
        raise TypeError("encoder must be a model name, a SentenceTransformer, or a "
                        "torch.nn.Module mapping a list of strings to a tensor.")
    if not hasattr(enc, "dim"):
        with th.no_grad():
            enc.dim = int(enc(["thing"]).shape[-1])
    if device is not None:
        enc.to(device)
    return enc
