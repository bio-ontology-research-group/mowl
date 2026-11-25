"""EIKE: extensional and intensional knowledge embedding (Wang et al., 2024)."""
import re
from urllib.parse import unquote

import numpy as np
import torch as th

from mowl.baselines.base_models.elmodel import EmbeddingELModel
from mowl.baselines.nn.el.eike.module import EIKEModule


def name_to_text(iri):
    """The text the paper feeds to the language model for a concept without description: the
    name part of its IRI, underscores replaced by spaces, surrounding brackets stripped."""
    name = unquote(re.split(r"[#/]", iri.strip("<>"))[-1])
    return name.replace("_", " ").strip("<> ")


class EIKE(EmbeddingELModel):
    """EIKE [wang2024eike]_: concepts and instances in an extensional and an intensional space.

    Extensional space: concepts are ellipsoids, instances points, relations vectors.
    Intensional space: concepts are vectors (initialised from language-model vectors you pass
    in, see ``intensional_vectors`` and :meth:`encode_class_texts`), instances are
    ``M i``. Trained on three kinds of triples with the margin-based ranking loss of the
    paper: SubClassOf (``gci0`` of the normalised ontology), InstanceOf (class assertions)
    and relational triples (object property assertions). Axioms of the other normal forms
    (``gci1``, ``gci2``, ``gci3``, bottom forms, role axioms) are not used.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of both spaces (the paper uses 100). Must equal the
        dimension of ``intensional_vectors`` if given.
    :param alpha: Weight of the intensional scores (paper: from {0.1, 0.5, 1, 2}).
    :param margins: ``(gamma_sub, gamma_ins, gamma_rel)`` of the ranking losses (paper: each
        from {0.1, 0.3, 0.4, 0.5, 1, 2}).
    :param instance_map: ``"eye"`` (EYE) or ``"matrix"`` (MAT) for the virtual instance
        embedding :math:`i^i = M i^e`.
    :param intensional_vectors: Optional dict class IRI -> vector (e.g. Sentence-BERT
        encodings of the class names), or the path of a file written by
        :meth:`save_intensional_vectors`; classes missing from it keep a random vector (the
        paper's UNP variant for them). No language model is needed or downloaded by mOWL.
    :param freeze_intensional: Do not train the intensional concept vectors.
    :param neg_sampling_gcis: As in :class:`~mowl.base_models.EmbeddingELModel`.
    """

    axiom_scoring = "module"
    scored_forms = ("gci0", "class_assertion", "object_property_assertion")
    _EIKE_FORMS = ("gci0", "class_assertion", "object_property_assertion")

    def __init__(self, dataset, embed_dim=100, alpha=1.0, margins=(1.0, 1.0, 1.0),
                 instance_map="eye", intensional_vectors=None, freeze_intensional=False,
                 learning_rate=0.001, batch_size=4096 * 8, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None):
        if isinstance(intensional_vectors, str):
            intensional_vectors = self.load_intensional_vectors(intensional_vectors)
        if intensional_vectors:
            dims = {np.asarray(v).shape[-1] for v in intensional_vectors.values()}
            if dims != {embed_dim}:
                raise ValueError(f"intensional_vectors have dimension(s) {sorted(dims)} but "
                                 f"embed_dim is {embed_dim}.")
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis)
        self.alpha = alpha
        self.margins = tuple(margins)
        self.instance_map = instance_map
        self.intensional_vectors = intensional_vectors
        self.freeze_intensional = freeze_intensional
        self.init_module()

    # -- optional language-model input ---------------------------------------------------
    def class_texts(self, descriptions=None):
        """Text per class IRI (the paper: name, or name followed by description)."""
        descriptions = descriptions or {}
        out = {}
        for iri in self.named_class_index_dict:
            text = name_to_text(iri)
            if iri in descriptions:
                text = f"{text} {descriptions[iri]}"
            out[iri] = text
        return out

    @staticmethod
    def encode_class_texts(texts, encoder):
        """Vectors for ``texts`` (dict IRI -> text) with ``encoder``, any callable that maps a
        list of strings to an array ``(n, d)``, for example
        ``SentenceTransformer("all-MiniLM-L6-v2").encode``. The language model is the
        caller's choice; mOWL does not import or download one."""
        iris = list(texts)
        vectors = np.asarray(encoder([texts[i] for i in iris]))
        return {i: v for i, v in zip(iris, vectors)}

    @staticmethod
    def pca_vectors(vectors, dim):
        """Reduce vectors (dict IRI -> vector) to ``dim`` dimensions with a principal component
        projection, for language-model vectors that are larger than ``embed_dim``. The
        paper's code instead trains a linear map from the (frozen) language-model vectors to
        the embedding space; mOWL's intensional vectors are the embedding itself."""
        iris = list(vectors)
        x = np.asarray([vectors[i] for i in iris], dtype="float64")
        x = x - x.mean(axis=0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        k = min(dim, vt.shape[0])
        y = x @ vt[:k].T
        if k < dim:
            y = np.pad(y, ((0, 0), (0, dim - k)))
        return {i: v.astype("float32") for i, v in zip(iris, y)}

    @staticmethod
    def save_intensional_vectors(path, vectors):
        """Write a dict class IRI -> vector to ``path`` (an ``.npz`` file with the arrays
        ``iris`` and ``vectors``), the form ``intensional_vectors=path`` reads."""
        iris = list(vectors)
        np.savez(path, iris=np.asarray(iris),
                 vectors=np.asarray([vectors[i] for i in iris], dtype="float32"))

    @staticmethod
    def load_intensional_vectors(path):
        """Read what :meth:`save_intensional_vectors` wrote."""
        with np.load(path, allow_pickle=False) as data:
            return {str(i): v for i, v in zip(data["iris"], data["vectors"])}

    def init_module(self):
        init = None
        if self.intensional_vectors:
            init = np.full((len(self.class_index_dict), self.embed_dim), np.nan, "float32")
            for iri, vec in self.intensional_vectors.items():
                if iri in self.class_index_dict:
                    init[self.class_index_dict[iri]] = np.asarray(vec, dtype="float32")
        self.module = EIKEModule(
            len(self.class_index_dict), len(self.object_property_index_dict),
            len(self.individual_index_dict), embed_dim=self.embed_dim, alpha=self.alpha,
            margins=self.margins, instance_map=self.instance_map, intensional_init=init,
            freeze_intensional=self.freeze_intensional).to(self.device)

    # -- training ------------------------------------------------------------------------
    @property
    def training_datasets(self):
        """The training datasets EIKE uses: SubClassOf, InstanceOf and relational triples."""
        return {k: v for k, v in super().training_datasets.items() if k in self._EIKE_FORMS}

    def get_negative_sampling_config(self):
        """Bordes-style negatives (the paper): corrupt head or tail, one negative of each
        kind per positive."""
        cfg = {
            "gci0": {"index_pool": "classes", "corrupt_column": [0, 1]},
            "class_assertion": {"index_pool": ["individuals", "classes"],
                                "corrupt_column": [0, 1]},
            "object_property_assertion": {"index_pool": "individuals",
                                          "corrupt_column": [0, 2]},
        }
        if self.neg_sampling_gcis is not None:
            cfg = {k: v for k, v in cfg.items() if k in self.neg_sampling_gcis}
        return cfg

    def compute_loss(self, pos_scores, neg_scores=None):
        """The ranking loss :math:`[\\gamma + f(\\delta) - f(\\delta')]_+` of Section 4.3, where
        :math:`\\delta'` is a corruption of :math:`\\delta` (Bordes et al., as the paper
        prescribes): each positive is paired with its own negatives, the blocks of
        ``neg_scores`` follow the corrupted columns. The module returns negative scores
        already reduced by the margin of their axiom type. Averaged over pairs."""
        if neg_scores is None:
            return th.mean(pos_scores)
        k = neg_scores.numel() // pos_scores.numel()
        return th.relu(pos_scores.view(1, -1) - neg_scores.view(k, -1)).mean()
