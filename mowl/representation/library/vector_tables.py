"""Tables of named vectors (class IRI -> vector): the form in which vectors computed outside
mOWL (language-model encodings of class names, pre-trained embeddings) enter a model.

* :func:`pca_reduce` reduces the vectors to a given dimension with a principal component
  projection.
* :func:`save_vector_table` and :func:`load_vector_table` write and read the table as an
  ``.npz`` file with the arrays ``iris`` and ``vectors``.

Used by: the intensional vectors of EIKE; any model that is initialised from vectors per name.
"""
import numpy as np

__all__ = ["pca_reduce", "save_vector_table", "load_vector_table"]


def pca_reduce(vectors, dim):
    """Reduce a dict name -> vector to ``dim`` dimensions with a principal component
    projection (centred; padded with zeros if there are fewer components than ``dim``).

    :return: dict name -> ``float32`` vector of length ``dim``.
    """
    names = list(vectors)
    x = np.asarray([vectors[i] for i in names], dtype="float64")
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    k = min(dim, vt.shape[0])
    y = x @ vt[:k].T
    if k < dim:
        y = np.pad(y, ((0, 0), (0, dim - k)))
    return {i: v.astype("float32") for i, v in zip(names, y)}


def save_vector_table(path, vectors):
    """Write a dict name -> vector to ``path`` (``.npz`` with ``iris`` and ``vectors``)."""
    names = list(vectors)
    np.savez(path, iris=np.asarray(names),
             vectors=np.asarray([vectors[i] for i in names], dtype="float32"))


def load_vector_table(path):
    """Read what :func:`save_vector_table` wrote: dict name -> vector."""
    with np.load(path, allow_pickle=False) as data:
        return {str(i): v for i, v in zip(data["iris"], data["vectors"])}
