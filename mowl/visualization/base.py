from sklearn.manifold import TSNE as SKTSNE
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import logging
import warnings
logging.basicConfig(level=logging.INFO)


class Visualizer():

    def __init_(self):
        return

    def show(self):
        raise NotImplementedError()

    def savefig(self, outfile):
        raise NotImplementedError()


class TSNE(Visualizer):
    """
    Wrapper for :class:`sklearn.manifold.TSNE`

    :param embeddings: Embeddings dictionary
    :type embeddings: dict or :class:`gensim.models.keyedvectors.KeyedVectors`
    :param labels: Dictionary containing label information of the entities
    :type labels: dict of {str: str}
    :param entities: List of entities to consider for computing the TSNE. If `None`, then all \
        the entitites in the embeddings dictionary will be considered.
    :type entities: list of str
    """

    def __init__(self, embeddings, labels, entities=None):

        self.total_embeddings = len(embeddings)
        self.labels = labels
        self.embeddings = dict()

        self.not_to_process = 0
        if isinstance(embeddings, KeyedVectors):
            for idx, word in enumerate(embeddings.index_to_key):
                if (entities is not None) and (word not in entities):
                    self.not_to_process += 1
                    continue
                if word not in self.labels:
                    self.not_to_process += 1
                    continue
                self.embeddings[word] = embeddings[word]
        elif isinstance(embeddings, dict):
            if entities is None:
                self.embeddings = {name: emb for name, emb in embeddings.items()
                                   if name in self.labels}
            else:
                self.embeddings = {name: emb for name, emb in embeddings.items()
                                   if name in entities and name in self.labels}
        else:
            raise TypeError("Embeddings type {type(embeddings)} not recognized. Expected types \
                are dict or gensim.models.keyedvectors.KeyedVectors")

        logging.info("Found %d embedding vectors. Processing only %d.", self.total_embeddings,
                     len(self.embeddings))
        self.embedding_idx_dict = {v: k for k, v in enumerate(self.embeddings.keys())}

        self.classes = set(self.labels.values())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.classes)))
        self.class_color_dict = {cl: col for cl, col in zip(self.classes, colors)}

    def generate_points(self, epochs, workers=1, verbose=0):
        """This method will call the :meth:`sklearn.manifold.TSNE.fit_transform`
        method to generate the points for the plot.

        :param epochs: Number of epochs to run the TSNE algorithm
        :type epochs: int
        :param workers: Number of workers to use for parallel processing. Defaults to 1.
        :type workers: int, optional
        :param verbose: Verbosity level. Defaults to 0.
        """
        points = np.array(list(self.embeddings.values()))
        if np.iscomplexobj(points):
            if verbose:
                warnings.warn("Complex numpy array detected. Only real part will be considered",
                              UserWarning)
            points = points.real
        self.points = SKTSNE(n_components=2, verbose=verbose, n_iter=epochs, n_jobs=workers)
        self.points = self.points.fit_transform(points)
        self.plot_data = {}

        for name, idx in self.embedding_idx_dict.items():
            label = self.labels[name]
            x, y = tuple(self.points[idx])

            if label not in self.plot_data:
                self.plot_data[label] = [], []
            self.plot_data[label][0].append(x)
            self.plot_data[label][1].append(y)

    def show(self):
        """ This method will call the :meth:`matplotlib.pyplot.show` method to show the plot.
        """

        fig, ax = plt.subplots(figsize=(20, 20))

        for label, (xs, ys) in self.plot_data.items():
            color = self.class_color_dict[label]
            ax.scatter(xs, ys, color=color, label=label)

            ax.legend()
            ax.grid(True)

        plt.show()

    def savefig(self, outfile):
        """ This method will call the :meth:`matplotlib.pyplot.savefig` method to save the plot.
        :param outfile: Path to the output file
        :type outfile: str
        """

        fig, ax = plt.subplots(figsize=(20, 20))

        for label, (xs, ys) in self.plot_data.items():
            color = self.class_color_dict[label]
            ax.scatter(xs, ys, color=color, label=label)

            ax.legend()
            ax.grid(True)

        plt.savefig(outfile)
        plt.close()
