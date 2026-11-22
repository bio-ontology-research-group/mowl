"""Random walks on a projection of the ontology, encoded with Word2Vec."""

import logging

import torch as th
from deprecated.sphinx import versionchanged

import mowl.error.messages as msg
from mowl.base_models.graph_model import RandomWalkModel
from mowl.models.composed.sequence import ComposedSequenceModel

logger = logging.getLogger(__name__)


@versionchanged(version='1.0.0')
class RandomWalkPlusW2VModel(ComposedSequenceModel, RandomWalkModel):
    """
    Embedding model that combines graph projections + random walks.

    It is the combination of parts (:mod:`mowl.representation.sequence`) of the methods
    ``dl2vec``, ``owl2vec_star`` and ``taxonomy_walks``: the graph of a projector
    (:meth:`set_projector`), the walks of a walker (:meth:`set_walker`) on it, a Word2Vec
    encoder (:meth:`set_w2v_model`), the vector of every entity's token, the similarity
    :math:`1 - \\sigma(x \\cdot y)` (:attr:`evaluation_model`) and a pair classifier for the
    ranking of candidate axioms. Every part can be replaced through ``model.module``.

    A new walk replaces the walks of an earlier one (the walkers append to their file), and
    a Word2Vec model that has a vocabulary keeps its vectors when it is trained again
    (after :meth:`from_pretrained` or a first :meth:`train`); pass ``restart=True`` to
    :meth:`set_w2v_model` to start anew from the seed on every :meth:`train`.
    """

    # Word2Vec gives one vector per class; an asymmetric axiom score needs a scoring head
    # (mowl.evaluation.model_scoring.PairClassifierHead): the scorer of the geometry.
    axiom_scoring = "vectors"
    scored_forms = ("gci0", "gci2")

    def __init__(self, *args, **kwargs):
        from mowl.representation.library.walks_views import Walks
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        super().__init__(*args, source=Walks(), device=device, **kwargs)

    # -- parts --------------------------------------------------------------------------
    def set_projector(self, projector):
        """Sets the projector: the graph view of the walks
        (:class:`~mowl.representation.library.walks_views.ProjectionView`).

        :param projector: the projector to be set
        :type projector: :class:`Projection <mowl.projection.base.ProjectionModel>` object
        """
        from mowl.representation.library.walks_views import ProjectionView
        super().set_projector(projector)
        self.module.source.view = ProjectionView(projector)
        self._written = False

    def set_walker(self, walker):
        """Sets the walker of the walks.

        :param walker: the walker to be set
        :type walker: :class:`WalkingModel <mowl.walking.walking.WalkingModel>` object
        """
        super().set_walker(walker)
        self.module.source.walker = walker
        self._written = False

    def set_w2v_model(self, *args, **kwargs):
        """
        Sets the Word2Vec model to be used for training. The set model will be a :class:`gensim.models.word2vec.Word2Vec` object.

        :param args: Arguments to be passed to the :class:`Word2Vec <gensim.models.word2vec.Word2Vec>` constructor.
        :param kwargs: Keyword arguments to be passed to the :class:`Word2Vec <gensim.models.word2vec.Word2Vec>` constructor
            (and ``restart``, see :class:`~mowl.representation.library.walks_encoders.Word2VecEncoder`).
        """
        super().set_w2v_model(*args, **kwargs)

    # -- training -----------------------------------------------------------------------
    def _check_parts(self):
        if self.projector is None:
            raise AttributeError(msg.GRAPH_MODEL_PROJECTOR_NOT_SET)
        if self.walker is None:
            raise AttributeError(msg.RANDOM_WALK_MODEL_WALKER_NOT_SET)
        super()._check_parts()

    def _ensure_sequences(self):
        renew = not self._written or self.axioms_added
        super()._ensure_sequences()
        if renew:
            self._edges = self.module.source.edges

    def train(self, epochs=None):
        """
        Triggers the Word2Vec training process.

        :param epochs: Number of epochs to train the model. If None, the value of the epochs parameter passed to the constructor will be used.
        :type epochs: int
        """
        super().train(epochs)

    def add_axioms(self, *axioms):
        super().add_axioms(*axioms)
        self._edges = self.module.source.edges
