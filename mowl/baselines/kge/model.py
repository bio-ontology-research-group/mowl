# PyKEEN imports
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.models import ERModel
from pykeen.training import SLCWATrainingLoop

import tempfile
import torch as th
from torch.optim import Adam, Optimizer
import os
import mowl.error as err
import logging
logging.basicConfig(level=logging.INFO)


class KGEModel():

    '''
    :param triples_factory: PyKEEN triples factory.
    :type triples_factory: :class:`pykeen.triples.triples_factory.TriplesFactory`
    :param model: Initialized PyKEEN model
    :type model: Initialized model of the type :class:`EntityRelationEmbeddingModel \
    <pykeen.models.base.EntityRelationEmbeddingModel>` or \
    :class:`ERModel <pykeen.models.nbase.ERModel>`.
    :param epochs: Number of epochs.
    :type epochs: int
    :param batch_size: Number of each data samples in each batch. Defaults to 32.
    :type batch_size: int, optional
    :param optimizer: Optimizer to be used while training the model. Defaults to \
    :class:`torch.optim.Adam`.
    :type optimizer: subclass of :class:`torch.optim.Optimizer`, optional
    :param lr: Learning rate. Defaults to 1e-3.
    :type lr: float, optional
    :param device: Device to run the model. Defaults to `cpu`.
    :type device: str
    :param model_filepath: Path for saving the model. Defaults to \
    :class:`tempfile.NamedTemporaryFile`
    :type model_filepath: str, optional
    '''

    def __init__(self,
                 triples_factory,
                 pykeen_model,
                 epochs,
                 batch_size=32,
                 optimizer=Adam,
                 lr=1e-3,
                 device="cpu",
                 model_filepath=None,
                 ):

        if not isinstance(triples_factory, TriplesFactory):
            raise TypeError(
                "Parameter triples_factory must be of type or subtype of \
pykeen.triples.triples_factory.TriplesFactory.")
        if not isinstance(pykeen_model, ERModel):
            raise TypeError(
                "Parameter pykeen_model must be of type or subtype of pykeen.models.ERModel.")
        if not isinstance(epochs, int):
            raise TypeError("Parameter epochs must be of type int.")
        if not isinstance(batch_size, int):
            raise TypeError("Optional parameter batch_size must be of type int.")
        try:
            optimizer(params=[th.empty(1)])
        except Exception:
            raise TypeError(
                "Optional parameter optimizer must be a subtype of torch.optim.Optimizer.")
        if not isinstance(lr, float):
            raise TypeError("Optional parameter lr must be of type float.")
        if not isinstance(device, str):
            raise TypeError("Optional parameter device must be of type str.")
        if not isinstance(model_filepath, str) and model_filepath is not None:
            raise TypeError("Optional parameter model_filepath must be of type str or None.")

        self.triples_factory = triples_factory
        self.device = device
        self.model = pykeen_model.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr

        if model_filepath is None:
            model_filepath = tempfile.NamedTemporaryFile()
            model_filepath = model_filepath.name
        self.model_filepath = model_filepath

        self._trained = False
        self._data_loaded = False

        self._class_index_dict = None
        self._class_embeddings_dict = None
        self._object_property_index_dict = None
        self._object_property_embeddings_dict = None

    @property
    def class_index_dict(self):
        """This returns a dictionary of the form class_name -> class_index. This equivalent to \
the method triples_factory.entity_to_id from PyKEEN."""

        if self._class_index_dict is None:
            self._class_index_dict = self.triples_factory.entity_to_id
        return self._class_index_dict

    @property
    def object_property_index_dict(self):
        """This returns a dictionary of the form object_property_name -> object_property_index. \
This equivalent to the method triples_factory.relation_to_id from PyKEEN."""

        if self._object_property_index_dict is None:
            self._object_property_index_dict = self.triples_factory.relation_to_id
        return self._object_property_index_dict

    @property
    def class_embeddings_dict(self):
        if self._class_embeddings_dict is None:
            try:
                self._get_embeddings()
            except FileNotFoundError:
                raise AttributeError(err.EMBEDDINGS_NOT_FOUND_MODEL_NOT_TRAINED)
        return self._class_embeddings_dict

    @property
    def object_property_embeddings_dict(self):
        if self._object_property_embeddings_dict is None:
            try:
                self._get_embeddings()
            except FileNotFoundError:
                raise AttributeError(err.EMBEDDINGS_NOT_FOUND_MODEL_NOT_TRAINED)
        return self._object_property_embeddings_dict

    def load_best_model(self):
        if not os.path.exists(self.model_filepath):
            raise FileNotFoundError(
                "Loading best model failed because file was not found at the given path. \
Please train the model first.")
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

    def train(self):
        optimizer = self.optimizer(params=self.model.get_grad_params(), lr=self.lr)

        training_loop = SLCWATrainingLoop(model=self.model, triples_factory=self.triples_factory,
                                          optimizer=optimizer)

        _ = training_loop.train(triples_factory=self.triples_factory, num_epochs=self.epochs,
                                batch_size=self.batch_size)

        th.save(self.model.state_dict(), self.model_filepath)
        self._trained = True

    def _get_embeddings(self, load_best_model=True):

        if load_best_model:
            self.load_best_model()

        cls_embeddings = self.model.entity_representations[0](indices=None).cpu().detach().numpy()
        cls_ids = {item[0]: item[1] for item in self.triples_factory.entity_to_id.items()}
        cls_embeddings = {item[0]: cls_embeddings[item[1]] for item in
                          self.triples_factory.entity_to_id.items()}

        self._class_index_dict = cls_ids
        self._class_embeddings_dict = cls_embeddings

        rel_embeddings = self.model.relation_representations[0](indices=None)
        rel_embeddings = rel_embeddings.cpu().detach().numpy()
        rel_ids = {item[0]: item[1] for item in self.triples_factory.relation_to_id.items()}
        rel_embeddings = {item[0]: rel_embeddings[item[1]] for item in
                          self.triples_factory.relation_to_id.items()}

        self._object_property_index_dict = rel_ids
        self._object_property_embeddings_dict = rel_embeddings

    @th.no_grad()
    def score_method_point(self, point):
        """Receives the embedding of a point and returns its score."""
        self.model.eval()
        # TODO implement code that checks dimensionality
        point = self.point_to_tensor(point)

        return self.model.predict_hrt(point)

    @th.no_grad()
    def score_method_tensor(self, data):
        self.model.eval()
        return self.model.predict_hrt(data)

    def point_to_tensor(self, point):
        point = [list(point)]
        point = th.tensor(point).to(self.device)
        return point
