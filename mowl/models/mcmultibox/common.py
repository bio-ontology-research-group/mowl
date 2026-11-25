"""What the volume-based MultiBoxEL model class and its direct implementation share: the
options, the datasets that are trained on, the order of the rows, and the evaluation distance
the module computes.

Nothing here computes a loss: the class users get (:class:`mowl.models.VolumeMultiBoxEL`)
builds its module from parts; the direct implementation in :mod:`mowl.baselines` has its own
module.
"""
import logging


from mowl.base_models.elmodel import EmbeddingELModel

logger = logging.getLogger("mowl.models.mcmultibox.model")

__all__ = ["VolumeMultiBoxELOptions"]


class VolumeMultiBoxELOptions(EmbeddingELModel):
    """The model of the volume-based MultiBoxEL without its module and its training loop: the
    constructor (the options of :class:`mowl.models.VolumeMultiBoxEL`, documented there), the
    trainable datasets and the evaluation. A subclass names the module class
    (:attr:`module_class`)."""

    #: The module class: built in :meth:`init_module` with the options of the model.
    module_class = None

    #: Normal forms with a loss in the thesis.
    TRAINABLE_GCIS = ("gci0", "gci1", "gci2", "gci3", "role_inclusion", "role_chain")

    def __init__(self, dataset, embed_dim=5, num_boxes=5, bound=1.0, num_points=20000,
                 temperature=1.0, aggregate="min", learning_rate=0.01, num_batches=5,
                 seed=0, point_chunk_size=2048, eval_points_per_box=32,
                 model_filepath=None, device="cpu", load_role_axioms=False,
                 max_chunk_elements=2 ** 25):
        super().__init__(dataset, embed_dim, 4096, extended=True, model_filepath=model_filepath,
                         device=device, learning_rate=learning_rate,
                         load_role_axioms=load_role_axioms)
        if num_batches < 1:
            raise ValueError("num_batches must be at least 1.")
        self.num_boxes = num_boxes
        self.bound = bound
        self.num_points = num_points
        self.temperature = temperature
        self.aggregate = aggregate
        self.num_batches = num_batches
        self.seed = seed
        self.point_chunk_size = point_chunk_size
        self.eval_points_per_box = eval_points_per_box
        self.max_chunk_elements = max_chunk_elements
        #: Mean training loss of each epoch (sum over normal forms), filled by :meth:`train`.
        self.training_losses = []
        self.init_module()

    def init_module(self):
        self.module = self.module_class(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            num_boxes=self.num_boxes,
            bound=self.bound,
            num_points=self.num_points,
            temperature=self.temperature,
            aggregate=self.aggregate,
            seed=self.seed,
            point_chunk_size=self.point_chunk_size,
            eval_points_per_box=self.eval_points_per_box,
            max_chunk_elements=self.max_chunk_elements,
        ).to(self.device)

    def _trainable_datasets(self):
        usable = {}
        skipped = []
        for name, ds in self.training_datasets.items():
            if len(ds) == 0:
                continue
            if name in self.TRAINABLE_GCIS:
                usable[name] = ds
            else:
                skipped.append(name)
        if skipped:
            logger.warning("VolumeMultiBoxEL: the thesis defines no loss for %s; these "
                           "axioms are not used for training.", skipped)
        return usable

    def eval_method(self, data):
        """Evaluation distance of thesis Section 7.1.2 for the GCI type ``eval_gci_name``."""
        if self._eval_gci_name is None:
            raise ValueError("eval_gci_name must be set before evaluation.")
        return self.module.distance(data, self._eval_gci_name)

    @property
    def evaluation_model(self):
        """Callable ``(data, gci_name) -> distances`` for the mOWL evaluators (lower is better)."""
        if self._evaluation_model is not None:
            return self._evaluation_model
        module = self.module

        def model(data, gci_name, neg=False):
            return module.distance(data, gci_name)

        return model

    @evaluation_model.setter
    def evaluation_model(self, value):
        self._evaluation_model = value
