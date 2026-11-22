"""A model put together from parts."""

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.representation import methods as M
from mowl.representation.objectives import Objective


class ComposedEL(EmbeddingELModel):
    """An :math:`\\mathcal{EL}` embedding model put together from parts: a class model, a
    relation model, a measure and an objective (:mod:`mowl.representation`).

    Either name a published combination,

    .. code-block:: python

       model = ComposedEL(dataset, method="elbe", margin=0.05)

    or choose the parts:

    .. code-block:: python

       model = ComposedEL(dataset, classes="ball", relations="translation",
                          objective=Objective(margin=0.05))

    The default objective is the semantic one with a strict margin, so that a zero loss
    implies the axiom; the objectives of the publications are available through
    ``method`` or the constructors of :class:`~mowl.representation.Objective`.

    :param method: Key of :data:`mowl.representation.METHODS`. The parts are then those of
        the method and ``method_options`` are passed to its function (for example
        ``gci3="inclusion"``).
    :param classes: Name of a class model (:data:`mowl.representation.CLASS_MODELS`) or a
        function ``(size, embed_dim)`` returning one.
    :param relations: Name of a relation model
        (:data:`mowl.representation.RELATION_MODELS`), such a function, or ``None``.
    :param measure: Name of a measure (:data:`mowl.representation.MEASURES`), a measure
        object, or ``None`` for the default of the class model.
    :param objective: An :class:`~mowl.representation.Objective`; default
        ``Objective(margin)``.
    :param margin: Margin of the default objective, or of ``method``.
    :param regularizers: Penalties (:class:`mowl.training.Regularizer`) of the module.
    """

    def __init__(self, dataset, method=None, classes="box", relations="translation",
                 measure=None, objective=None, embed_dim=50, margin=0.0, regularizers=(),
                 learning_rate=0.001, batch_size=4096 * 8, model_filepath=None,
                 device="cpu", neg_sampling_gcis=None, **method_options):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis)
        if method is not None and method not in M.METHODS:
            raise ValueError(f"Unknown method {method!r}; available: {sorted(M.METHODS)}.")
        if method is None and method_options:
            raise TypeError(f"Unexpected arguments {sorted(method_options)}.")
        self.method = method
        self.classes, self.relations, self.measure = classes, relations, measure
        self.objective = objective
        self.margin = margin
        self.regularizers = tuple(regularizers)
        self.method_options = method_options
        self.init_module()

    def init_module(self):
        sizes = (len(self.class_index_dict), len(self.object_property_index_dict),
                 len(self.individual_index_dict))
        if self.method is not None:
            module = M.METHODS[self.method](*sizes, embed_dim=self.embed_dim,
                                            margin=self.margin, **self.method_options)
            if self.regularizers:
                module.regularizers = module.regularizers + self.regularizers
        else:
            objective = self.objective if self.objective is not None \
                else Objective(self.margin)
            module = M.compose(*sizes, embed_dim=self.embed_dim, classes=self.classes,
                               relations=self.relations, measure=self.measure,
                               objective=objective, regularizers=self.regularizers)
        self.module = module.to(self.device)

    @property
    def geometry(self):
        """The module, which is a :class:`~mowl.representation.Geometry`."""
        return self.module
