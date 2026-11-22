"""A model put together from parts."""

import inspect

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.elmodule import ELModule
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

    The model trains on the axiom forms its geometry can read
    (:attr:`~mowl.representation.ComposedGeometry.supported_forms`): the others are left
    out of :attr:`training_datasets`, so a hierarchy method trains on the subclass axioms
    only. The EL++ role axioms of the dataset are loaded when the geometry reads them (its
    relation model defines ``role_inclusion`` and ``role_chain``). A form whose objective
    has a ranking loss (``Objective(ranking={form: gamma})``) is trained with it
    :math:`[\\mathrm{positive} - \\mathrm{negative}]_+`; after each optimisation step the
    parameters are projected back onto their constraint set (``module.project()``, for
    example the unit cube). Methods that are trained on queries and not on normal-form
    axioms (``"acone"``, ``"query2box"``, ...) are refused with an error that says so.
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
        if method is not None and M.METHOD_KINDS.get(method, "axioms") != "axioms":
            raise TypeError(
                f"The method {method!r} is trained on {M.METHOD_KINDS[method]}, not on "
                f"normal-form axioms, so ComposedEL cannot train it. Methods that ComposedEL "
                f"trains: "
                f"{sorted(k for k in M.METHODS if M.METHOD_KINDS.get(k) == 'axioms')}; "
                f"for {method!r} see mowl.models.query.")
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
            build = M.METHODS[self.method]
            options = dict(self.method_options)
            if "margin" in inspect.signature(build).parameters:
                options["margin"] = self.margin
            module = build(*sizes, embed_dim=self.embed_dim, **options)
            if self.regularizers:
                module.regularizers = module.regularizers + self.regularizers
        else:
            objective = self.objective if self.objective is not None \
                else Objective(self.margin)
            module = M.compose(*sizes, embed_dim=self.embed_dim, classes=self.classes,
                               relations=self.relations, measure=self.measure,
                               objective=objective, regularizers=self.regularizers)
        if not isinstance(module, ELModule):
            raise TypeError(
                f"The geometry of {self.method!r} ({type(module).__name__}) is not an "
                f"ELModule: it is not trained on normal-form axioms, so ComposedEL cannot "
                f"train it.")
        self.module = module.to(self.device)
        # EL++ role axioms: loaded when the geometry reads them
        self.load_role_axioms = bool(getattr(module, "role_axiom_capable", False))
        self._attach_structural_tables()

    def _attach_structural_tables(self):
        """A geometry that reads the structure of the ontology (definitions of classes,
        classes that must not be empty: the structural reading of MultiBoxEL and of the
        prism, ``geometry.wants_structural_tables``) gets its tables from the dataset of the
        model, once the module exists; they are kept, so that a module that is built again
        (:meth:`init_module`) gets them too."""
        module = self.module
        if not getattr(module, "wants_structural_tables", False):
            self._structural_tables = None
            return
        tables = getattr(self, "_structural_tables", None)
        if tables is None:
            from mowl.nn.el.structural import build_structural_tables
            tables = build_structural_tables(self, **module.structural_table_options)
            self._structural_tables = tables
        table, targets, top_idx = tables
        module.set_structural_tables(table, targets, top_idx)

    @property
    def reseed_components(self):
        """The hook of the regime :class:`~mowl.training.structure.IdleComponentReseeding`:
        the geometry's own ``reseed_components(epoch)`` (relocating components that cover
        nothing: a structural union of boxes has it). A model whose geometry has none does
        not have the attribute, and the regime says so."""
        hook = getattr(self.module, "reseed_components", None)
        if hook is None:
            raise AttributeError(
                f"The geometry {type(self.module).__name__} has no 'reseed_components'.")
        return hook

    @property
    def training_datasets(self):
        """The training datasets of the forms the geometry can read."""
        datasets = super().training_datasets
        supported = getattr(self.module, "supported_forms", None)
        if supported is None:
            return datasets
        return {form: data for form, data in datasets.items() if form in supported}

    def compute_loss(self, pos_scores, neg_scores=None):
        """The mean of the positive and the negative losses; for a form with a ranking loss
        (``Objective.ranking``) :math:`[\\mathrm{pos} - \\mathrm{neg}]_+`, each positive
        against each of its corrupted axioms."""
        ranking = getattr(getattr(self.module, "objective", None), "ranking", None)
        form = getattr(self.module, "last_form", None)
        if ranking and neg_scores is not None and form in ranking:
            pos = pos_scores.reshape(-1)
            neg = neg_scores.reshape(-1)
            if neg.numel() % pos.numel():
                raise ValueError("The ranking loss needs a whole number of negatives per "
                                 "positive.")
            neg = neg.reshape(-1, pos.numel())
            return th.relu(pos.unsqueeze(0) - neg).mean()
        return super().compute_loss(pos_scores, neg_scores)

    def after_optimizer_step(self):
        super().after_optimizer_step()
        project = getattr(self.module, "project", None)
        if project is not None:
            with th.no_grad():
                project()

    @property
    def as_geometry(self):
        """The module, which is a :class:`~mowl.representation.Geometry`."""
        return self.module
