"""A model put together from parts."""

import inspect

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.elmodule import ELModule
from mowl.representation import methods as M
from mowl.representation.objectives import Objective
from mowl.representation.swap import NO_PART, build_method, swap
from mowl.training.losses import ranking_hinge_loss


#: Where the methods that are not trained on axioms are trained.
_TRAINERS = {"queries": "mowl.models.query",
             "sequences": "mowl.models.ComposedSequenceModel",
             "expressions": "mowl.models.ComposedALC (mowl.models.ComposedSubspace for 'e2r')",
             "construction": "mowl.models.constructed.ConstructedModel"}


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
        ``gci3="inclusion"``). Every part argument below that is given together with
        ``method`` *replaces* that part of the method
        (:func:`mowl.representation.swap.build_method`); a replacement that cannot exist
        raises :class:`~mowl.representation.ConstructorUnavailable` naming both sides, an
        argument the method cannot take raises ``TypeError``.
    :param classes: Name of a class model (:data:`mowl.representation.CLASS_MODELS`), a
        function ``(size, embed_dim)`` returning one, or a class model (taken as it is).
    :param relations: Name of a relation model
        (:data:`mowl.representation.RELATION_MODELS`), such a function or a relation model;
        ``"none"``: no relation model.
    :param individuals: As ``classes``, for the individuals (``"none"``: none). Default: of
        the kind of the classes.
    :param measure: Name of a measure (:data:`mowl.representation.MEASURES`), a measure
        object, or ``None`` for the default of the class model (of the method).
    :param objective: An :class:`~mowl.representation.Objective`; default
        ``Objective(margin)`` (the objective of the method).
    :param objective_options: With ``method``: options of the method's own objective that
        are replaced (``{"gci3": "inclusion"}``, see
        :meth:`~mowl.representation.Objective.replace`).
    :param margin: Margin of the default objective, or of ``method``.
    :param regularizers: Penalties (:class:`mowl.training.Regularizer`) of the module; with
        ``method`` they are added to those of the method.

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

    def __init__(self, dataset, method=None, classes=None, relations=None,
                 measure=None, objective=None, embed_dim=50, margin=0.0, regularizers=(),
                 individuals=None, objective_options=None, learning_rate=0.001, batch_size=4096 * 8, model_filepath=None,
                 device="cpu", neg_sampling_gcis=None, **method_options):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis)
        if method is not None and method not in M.METHODS:
            raise ValueError(f"Unknown method {method!r}; available: {sorted(M.METHODS)}.")
        if method is not None and M.METHOD_KINDS.get(method, "axioms") != "axioms":
            kind = M.METHOD_KINDS[method]
            raise TypeError(
                f"The method {method!r} is trained on {kind}, not on "
                f"normal-form axioms, so ComposedEL cannot train it. Methods that ComposedEL "
                f"trains: "
                f"{sorted(k for k in M.METHODS if M.METHOD_KINDS.get(k) == 'axioms')}; "
                f"for {method!r} see "
                f"{_TRAINERS.get(kind, 'mowl.models.query')}.")
        if method is None and method_options:
            raise TypeError(f"Unexpected arguments {sorted(method_options)}.")
        if method is None and objective_options:
            raise TypeError("objective_options are options of the objective of a method; "
                            "without one, pass objective=Objective(...).")
        if method is None:
            classes = "box" if classes is None else classes
            relations = "translation" if relations is None else relations
        self.method = method
        self.classes, self.relations, self.measure = classes, relations, measure
        self.individuals, self.objective_options = individuals, objective_options
        # (kept under a name of its own: a named method may use ``objective`` for an option
        # of its own, as PrismEL does for its reading)
        self._objective_part = objective
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
            module = build_method(
                self.method, *sizes, embed_dim=self.embed_dim, classes=self.classes,
                relations=self.relations, individuals=self.individuals,
                measure=self.measure, objective=self._objective_part,
                objective_options=self.objective_options, regularizers=self.regularizers,
                **options)
        else:
            relations = None if (isinstance(self.relations, str)
                                 and self.relations == NO_PART) else self.relations
            objective = self._objective_part if self._objective_part is not None \
                else Objective(self.margin)
            module = M.compose(*sizes, embed_dim=self.embed_dim, classes=self.classes,
                               relations=relations, measure=self.measure,
                               individuals=not (isinstance(self.individuals, str)
                                                and self.individuals == NO_PART),
                               objective=objective, regularizers=self.regularizers)
            if self.individuals is not None and not (
                    isinstance(self.individuals, str) and self.individuals == NO_PART):
                module = swap(module, "compose", sizes, self.embed_dim,
                              individuals=self.individuals)
        if not isinstance(module, ELModule):
            from mowl.representation.regions import ConstructorUnavailable
            trainer = getattr(module, "trained_by", None)
            raise ConstructorUnavailable(
                f"The geometry of {self.method!r} ({type(module).__name__}) is not an "
                f"ELModule: it is not trained on normal-form axioms, so ComposedEL cannot "
                f"train it." + (f" Train it, and swap its parts, with {trainer}."
                                if trainer else ""))
        self.module = module.to(self.device)
        # EL++ role axioms: loaded when the geometry reads them
        self.load_role_axioms = bool(getattr(module, "role_axiom_capable", False))
        self._attach_structural_tables()
        self._attach_default_regimes()
        if self.method is not None and self.swapped_parts:
            self._probe()

    @property
    def swapped_parts(self):
        """The part arguments given together with ``method`` (they replace the parts of the
        method)."""
        given = {"classes": self.classes, "relations": self.relations,
                 "individuals": self.individuals, "measure": self.measure,
                 "objective": self._objective_part, "objective_options": self.objective_options,
                 "regularizers": self.regularizers or None}
        return {k: v for k, v in given.items() if v is not None}

    def _probe(self):
        """Evaluate the losses of two axioms of every training form (and the penalties), so
        that a combination whose parts do not fit raises
        :class:`~mowl.representation.ConstructorUnavailable` when it is built, naming the
        method and the parts, not in the middle of training."""
        from mowl.representation.regions import ConstructorUnavailable
        module = self.module
        try:
            with th.no_grad():
                for form, data in self.training_datasets.items():
                    if len(data) == 0:
                        continue
                    rows = data[:2]
                    module(rows, form)
                    if form in getattr(module, "neg_capable_gcis", ()):
                        module(rows, form, neg=True)
                module.regularization_loss()
        except ConstructorUnavailable as error:
            raise ConstructorUnavailable(
                f"{self.method!r} cannot take {sorted(self.swapped_parts)}: {error}") from error
        except (TypeError, AttributeError, RuntimeError, ValueError, IndexError,
                NotImplementedError, KeyError) as error:
            raise ConstructorUnavailable(
                f"{self.method!r} cannot take {sorted(self.swapped_parts)}: the combination "
                f"{module.parts} does not evaluate ({type(error).__name__}: {error}).") from error

    def _attach_default_regimes(self):
        """The training regimes that belong to the method (``geometry.default_regimes``: a
        tuple of functions that make one regime each, for example the negatives of the paper
        or its loss), unless a regime is attached already."""
        factories = getattr(self.module, "default_regimes", ())
        if factories and self.training_regime is None:
            from mowl.training.regime import ComposedRegime
            regimes = [make() for make in factories]
            self.training_regime = regimes[0] if len(regimes) == 1 else ComposedRegime(*regimes)

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
            return ranking_hinge_loss(pos_scores, neg_scores)
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
