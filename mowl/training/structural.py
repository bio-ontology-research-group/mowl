"""Regimes of geometries that read the structure of the ontology or supervise their regions by
points (the structural and the point readings of MultiBoxEL and MultiELBE; any geometry with
the same hooks).

:class:`StructuralTables`
    When training starts, the definition table and the existence (and witness) targets are
    compiled from the training data of the model and installed in the geometry
    (``set_structural_tables``): defined classes become the intersections of their definers,
    the existence and witness penalties get their targets.
:class:`ExtraPenalties`
    Adds, at every step, the penalties of the geometry that are not part of its
    regularisation loss (``geometry.extra_penalties()``: for the structural reading the
    existence, size and witness penalties).
:class:`SampledPointLoss`
    Adds, at every step, the loss of the point terms of the geometry (``point_terms``,
    :class:`~mowl.representation.library.structural_points.PointTerms`) on a sample of the
    point supervision of the model, drawn with the seed ``seed + epoch``; the volume penalty
    covers the named classes other than ``owl:Thing`` and ``owl:Nothing``.

The three are separate: a geometry with definitions but no penalties takes the first only, a
point reading the third only.
"""

import numpy as np
import torch as th

from mowl.training.regime import TrainingRegime

__all__ = ["StructuralTables", "ExtraPenalties", "SampledPointLoss", "MethodRegimes"]

_TOP = "http://www.w3.org/2002/07/owl#Thing"
_BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"


class StructuralTables(TrainingRegime):
    """Compile the structural tables of the training data (definitions
    :math:`A \\equiv X \\sqcap Y`, satisfiable classes and pairs, witness targets;
    :func:`~mowl.nn.el.structural.build_structural_tables`) at the start of every call of
    ``train`` and install them in the geometry. The model keeps them
    (``model._structural_tables``), so that a module built again or loaded gets them too.

    :param options: ``model -> dict`` of the options of ``build_structural_tables`` (``k``,
        ``max_components``, ``enabled``, ``cm``); default: the geometry's
        ``structural_table_options`` and the model's ``canonical_model``.
    """

    key = "structural-tables"

    def __init__(self, options=None):
        super().__init__()
        self.options = options

    def attach(self, model):
        if not callable(getattr(model.module, "set_structural_tables", None)):
            raise TypeError(f"StructuralTables needs a geometry with set_structural_tables; "
                            f"{type(model.module).__name__} has none.")
        super().attach(model)

    def _options(self):
        if self.options is not None:
            return dict(self.options(self.model))
        module = self.model.module
        options = dict(module.structural_table_options)
        options["cm"] = getattr(self.model, "canonical_model", None)
        return options

    def training_datasets(self, datasets):
        from mowl.nn.el.structural import build_structural_tables
        tables = build_structural_tables(self.model, **self._options())
        self.model._structural_tables = tables
        table, targets, top_idx = tables
        self.model.module.set_structural_tables(table, targets, top_idx=top_idx)
        return datasets


class ExtraPenalties(TrainingRegime):
    """Add the penalties of the geometry that its regularisation loss does not count
    (``geometry.extra_penalties()``, :class:`~mowl.training.Regularizer` objects) to every
    step."""

    key = "extra-penalties"

    def attach(self, model):
        if not callable(getattr(model.module, "extra_penalties", None)):
            raise TypeError(f"ExtraPenalties needs a geometry with extra_penalties(); "
                            f"{type(model.module).__name__} has none.")
        super().attach(model)

    def additional_loss(self, epoch):
        module = self.model.module
        total = th.zeros((), device=self.model.device)
        for penalty in module.extra_penalties():
            term = penalty.penalty(module)
            if term is not None:
                total = total + term
        return total


class SampledPointLoss(TrainingRegime):
    """Add the loss of the point terms of the geometry on a sample of the point supervision.

    Every step draws at most ``points_per_epoch`` pairs and role triples from the model's
    :class:`~mowl.datasets.el.points.PointSupervision` (``model._ensure_point_supervision()``)
    with the generator seeded ``seed + epoch`` and adds ``geometry.point_terms.loss``:
    membership, role and the volume penalty (weight ``volume_reg`` of the terms) of the named
    classes other than ``owl:Thing`` and ``owl:Nothing``.

    :param points_per_epoch: Size of a sample.
    :param seed: Base seed (``points_seed`` of the model).
    """

    key = "sampled-point-loss"

    def __init__(self, points_per_epoch=100_000, seed=0):
        super().__init__()
        self.points_per_epoch, self.seed = int(points_per_epoch), int(seed)

    def attach(self, model):
        if getattr(model.module, "point_terms", None) is None \
                or not callable(getattr(model, "_ensure_point_supervision", None)):
            raise TypeError(f"SampledPointLoss needs a geometry with point terms and a model "
                            f"with point supervision; {type(model).__name__} has none.")
        super().attach(model)

    def volume_classes(self):
        return [int(i) for iri, i in self.model.named_class_index_dict.items()
                if iri not in (_TOP, _BOTTOM)]

    def additional_loss(self, epoch):
        supervision = self.model._ensure_point_supervision()
        batch = supervision.sample(self.points_per_epoch,
                                   rng=np.random.default_rng(self.seed + int(epoch)))
        module = self.model.module
        terms = module.point_terms
        terms.volume_classes = self.volume_classes()
        return terms.loss(module, batch)


class MethodRegimes:
    """Mixin for a named model whose method needs regimes (the structural tables, the point
    loss): ``method_regimes()`` returns them (a subclass defines it), they are attached when
    the module is made, and a regime assigned to ``training_regime`` is *added* to those of
    the method rather than replacing them; a regime of the same class as one of the method's
    replaces that one. ``training_regime = None`` removes them all."""

    def method_regimes(self):
        return ()

    def _attach_method_regimes(self):
        if getattr(self, "_training_regime", None) is None:
            regimes = list(self.method_regimes())
            if regimes:
                self._set_regime(regimes)

    def _set_regime(self, regimes):
        from mowl.training.regime import ComposedRegime
        regime = regimes[0] if len(regimes) == 1 else ComposedRegime(*regimes)
        _BASE_SETTER(self, regime)

    @property
    def training_regime(self):
        return getattr(self, "_training_regime", None)

    @training_regime.setter
    def training_regime(self, regime):
        if regime is None:
            _BASE_SETTER(self, None)
            return
        given = list(getattr(regime, "regimes", (regime,)))
        kinds = {type(r) for r in given}
        own = [r for r in self.method_regimes() if type(r) not in kinds]
        self._set_regime(own + given)


def _BASE_SETTER(model, regime):
    from mowl.base_models.elmodel import EmbeddingELModel
    EmbeddingELModel.training_regime.fset(model, regime)
