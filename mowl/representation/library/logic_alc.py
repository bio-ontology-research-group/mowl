"""Geometries trained on :math:`\\mathcal{ALC}` axioms: class expressions of any depth.

The normal-form losses of :class:`~mowl.representation.ComposedGeometry` read an axiom
through its columns (``gci0`` ... ``object_property_assertion``). The ALC datasets
(:class:`~mowl.datasets.alc.ALCDataset`) give axioms of any shape instead: the axiom is a
*pattern* (a ``SubClassOf``, ``EquivalentClasses``, ``DisjointClasses``, class assertion or
role assertion over class expressions with placeholder names) and a batch of rows that fill
the placeholders. This module adds the two parts that read such axioms:

* :class:`ExpressionReading`: an objective mixin. It evaluates the class expressions of a
  pattern on the geometry (:func:`~mowl.representation.library.logic_expressions.pattern_region`)
  and has the measure compare the regions, so that it works for every geometry that has the
  constructors the pattern needs;
* :class:`ExpressionGeometry`: a :class:`~mowl.representation.ComposedGeometry` whose
  ``forward(axiom, x, e_emb)`` is that reading (and that still reads the normal forms:
  ``forward(rows, "gci0")``), together with the operator interface of
  :class:`~mowl.nn.alc.module.ALCModule` (``forward_fs``, ``_logical_and``...).

and a training hook, :class:`TrainingSchedule`, for methods whose loop changes the geometry
during training (a temperature that is annealed to zero).
"""

import torch as th

from mowl.nn.alc.module import ALCModule
from mowl.owlapi import (OWLClassAssertionAxiom, OWLDisjointClassesAxiom,
                         OWLEquivalentClassesAxiom, OWLObjectPropertyAssertionAxiom,
                         OWLSubClassOfAxiom)
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.logic_expressions import pattern_region

__all__ = ["ExpressionReading", "ExpressionGeometry", "TrainingSchedule", "Relaxation",
           "AnnealedRelaxation"]


class ExpressionReading:
    """The reading of ALC axioms (patterns of class expressions) as relations between
    regions. Mixed into an :class:`~mowl.representation.Objective`; the geometry is asked
    for nothing but its constructors and its measure, so the reading holds for every
    geometry.

    =======================================  =====================================
    :math:`C \\sqsubseteq D`                   ``inside(C, D)``
    :math:`C \\equiv D`                        ``inside(C, D) + inside(D, C)``
    :math:`C \\sqcap D \\sqsubseteq \\bot`        ``apart(C, D)``
    :math:`C(a)`                              ``member(a, C)``
    :math:`r(a, b)`                           ``member(a, exists r.{b})``
    =======================================  =====================================

    The cost is one number per row, shape ``(B,)``. The empty intersection of a conjunction
    inside :math:`C` of an inclusion adds its gap (zero for fuzzy sets).

    The pieces a method reads differently are methods of their own, so that a subclass
    changes one of them: :meth:`inclusion`, :meth:`equivalence`, :meth:`disjointness`,
    :meth:`class_assertion_pattern`, :meth:`role_assertion_pattern`.
    """

    def axiom_cost(self, g, axiom, x):
        """The cost of the rows ``x`` of the axiom pattern ``axiom``, shape ``(B,)`` (or a
        scalar for a reading that averages over the batch)."""
        if isinstance(axiom, OWLSubClassOfAxiom):
            return self.inclusion(g, axiom.getSubClass(), axiom.getSuperClass(), x)
        if isinstance(axiom, OWLEquivalentClassesAxiom):
            expressions = axiom.getClassExpressionsAsList()
            return self.equivalence(g, expressions[0], expressions[1], x)
        if isinstance(axiom, OWLDisjointClassesAxiom):
            expressions = axiom.getClassExpressionsAsList()
            return self.disjointness(g, expressions[0], expressions[1], x)
        if isinstance(axiom, OWLClassAssertionAxiom):
            return self.class_assertion_pattern(g, axiom.getClassExpression(), x)
        if isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            return self.role_assertion_pattern(g, x)
        raise NotImplementedError(f"Axiom type not supported: {axiom}")

    # -- the pieces ---------------------------------------------------------------------
    def regions(self, g, first, second, x):
        """The regions of two consecutive class expressions of the row (the second starts
        where the first ends) and the gap of the conjunctions of the first."""
        a, column, gap = pattern_region(g, first, x, 0, with_gap=True)
        b, _ = pattern_region(g, second, x, column)
        return a, b, gap

    def inclusion(self, g, sub, sup, x):
        a, b, gap = self.regions(g, sub, sup, x)
        cost = g.inside(a, b).reshape(len(x), -1).sum(-1)
        return cost if gap is None else cost + gap.reshape(len(x), -1).sum(-1)

    def equivalence(self, g, first, second, x):
        a, b, _ = self.regions(g, first, second, x)
        return (g.inside(a, b) + g.inside(b, a)).reshape(len(x), -1).sum(-1)

    def disjointness(self, g, first, second, x):
        a, b, _ = self.regions(g, first, second, x)
        return g.apart(a, b).reshape(len(x), -1).sum(-1)

    def class_assertion_pattern(self, g, expression, x):
        region, _ = pattern_region(g, expression, x, 1)
        return g.member(x[:, 0], region).reshape(len(x), -1).sum(-1)

    def role_assertion_pattern(self, g, x):
        """``x`` rows ``[object, role, subject]`` (the encoding of the ALC dataset)."""
        subject, role, obj = x[:, 2], x[:, 1], x[:, 0]
        target = g.exists(role, g.individuals(obj))
        return g.measure.member(g.individuals(subject), target,
                                self.m).reshape(len(x), -1).sum(-1)


class Relaxation:
    """The temperature of the smooth relaxation of the operators of a geometry, a number that
    its operator family and its measure read at every call (``source``), and that a
    :class:`TrainingSchedule` changes during training."""

    def __init__(self, temperature=0.0):
        self.temperature = float(temperature)


class TrainingSchedule:
    """What a training loop of an ALC model does around the optimiser step, so that the
    loop is a part (swappable) and not code of one method. The default does nothing.
    The geometry carries it as ``geometry.schedule``; :class:`~mowl.models.ComposedALC`
    calls the hooks."""

    def on_train_start(self, module, epochs):
        """Before the first epoch."""

    def on_epoch_start(self, module, epoch, epochs):
        """Before the loss of an epoch."""

    def after_step(self, module, epoch, epochs):
        """After the optimiser step of an epoch (inside the epoch, before validation)."""

    def on_validation(self, module):
        """Before a validation loss is computed (the exact operators are wanted)."""

    def on_train_end(self, module):
        """After the last epoch."""


class AnnealedRelaxation(TrainingSchedule):
    """The training loop of Al-cone: the operators are smoothed with a temperature that
    falls linearly from ``temperature`` to ``0`` over the epochs, and every parameter gets
    Gaussian noise of scale ``noise * temperature`` after each step (it vanishes at the end,
    and helps to leave local minima). Validation, and the geometry after training, use the
    exact operators (temperature ``0``).

    :param temperature: Initial temperature.
    :param noise: Scale of the noise.
    """

    def __init__(self, temperature=0.3, noise=0.3):
        self.temperature = temperature
        self.noise = noise
        self._tau0 = temperature

    def on_train_start(self, module, epochs):
        self._tau0 = self.temperature

    def on_epoch_start(self, module, epoch, epochs):
        module.temperature = self._tau0 * (1.0 - epoch / max(epochs, 1))

    def after_step(self, module, epoch, epochs):
        if self.noise > 0 and module.temperature > 0:
            with th.no_grad():
                for p in module.parameters():
                    p.add_(self.noise * module.temperature * th.randn_like(p))

    def on_validation(self, module):
        module.temperature = 0.0

    def on_train_end(self, module):
        module.temperature = 0.0


class ExpressionGeometry(ComposedGeometry, ALCModule):
    """A composed geometry that is trained on ALC axioms.

    ``forward(axiom, x, e_emb=None)`` is the loss of a batch ``x`` of an axiom pattern (the
    ``axiom_cost`` of the objective, which has to be an :class:`ExpressionReading`);
    ``forward(rows, "gci0")`` still reads a normal form, so the same geometry is also trained
    by :class:`~mowl.models.ComposedEL`. ``forward_fs`` and the operators of
    :class:`~mowl.nn.alc.module.ALCModule` are the geometry's own: the family of the class
    model for fuzzy geometries.

    A geometry whose domain is made of elements that are set from the outside for one
    evaluation (the sampled anonymous entities of FALCON) names it ``domain`` and gets the
    ``e_emb`` of the call as the context of that domain.

    :param schedule: A :class:`TrainingSchedule`, or ``None``.
    """

    #: Whether the geometry has anonymous elements that a training loop samples every epoch
    #: (and passes as ``e_emb``).
    samples_anonymous = False
    #: The ALC methods have no axioms about relations (``r ⊑ s``, ``r ∘ t ⊑ s``).
    role_axiom_capable = False

    def __init__(self, classes, relations, measure, objective=None, individuals=None,
                 regularizers=(), schedule=None, relaxation=None, domain=None, **kwargs):
        ComposedGeometry.__init__(self, classes, relations, measure, objective,
                                  individuals=individuals, regularizers=regularizers,
                                  **kwargs)
        if not hasattr(self.objective, "axiom_cost"):
            raise TypeError(f"{type(self.objective).__name__} does not read ALC axioms: an "
                            f"objective for an expression geometry mixes in "
                            f"ExpressionReading.")
        self.schedule = schedule
        self.relaxation = relaxation
        self.domain = domain

    # -- ALC axioms -----------------------------------------------------------------------
    def _context(self, e_emb):
        domain = self.domain
        if domain is not None and e_emb is not None:
            domain.context = e_emb
        return domain

    def forward(self, *args, **kwargs):
        if len(args) > 1 and isinstance(args[1], str):
            # a normal form: (rows, form[, neg])
            return ComposedGeometry.forward(self, *args, **kwargs)
        axiom, x = args[0], args[1]
        e_emb = args[2] if len(args) > 2 else kwargs.pop("e_emb", None)
        domain = self._context(e_emb)
        try:
            return self.objective.axiom_cost(self, axiom, x)
        finally:
            if domain is not None:
                domain.context = None

    def forward_fs(self, cexpr, x, e_emb=None, cur_index=0):
        """The degrees of a class expression pattern on the elements of the domain, and the
        next column (:meth:`~mowl.nn.alc.module.ALCModule.forward_fs`)."""
        domain = self._context(e_emb)
        try:
            region, nxt = pattern_region(self, cexpr, x, cur_index)
        finally:
            if domain is not None:
                domain.context = None
        return region.values, nxt

    # -- the operators of the family (ALCModule) -------------------------------------------
    @property
    def family(self):
        return self.class_model.family

    def _logical_and(self, x, y):
        return self.family.and_(x, y)

    def _logical_or(self, x, y):
        return self.family.or_(x, y)

    def _logical_not(self, x):
        return self.family.not_(x)

    def _logical_residuum(self, r_fs, c_fs):
        return self.family.implies(r_fs, c_fs)

    def _logical_exist(self, r_fs, c_fs):
        return self.family.exists(r_fs, c_fs)

    def _logical_forall(self, r_fs, c_fs):
        return self.family.forall(r_fs, c_fs)

    # -- names other code expects of an ALC module -----------------------------------------
    @property
    def c_embedding(self):
        """The table of the class parameters."""
        return self.class_model.centres

    @property
    def r_embedding(self):
        """The table of the relation parameters."""
        return self.relation_model.vectors

    @property
    def e_embedding(self):
        """The table of the individual (or domain element) parameters."""
        if self.domain is not None:
            return self.domain.table
        return None if self.individual_model is None else self.individual_model.centres

    @property
    def temperature(self):
        """The temperature of the smooth relaxation of the operators (``0``: exact); only
        for geometries that have one (:class:`Relaxation`)."""
        return 0.0 if self.relaxation is None else self.relaxation.temperature

    @temperature.setter
    def temperature(self, value):
        if self.relaxation is None:
            raise AttributeError("This geometry has no relaxation to set a temperature on.")
        self.relaxation.temperature = float(value)
