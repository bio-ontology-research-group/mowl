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

and the parts that the training of these geometries is made of:

* :class:`AxiomKindWeighting`: the objective part that combines the costs of the axiom
  patterns into the loss of a step (terminological axioms, class assertions and role
  assertions weighted, FALCON's ``alpha``/``beta``); every :class:`ExpressionReading`
  objective carries one as ``objective.weighting``;
* :class:`AnonymousElements`: the sampler of the anonymous domain elements a geometry with an
  open domain is evaluated on (FALCON);
* :class:`TrainingSchedule`: a :class:`~mowl.training.TrainingRegime` that a geometry carries
  (``geometry.schedule``) and that the shared training loop (:mod:`mowl.training.loop`)
  consults, for methods whose training changes the geometry (:class:`AnnealedRelaxation`: a
  temperature annealed to zero).
"""

import torch as th

from mowl.nn.alc.module import ALCModule
from mowl.owlapi import (OWLClassAssertionAxiom, OWLDisjointClassesAxiom,
                         OWLEquivalentClassesAxiom, OWLObjectPropertyAssertionAxiom,
                         OWLSubClassOfAxiom)
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.logic_expressions import pattern_region
from mowl.training.regime import TrainingRegime

__all__ = ["ExpressionReading", "ExpressionGeometry", "TrainingSchedule", "Relaxation",
           "AnnealedRelaxation", "AxiomKindWeighting", "AnonymousElements", "TBOX",
           "ABOX_EC", "ABOX_EE"]

#: The kinds of axioms :class:`AxiomKindWeighting` weighs: terminological axioms, class
#: assertions (concept-assertion, "ABox-EC") and role assertions ("ABox-EE").
TBOX = "tbox"
ABOX_EC = "abox_ec"
ABOX_EE = "abox_ee"


class AxiomKindWeighting:
    """Objective part: how the costs of the axiom patterns of an ontology are combined into
    the loss of one training step of an ALC geometry.

    An axiom is of one of three kinds (:meth:`kind`): terminological (``SubClassOf``,
    ``EquivalentClasses``, ``DisjointClasses``), a class assertion or a role assertion. Every
    axiom pattern is evaluated in batches; the cost of a batch is the mean of the costs of its
    rows (:meth:`batch_costs`), and the loss is

    .. math::

       \\sum_k w_k \\cdot \\frac{1}{n_k} \\sum_{b \\in k} \\mathrm{cost}(b)

    over the kinds :math:`k` that have batches, :math:`n_k` the number of batches of kind
    :math:`k` (all patterns of the kind together). Axioms of no kind are not trained. This is
    the weighting of FALCON (``alpha``, ``beta``), which DF-ALC and Al-cone use as well; any
    other combination of weights is a swap (``ComposedALC(objective=...)``, or
    ``objective_options={"weighting": AxiomKindWeighting(...)}``).

    :param tbox: Weight :math:`\\alpha` of the terminological axioms.
    :param class_assertions: Weight :math:`\\beta` of the class assertions.
    :param role_assertions: Weight of the role assertions; ``None`` (default):
        :math:`1 - \\alpha - \\beta`, and then :math:`\\alpha, \\beta \\in [0, 1]` with
        :math:`\\alpha + \\beta \\le 1`.
    """

    KINDS = (TBOX, ABOX_EC, ABOX_EE)

    def __init__(self, tbox=0.3, class_assertions=0.3, role_assertions=None):
        if role_assertions is None:
            if not (0 <= tbox <= 1 and 0 <= class_assertions <= 1):
                raise ValueError("Parameters alpha and beta must be in the interval [0, 1].")
            if tbox + class_assertions > 1:
                raise ValueError("The sum of alpha and beta must not exceed 1 "
                                 "(the ABox-EE weight is 1 - alpha - beta).")
            role_assertions = 1 - tbox - class_assertions
        elif min(tbox, class_assertions, role_assertions) < 0:
            raise ValueError("The weights of the axiom kinds must not be negative.")
        self.tbox, self.class_assertions = tbox, class_assertions
        self.role_assertions = role_assertions

    def __repr__(self):
        return (f"AxiomKindWeighting(tbox={self.tbox}, class_assertions="
                f"{self.class_assertions}, role_assertions={self.role_assertions})")

    @property
    def weights(self):
        """``{kind: weight}``."""
        return {TBOX: self.tbox, ABOX_EC: self.class_assertions,
                ABOX_EE: self.role_assertions}

    @staticmethod
    def kind(axiom):
        """The kind of an axiom (or axiom pattern): :data:`TBOX`, :data:`ABOX_EC`,
        :data:`ABOX_EE`, or ``None`` for an axiom that is not trained."""
        if isinstance(axiom, (OWLSubClassOfAxiom, OWLEquivalentClassesAxiom,
                              OWLDisjointClassesAxiom)):
            return TBOX
        if isinstance(axiom, OWLClassAssertionAxiom):
            return ABOX_EC
        if isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            return ABOX_EE
        return None

    @staticmethod
    def batch_costs(geometry, axiom, batches, context=None, device="cpu"):
        """The cost of every batch of rows of the pattern ``axiom``: the mean over its rows of
        ``geometry(axiom, rows, context)``, one value per batch (shape ``(n_batches,)``).
        The batches are evaluated one after the other as they come (a loader that shuffles
        draws its order here)."""
        costs = [th.mean(geometry(axiom, batch[0].to(device), context)) for batch in batches]
        return th.stack(costs) if costs else th.zeros(0, device=device)

    def batch_counts(self, patterns):
        """``{kind: number of batches}`` of ``{axiom: batches}`` (anything with ``len``)."""
        counts = dict.fromkeys(self.KINDS, 0)
        for axiom, batches in patterns.items():
            kind = self.kind(axiom)
            if kind is not None:
                counts[kind] += len(batches)
        return counts

    def combine(self, axiom, costs, counts):
        """The share of the loss of the batch costs ``costs`` of the pattern ``axiom``, given
        the numbers of batches of every kind of the step (:meth:`batch_counts`)."""
        kind = self.kind(axiom)
        return self.weights[kind] * costs.sum() / counts[kind]

    def loss(self, geometry, patterns, context=None, device="cpu"):
        """The loss of ``{axiom: batches}``, or ``None`` if no axiom is of a kind."""
        per_kind = {kind: [] for kind in self.KINDS}
        for axiom, batches in patterns.items():
            kind = self.kind(axiom)
            if kind is not None:
                per_kind[kind].append(self.batch_costs(geometry, axiom, batches, context,
                                                       device))
        total = None
        for kind, costs in per_kind.items():
            costs = [c for c in costs if len(c)]
            if not costs:
                continue
            term = self.weights[kind] * th.cat(costs).mean()
            total = term if total is None else total + term
        return total


class AnonymousElements:
    """Sampler of the anonymous domain elements that a geometry with an open domain is
    evaluated on (FALCON): every epoch ``count`` element vectors, half of them (at most the
    number of named elements) perturbations of the first named element vectors by Gaussian
    noise of scale ``noise``, the others drawn fresh (Xavier uniform), as in the original
    FALCON code. The context of one evaluation is the named elements followed by the sample
    (:meth:`context`).

    :param count: Number of anonymous elements per epoch (``0``: none).
    :param noise: Scale of the perturbation.
    :param perturbed: ``False``: every element is drawn fresh (no perturbations).
    """

    def __init__(self, count=4, noise=0.1, perturbed=True):
        self.count, self.noise, self.perturbed = int(count), float(noise), bool(perturbed)

    def sample(self, named, dim, device="cpu"):
        """``count`` anonymous element vectors of dimension ``dim`` (``named``: the table of
        the named element vectors, detached)."""
        n_perturbed = min(self.count // 2, named.shape[0]) if self.perturbed else 0
        n_fresh = self.count - n_perturbed
        parts = []
        if n_perturbed > 0:
            noise = th.normal(0, self.noise, size=(n_perturbed, dim)).to(device)
            parts.append(named[:n_perturbed] + noise)
        if n_fresh > 0:
            fresh = th.empty(n_fresh, dim).to(device)
            th.nn.init.xavier_uniform_(fresh)
            parts.append(fresh)
        return th.cat(parts, dim=0)

    @staticmethod
    def context(named, n_named, sample):
        """The first ``n_named`` rows of ``named`` followed by ``sample`` (only the sample for
        an ontology without individuals)."""
        return th.cat([named[:n_named], sample], dim=0)

    @staticmethod
    def with_seed(make, seed=0):
        """``make()`` evaluated with the random generator seeded with ``seed``, which is
        restored afterwards (a fixed context, so that scores are reproducible)."""
        state = th.random.get_rng_state()
        th.manual_seed(seed)
        try:
            return make()
        finally:
            th.random.set_rng_state(state)


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

    # -- scores ---------------------------------------------------------------------------
    @staticmethod
    def inclusion_score(g, sub, sup, x):
        """The cost of the inclusion of the class expression patterns ``sub`` in ``sup`` for
        the rows ``x``, one value per row: the ``inside`` of the measure with no margin and
        no gap (what the rank-based evaluation regimes rank)."""
        a, column = pattern_region(g, sub, x, 0)
        b, _ = pattern_region(g, sup, x, column)
        return g.measure.inside(a, b, 0.0).reshape(len(x), -1).sum(-1)

    @staticmethod
    def class_inclusion_score(g, sub, sup):
        """The cost of ``sub ⊑ sup`` for class ids ``sub``, ``sup`` (tensors of equal
        length), one value per pair, as :meth:`inclusion_score`."""
        return g.measure.inside(g.classes(sub), g.classes(sup), 0.0).reshape(len(sub), -1) \
            .sum(-1)

    #: How the costs of the axiom patterns are combined into a loss
    #: (:class:`AxiomKindWeighting`); ``None``: the model that trains the geometry gives one
    #: (from its ``alpha`` and ``beta``).
    weighting = None


class Relaxation:
    """The temperature of the smooth relaxation of the operators of a geometry, a number that
    its operator family and its measure read at every call (``source``), and that a
    :class:`TrainingSchedule` changes during training."""

    def __init__(self, temperature=0.0):
        self.temperature = float(temperature)


class TrainingSchedule(TrainingRegime):
    """A :class:`~mowl.training.TrainingRegime` that belongs to a geometry: what the training
    of a method does to the geometry around the optimiser steps (a temperature, noise), so
    that it is a part (swappable) and not code of one model. The geometry carries it as
    ``geometry.schedule`` and names it among its ``default_regimes``; the shared loop
    (:mod:`mowl.training.loop`) of :class:`~mowl.models.ComposedALC` consults it like any
    regime. ``self.model.module`` is the geometry. The default changes nothing."""


class AnnealedRelaxation(TrainingSchedule):
    """The training schedule of Al-cone: the operators are smoothed with a temperature that
    falls linearly from ``temperature`` to ``0`` over the epochs (``temperature * (1 -
    epoch / epochs)``), and every parameter gets Gaussian noise of scale ``noise *
    temperature`` after each step (it vanishes at the end, and helps to leave local minima).
    At the end of every epoch, so for validation, and after training the geometry has the
    exact operators (temperature ``0``).

    :param temperature: Initial temperature (read when training starts).
    :param noise: Scale of the noise.
    """

    key = "annealed-relaxation"

    def __init__(self, temperature=0.3, noise=0.3):
        super().__init__()
        self.temperature = temperature
        self.noise = noise
        self._tau0 = temperature

    def attach(self, model):
        module = getattr(model, "module", None)
        if not hasattr(module, "temperature") or (
                isinstance(module, ExpressionGeometry) and module.relaxation is None):
            raise TypeError(f"AnnealedRelaxation needs a geometry with a temperature (a "
                            f"relaxation); the geometry of {type(model).__name__} has none.")
        super().attach(model)

    def on_epoch_start(self, epoch, epochs):
        if epoch == 0:
            self._tau0 = self.temperature
        self.model.module.temperature = self._tau0 * (1.0 - epoch / max(epochs, 1))

    def after_step(self):
        module = self.model.module
        if self.noise > 0 and module.temperature > 0:
            with th.no_grad():
                for p in module.parameters():
                    p.add_(self.noise * module.temperature * th.randn_like(p))

    def on_epoch_end(self, epoch, losses):
        self.model.module.temperature = 0.0

    def on_train_end(self):
        self.model.module.temperature = 0.0


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

    @property
    def default_regimes(self):
        """The regimes of the training of the geometry: its :attr:`schedule`, if it has one
        (what :class:`~mowl.training.RegimeTrained` attaches)."""
        schedule = self.schedule
        return () if schedule is None else (lambda: schedule,)

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

    def refuses(self):
        """Why the parts of the geometry cannot be trained together (a sentence), or
        ``None``: a geometry with an open domain (:attr:`domain`, FALCON's element vectors
        completed by sampled anonymous elements) needs class and relation models that
        evaluate on that domain; the objective may require more of the parts
        (``objective.requirements``)."""
        domain = self.domain
        if domain is not None:
            for kind, part in (("class model", self.class_model),
                               ("relation model", self.relation_model)):
                if part is not None and getattr(part, "domain", None) is not domain:
                    return (f"the geometry evaluates class expressions on an open domain of "
                            f"element vectors ({type(domain).__name__}, completed by sampled "
                            f"anonymous elements in training), and the {kind} "
                            f"{type(part).__name__} does not evaluate on that domain (it "
                            f"has a domain of its own, or none)")
        requirements = getattr(self.objective, "requirements", None)
        return None if requirements is None else requirements(self)

    def inclusion_costs(self, sub, sup, x, e_emb=None):
        """The cost of the inclusion of the class expression patterns ``sub`` in ``sup`` for
        the rows ``x`` on the context ``e_emb``, one value per row (the
        :meth:`~ExpressionReading.inclusion_score` of the objective)."""
        domain = self._context(e_emb)
        try:
            return self.objective.inclusion_score(self, sub, sup, x)
        finally:
            if domain is not None:
                domain.context = None

    def class_inclusion_costs(self, sub, sup, e_emb=None):
        """The cost of ``sub ⊑ sup`` for tensors of class ids on the context ``e_emb`` (the
        :meth:`~ExpressionReading.class_inclusion_score` of the objective)."""
        domain = self._context(e_emb)
        try:
            return self.objective.class_inclusion_score(self, sub, sup)
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
