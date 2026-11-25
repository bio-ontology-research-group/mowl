"""Masked ABox revision: revise a corrupted grounding of an ontology into a model of its TBox.

The experiment of Section "Performance Evaluation" of the DF-ALC paper (Wu, Zhu, Zhao and
Dai, 2022, arXiv:2211.12006), the only one that is a pure ontology experiment: it imitates
the output of a neural network by *masking* the ideal grounding of an ontology.

1. The ideal grounding is the ABox of the ontology completed by a reasoner: truth value 1
   for every entailed class and role assertion, and 0.5 (unknown) everywhere else (the
   released code fills the grounding with 0.5; negative assertions are not made).
2. A random part (``mask_rate``) of the grounding is replaced by a random truth value in the
   unknown region ``(1 - alpha, alpha)``: the *masked grounding*, the imitation of a
   perceptual grounding.
3. The model starts from the masked grounding and is trained on the TBox alone (the ABox
   is *not* part of the training data: it is the grounding the model revises).
4. The revised grounding is turned into a crisp interpretation and the **success rate** is
   the fraction of the TBox axioms of the original ontology that it satisfies, the domain
   being the set of individuals. The text of the paper uses the crisp transformation of its
   Eq. 6 with ``alpha = 0.5`` for this, and the released code (``evaluation/evaluator.py``,
   mode ``crisp``) takes ``> 0.5`` as true and everything else, 0.5 included, as false;
   that is the default here (``crisp_alpha=0.5``, ``crisp_unknown="false"``). With
   ``crisp_unknown="half"`` the middle region stays unknown (value 0.5, Goedel semantics on
   the values 0, 0.5, 1).

The paper also evaluates conjunctive query answering on the revised grounding and the
object classification of a neural network on PASCAL-Part; neither is part of this regime
(see the benchmark notes).

World assumption: open world for the ideal grounding (a value that is not an entailed
assertion is unknown, 0.5); the crisp evaluation of the released code is closed world over
the thresholded grounding (the value 0.5 counts as false).

What is evaluated is the success rate over the TBox axioms the model trains on:
``SubClassOf``, ``EquivalentClasses`` (pairwise) and ``DisjointClasses`` (pairwise) between
:math:`\\mathcal{ALC}` class expressions of named classes and properties (the paper also
truncates the ontologies to their :math:`\\mathcal{ALC}` fragment). Such axioms with another
class expression (nominals, cardinalities, inverse roles) are removed from the training data
and counted in ``info["axioms_outside_alc"]``; other kinds of axioms (domain and range, role
axioms) are ignored. The paper's axiom counts (Table 1) are not the counts of this regime.
"""

import itertools
import logging
import uuid

import numpy as np

from mowl.evaluation.regime import EvaluationRegime

logger = logging.getLogger(__name__)

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


def in_alc(expr):
    """Whether an OWL API class expression is in :math:`\\mathcal{ALC}` over named classes and
    named (not inverse) object properties."""
    from mowl.owlapi import ClassExpressionType as T
    t = expr.getClassExpressionType()
    if t == T.OWL_CLASS:
        return True
    if t == T.OBJECT_COMPLEMENT_OF:
        return in_alc(expr.getOperand())
    if t in (T.OBJECT_INTERSECTION_OF, T.OBJECT_UNION_OF):
        return all(in_alc(o) for o in expr.getOperandsAsList())
    if t in (T.OBJECT_SOME_VALUES_FROM, T.OBJECT_ALL_VALUES_FROM):
        return not expr.getProperty().isAnonymous() and in_alc(expr.getFiller())
    return False


def crisp_transform(values, alpha=0.8, unknown="half"):
    """Eq. 6 of the paper: values above ``alpha`` become 1, below ``1 - alpha`` become 0;
    the values in between are unknown: ``unknown="half"`` makes them 0.5, ``"false"`` makes
    them 0 (with ``alpha = 0.5`` that is the released code's threshold: ``> 0.5`` is true,
    everything else false)."""
    values = np.asarray(values, dtype=float)
    middle = 0.5 if unknown == "half" else 0.0
    return np.where(values > alpha, 1.0, np.where(values < 1.0 - alpha, 0.0, middle))


class CrispALC:
    """Evaluate :math:`\\mathcal{ALC}` class expressions and TBox axioms on a finite
    interpretation with truth values in ``[0, 1]`` (Goedel semantics), used here with the
    three values 0, 0.5, 1.

    :param concepts: ``{class IRI: array of n truth values}``.
    :param roles: ``{role IRI: (n, n) array}``; ``roles[r][a, b]`` is ``r(a, b)``.
    :param n: Size of the domain.
    """

    def __init__(self, concepts, roles, n):
        self.concepts, self.roles, self.n = concepts, roles, n

    def concept(self, expr):
        """The truth values (array of ``n``) of an OWL API class expression; raises
        :class:`NotImplementedError` outside :math:`\\mathcal{ALC}` over named entities."""
        from mowl.owlapi import ClassExpressionType as T
        t = expr.getClassExpressionType()
        if t == T.OWL_CLASS:
            iri = str(expr.asOWLClass().toStringID())
            if iri == _THING:
                return np.ones(self.n)
            if iri == _NOTHING:
                return np.zeros(self.n)
            return self.concepts[iri]
        if t == T.OBJECT_COMPLEMENT_OF:
            return 1.0 - self.concept(expr.getOperand())
        if t in (T.OBJECT_INTERSECTION_OF, T.OBJECT_UNION_OF):
            ops = [self.concept(o) for o in expr.getOperandsAsList()]
            return (np.minimum.reduce(ops) if t == T.OBJECT_INTERSECTION_OF
                    else np.maximum.reduce(ops))
        if t in (T.OBJECT_SOME_VALUES_FROM, T.OBJECT_ALL_VALUES_FROM):
            prop = expr.getProperty()
            if prop.isAnonymous():
                raise NotImplementedError("inverse roles are not supported")
            r = self.roles[str(prop.asOWLObjectProperty().toStringID())]
            c = self.concept(expr.getFiller())
            if t == T.OBJECT_SOME_VALUES_FROM:
                return np.minimum(r, c[None, :]).max(axis=1)
            return np.maximum(1.0 - r, c[None, :]).min(axis=1)
        raise NotImplementedError(f"class expression type {t} is not in ALC")

    def satisfies(self, axiom):
        """Whether the interpretation satisfies a ``SubClassOf``, ``EquivalentClasses`` or
        ``DisjointClasses`` axiom; ``None`` if the axiom is of another kind or not in
        :math:`\\mathcal{ALC}` over named entities."""
        from mowl.owlapi import (OWLSubClassOfAxiom, OWLEquivalentClassesAxiom,
                                 OWLDisjointClassesAxiom)
        eps = 1e-9
        try:
            if isinstance(axiom, OWLSubClassOfAxiom):
                sub = self.concept(axiom.getSubClass())
                sup = self.concept(axiom.getSuperClass())
                return bool(np.all(sub <= sup + eps))
            if isinstance(axiom, (OWLEquivalentClassesAxiom, OWLDisjointClassesAxiom)):
                values = [self.concept(e) for e in axiom.getClassExpressionsAsList()]
                if isinstance(axiom, OWLEquivalentClassesAxiom):
                    return all(bool(np.all(np.abs(a - b) <= eps))
                               for a, b in zip(values, values[1:]))
                return all(bool(np.all(np.minimum(a, b) <= eps))
                           for a, b in itertools.combinations(values, 2))
        except (NotImplementedError, KeyError):
            return None
        return None

    def success_rate(self, axioms):
        """``(rate, satisfied, evaluated, skipped)`` over ``axioms``."""
        results = [self.satisfies(a) for a in axioms]
        evaluated = [r for r in results if r is not None]
        ok = sum(evaluated)
        return (ok / len(evaluated) if evaluated else float("nan"), ok, len(evaluated),
                len(results) - len(evaluated))


def ideal_grounding(class_assertions, role_assertions, class_index, role_index, ind_index):
    """The ideal grounding of an ABox: 1 for the given assertions, 0.5 elsewhere.

    :return: ``({class IRI: (n,) array}, {role IRI: (n, n) array})``; assertions about
        entities not in the index dictionaries are dropped.
    """
    n = len(ind_index)
    concepts = {c: np.full(n, 0.5) for c in class_index if c not in (_THING, _NOTHING)}
    roles = {r: np.full((n, n), 0.5) for r in role_index}
    for a, c in class_assertions:
        if a in ind_index and c in concepts:
            concepts[c][ind_index[a]] = 1.0
    for a, r, b in role_assertions:
        if a in ind_index and b in ind_index and r in roles:
            roles[r][ind_index[a], ind_index[b]] = 1.0
    return concepts, roles


def mask_grounding(concepts, roles, rate, alpha, rng, role_cells="code"):
    """Replace a random part of a grounding by random truth values in ``(1 - alpha, alpha)``.

    Concept cells: ``int(rate * C * n)`` of the ``C * n`` (class, individual) cells, drawn
    without replacement. Role cells: with ``role_cells="code"`` (the released code)
    ``int(rate * R * n)`` cells of the ``R * n * n`` role cells; with ``"rate"`` the
    fraction ``rate`` of all role cells, as the paper's wording suggests.

    :return: Masked copies ``(concepts, roles)`` and the numbers of masked cells.
    """
    if role_cells not in ("code", "rate"):
        raise ValueError("role_cells must be 'code' or 'rate'.")
    names = list(concepts)
    masked_c = {k: v.copy() for k, v in concepts.items()}
    masked_r = {k: v.copy() for k, v in roles.items()}
    n = len(next(iter(concepts.values()))) if concepts else 0

    def uniform(size):
        return rng.uniform(1.0 - alpha, alpha, size=size)

    total = len(names) * n
    count_c = int(rate * total)
    if count_c:
        for cell in rng.choice(total, count_c, replace=False):
            masked_c[names[cell // n]][cell % n] = uniform(None)
    rnames = list(roles)
    count_r = 0
    if rnames:
        total_r = len(rnames) * n * n
        count_r = int(rate * (len(rnames) * n if role_cells == "code" else total_r))
        count_r = min(count_r, total_r)
        if count_r:
            for cell in rng.choice(total_r, count_r, replace=False):
                r, rest = divmod(int(cell), n * n)
                masked_r[rnames[r]][rest // n, rest % n] = uniform(None)
    return masked_c, masked_r, count_c, count_r


class MaskedABoxRevisionRegime(EvaluationRegime):
    """The masked ABox revision task of the DF-ALC paper.

    :meth:`prepare` replaces the ontology of the dataset by its TBox (with the names and
    the individuals declared) and keeps the original for the evaluation;
    :meth:`initialise_model` gives the model the masked grounding as its starting point
    (the model must have ``set_grounding``, ``domain``, ``concept_grounding`` and
    ``role_grounding``, as :class:`~mowl.models.dfalc.model.DFALCModel`); :meth:`evaluate`
    reports the success rate of the revised grounding.

    :param mask_rate: Fraction of the grounding that is masked (paper: 0.2, 0.4, 0.6, 0.8).
    :param alpha: Upper end of the unknown region the masked values are drawn from (paper:
        0.8).
    :param crisp_alpha: Threshold of the crisp transformation of the evaluation (text of the
        paper and released code: 0.5).
    :param crisp_unknown: ``"false"`` (code: the middle region is false) or ``"half"``
        (unknown, 0.5) in the crisp transformation.
    :param mask_seed: Seed of the masking (the released code seeds NumPy with 42).
    :param role_cells: How many role cells are masked, ``"code"`` or ``"rate"``
        (:func:`mask_grounding`).
    :param reasoner: Reasoner that completes the ABox (a backend of
        :mod:`mowl.reasoning`; default HermiT, complete for :math:`\\mathcal{ALC}`). If the
        ontology is inconsistent, the asserted ABox is used and ``info["completion"]`` says
        so.
    """

    key = "masked-abox-revision"
    metrics = ("success_rate",)
    subsets = ("all",)

    def __init__(self, mask_rate=0.2, alpha=0.8, mask_seed=42, role_cells="code",
                 reasoner="hermit", crisp_alpha=0.5, crisp_unknown="false"):
        super().__init__()
        if not 0.0 <= mask_rate <= 1.0:
            raise ValueError("mask_rate must be in [0, 1].")
        if not 0.5 <= alpha < 1.0:
            raise ValueError("alpha must be in [0.5, 1).")
        if role_cells not in ("code", "rate"):
            raise ValueError("role_cells must be 'code' or 'rate'.")
        if not 0.5 <= crisp_alpha < 1.0:
            raise ValueError("crisp_alpha must be in [0.5, 1).")
        if crisp_unknown not in ("false", "half"):
            raise ValueError("crisp_unknown must be 'false' or 'half'.")
        self.crisp_alpha, self.crisp_unknown = float(crisp_alpha), crisp_unknown
        self.mask_rate, self.alpha, self.mask_seed = float(mask_rate), float(alpha), mask_seed
        self.role_cells, self.reasoner = role_cells, reasoner
        self.info = {}
        self._source = None
        self._class_assertions = self._role_assertions = None

    # -- data ---------------------------------------------------------------------------
    def _complete(self, ontology):
        from org.semanticweb.owlapi.model import AxiomType
        from mowl.datasets.builtin._abox import entailed_assertions
        try:
            c, r = entailed_assertions(ontology, self.reasoner)
            return c, r, f"{self.reasoner} (entailed)"
        except ValueError as error:
            logger.warning("%s Using the asserted ABox.", error)
        c = sorted({(str(a.getIndividual().toStringID()),
                     str(a.getClassExpression().asOWLClass().toStringID()))
                    for a in ontology.getAxioms(AxiomType.CLASS_ASSERTION)
                    if not a.getClassExpression().isAnonymous()})
        r = sorted({(str(a.getSubject().toStringID()),
                     str(a.getProperty().asOWLObjectProperty().toStringID()),
                     str(a.getObject().toStringID()))
                    for a in ontology.getAxioms(AxiomType.OBJECT_PROPERTY_ASSERTION)
                    if not a.getProperty().isAnonymous()})
        return c, r, "asserted (the ontology is inconsistent)"

    def prepare(self, dataset):
        """The dataset with the ABox removed: the TBox, and the individuals declared."""
        from mowl.datasets import Dataset
        from mowl.datasets.builtin._abox import tbox_axioms
        from mowl.owlapi import OWLAPIAdapter
        self._source = dataset.ontology
        c, r, how = self._complete(self._source)
        self._class_assertions, self._role_assertions, self._completion = c, r, how
        adapter = OWLAPIAdapter()
        df = adapter.data_factory
        ont = adapter.create_ontology(f"http://mowl.borg/abox-revision/{uuid.uuid4().hex}")
        from java.util import HashSet
        axioms = HashSet()
        trained = [ax for ax in tbox_axioms(self._source) if self._is_trained_form(ax)]
        self._axioms = [ax for ax in trained if self._in_fragment(ax)]
        self._outside_alc = len(trained) - len(self._axioms)
        for ax in self._axioms:
            axioms.add(ax)
        for e in self._source.getClassesInSignature():
            axioms.add(df.getOWLDeclarationAxiom(e))
        for e in self._source.getObjectPropertiesInSignature():
            axioms.add(df.getOWLDeclarationAxiom(e))
        for e in self._source.getIndividualsInSignature():
            axioms.add(df.getOWLDeclarationAxiom(e))
        adapter.owl_manager.addAxioms(ont, axioms)
        return Dataset(ont)

    @staticmethod
    def _in_fragment(axiom):
        from mowl.owlapi import OWLSubClassOfAxiom
        if isinstance(axiom, OWLSubClassOfAxiom):
            return in_alc(axiom.getSubClass()) and in_alc(axiom.getSuperClass())
        return all(in_alc(e) for e in axiom.getClassExpressionsAsList())

    @staticmethod
    def _is_trained_form(axiom):
        from mowl.owlapi import (OWLSubClassOfAxiom, OWLEquivalentClassesAxiom,
                                 OWLDisjointClassesAxiom)
        return isinstance(axiom, (OWLSubClassOfAxiom, OWLEquivalentClassesAxiom,
                                  OWLDisjointClassesAxiom))

    # -- the masked grounding -----------------------------------------------------------
    def _crisp(self, model, concepts, roles):
        n = len(model.domain)
        crisp_c = {k: crisp_transform(v, self.crisp_alpha, self.crisp_unknown)
                   for k, v in concepts.items()}
        crisp_r = {k: crisp_transform(v, self.crisp_alpha, self.crisp_unknown)
                   for k, v in roles.items()}
        return CrispALC(crisp_c, crisp_r, n)

    def initialise_model(self, model, dataset):
        for needed in ("set_grounding", "concept_grounding", "role_grounding", "domain"):
            if not hasattr(model, needed):
                raise TypeError(
                    f"The masked ABox revision regime starts the model from a masked "
                    f"grounding; {type(model).__name__} has no {needed!r}. Use a model "
                    f"with a grounding (DF-ALC).")
        if self._source is None:
            raise RuntimeError("prepare(dataset) must be called before the model is built.")
        ind_index = {name: i for i, name in enumerate(model.domain)}
        ideal_c, ideal_r = ideal_grounding(
            self._class_assertions, self._role_assertions, model.class_index_dict,
            model.object_property_index_dict, ind_index)
        rng = np.random.default_rng(self.mask_seed)
        masked_c, masked_r, count_c, count_r = mask_grounding(
            ideal_c, ideal_r, self.mask_rate, self.alpha, rng, self.role_cells)
        model.set_grounding(concepts=masked_c, roles=masked_r)
        ideal_rate = self._crisp(model, ideal_c, ideal_r).success_rate(self._axioms)
        masked_rate = self._crisp(model, masked_c, masked_r).success_rate(self._axioms)
        self._initial = {"ideal_success_rate": ideal_rate[0],
                         "masked_success_rate": masked_rate[0],
                         "masked_concept_cells": count_c, "masked_role_cells": count_r,
                         "completion": self._completion,
                         "completed_class_assertions": len(self._class_assertions),
                         "completed_role_assertions": len(self._role_assertions),
                         "axioms_outside_alc": self._outside_alc,
                         "axioms_evaluated": masked_rate[2], "axioms_skipped": masked_rate[3]}

    def evaluate(self, model, dataset):
        concepts, roles = model.concept_grounding(), model.role_grounding()
        rate, ok, evaluated, skipped = self._crisp(model, concepts, roles).success_rate(
            self._axioms)
        self.info = {**getattr(self, "_initial", {}), "satisfied_axioms": ok,
                     "axioms_evaluated": evaluated, "axioms_skipped": skipped,
                     "alpha": self.alpha, "crisp_alpha": self.crisp_alpha,
                     "crisp_unknown": self.crisp_unknown, "mask_rate": self.mask_rate,
                     "domain_size": len(model.domain), "role_cells": self.role_cells}
        return {"all": {"success_rate": rate}}
