r"""Models of subspace logic (Embed2Reason and relatives) put together from parts.

:class:`SubspaceAxiomModel` is what such models share, whatever trains them: the extraction
of the axioms into rows (one row kind per axiom form of the quantum logic), the
hierarchy-derived data (siblings, separation pairs, range negatives) and the reasoning and
scoring on a trained embedding. The maths of the scores is that of the subspace geometry of the module.

:class:`ComposedSubspace` trains it with the regimes of :mod:`mowl.training`, for any method
whose geometry is a :class:`~mowl.representation.library.logic_subspace.SubspaceGeometry`
(``method="e2r"``), with the parts replaceable like those of
:class:`~mowl.models.ComposedEL`. :class:`mowl.models.E2R` is that combination under the name
of the method; the direct implementation in :mod:`mowl.baselines.models.e2r` keeps its own loss
assembly and loop for the comparison in ``tests/baselines``."""
import math
import random
from collections import defaultdict
from functools import partial

import torch as th

from mowl.base_models.model import Model
from mowl.evaluation.score_ranking import rank_by_score
from mowl.evaluation.subspace_violation import discretised_violations
from mowl.evaluation.validation import summed_mean_cost
from mowl.nn.alc.e2r.module import AXIOM_KINDS
from mowl.owlapi import ClassExpressionType, Imports
from mowl.representation import methods as M
from mowl.representation.library.logic_subspace import (Orthogonality, Separation,
                                                        SubspaceStructure)
from mowl.representation.swap import build_method
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.graph import EdgeCorruption, KnownRowMask
from mowl.training.loop import RegimeTrained, Rows
from mowl.training.losses import SumLoss
from mowl.training.regularization import IntervalProjection, Regularization

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


def _iri(entity):
    return str(entity.toStringID())


class SubspaceAxiomModel(Model):
    """The data and the scoring of subspace-logic models (see :class:`mowl.models.E2R`). A
    subclass makes ``init_module`` and trains."""

    axiom_scoring = "method"
    scored_forms = ("gci0", "class_assertion")

    def __init__(self, dataset, embed_dim=100, learning_rate=0.01, batch_size=4096,
                 num_negatives=3, negative_weight=1.0, reg_orth=1.0, reg_sep=1.0,
                 sep_gap=None, range_negatives="shrunk", max_sibling_pairs=100000,
                 role_unit_length=True, model_filepath=None, device="cpu", seed=0):
        super().__init__(dataset, model_filepath=model_filepath)
        if range_negatives not in ("shrunk", "all"):
            raise ValueError("range_negatives must be 'shrunk' or 'all'.")
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.negative_weight = negative_weight
        self.reg_orth = reg_orth
        self.reg_sep = reg_sep
        self.sep_gap = math.sqrt(embed_dim) if sep_gap is None else sep_gap
        self.range_negatives = range_negatives
        self.max_sibling_pairs = max_sibling_pairs
        self.role_unit_length = role_unit_length
        self.device = device
        self._rng = random.Random(seed)

        self._cid = self.class_index_dict
        self._rid = self.object_property_index_dict
        self._iid = self.individual_index_dict
        self.module = None
        self.skipped_axioms = []
        self._load_axioms()
        self.init_module()

    # -- axiom extraction ---------------------------------------------------------------
    def _named_class(self, cexpr):
        if cexpr.getClassExpressionType() == ClassExpressionType.OWL_CLASS:
            return self._cid.get(_iri(cexpr.asOWLClass()))
        return None

    def _named_classes(self, cexprs):
        idx = [self._named_class(c) for c in cexprs]
        return None if any(i is None for i in idx) else idx

    def _named_property(self, prop):
        if prop.isAnonymous():
            return None
        return self._rid.get(_iri(prop.asOWLObjectProperty()))

    def _add_complex_equivalence(self, rows, a, expr):
        """Rows for A ≡ expr; returns False if expr is not of a supported form."""
        ctype = expr.getClassExpressionType()
        if ctype == ClassExpressionType.OBJECT_INTERSECTION_OF:
            ops = self._named_classes(expr.getOperandsAsList())
            if ops is None:
                return False
            rows[f"concept_conjunction_{len(ops)}"].append([a] + ops)
            return True
        if ctype == ClassExpressionType.OBJECT_UNION_OF:
            ops = self._named_classes(expr.getOperandsAsList())
            if ops is None:
                return False
            rows[f"concept_disjunction_{len(ops)}"].append([a] + ops)
            return True
        if ctype == ClassExpressionType.OBJECT_COMPLEMENT_OF:
            b = self._named_class(expr.getOperand())
            if b is None:
                return False
            rows["concept_negation"].append([a, b])
            return True
        return False

    def _axiom_rows(self, axiom, rows):
        """Adds the rows of one axiom to ``rows``; returns False if unsupported."""
        name = str(axiom.getAxiomType().getName())
        if name == "ClassAssertion":
            c = self._named_class(axiom.getClassExpression())
            ind = axiom.getIndividual()
            if c is None or ind.isAnonymous():
                return False
            rows["class_assertion"].append([self._iid[_iri(ind)], c])
            return True
        if name == "ObjectPropertyAssertion":
            prop, s, o = axiom.getProperty(), axiom.getSubject(), axiom.getObject()
            if s.isAnonymous() or o.isAnonymous():
                return False
            if prop.isAnonymous():  # inverse property: swap the arguments
                prop, s, o = prop.getNamedProperty(), o, s
            r = self._rid.get(_iri(prop))
            if r is None:
                return False
            rows["role_assertion"].append([self._iid[_iri(s)], r, self._iid[_iri(o)]])
            return True
        if name == "SubClassOf":
            sub, sup = axiom.getSubClass(), axiom.getSuperClass()
            a = self._named_class(sub)
            stype = sup.getClassExpressionType()
            if a is not None and stype == ClassExpressionType.OWL_CLASS:
                rows["concept_inclusion"].append([a, self._named_class(sup)])
                return True
            if a is not None and stype == ClassExpressionType.OBJECT_INTERSECTION_OF:
                ops = self._named_classes(sup.getOperandsAsList())
                if ops is None:
                    return False
                rows["concept_inclusion"].extend([a, b] for b in ops)
                return True
            if a is not None and stype == ClassExpressionType.OBJECT_COMPLEMENT_OF:
                b = self._named_class(sup.getOperand())
                if b is None:
                    return False
                rows["concept_disjointness"].append([a, b])
                return True
            if (sub.isOWLThing() and stype == ClassExpressionType.OBJECT_ALL_VALUES_FROM):
                r = self._named_property(sup.getProperty())
                c = self._named_class(sup.getFiller())
                if r is None or c is None:
                    return False
                rows["range"].append([r, c])
                return True
            return False
        if name == "EquivalentClasses":
            exprs = list(axiom.getClassExpressionsAsList())
            named = [self._named_class(e) for e in exprs]
            complex_ = [e for e, n in zip(exprs, named) if n is None]
            named = [n for n in named if n is not None]
            if len(complex_) > 1 or not named:
                return False
            for i in named:
                for j in named:
                    if i != j:
                        rows["concept_inclusion"].append([i, j])
            for i in named:
                if complex_ and not self._add_complex_equivalence(rows, i, complex_[0]):
                    return False
            return True
        if name == "DisjointClasses":
            idx = self._named_classes(list(axiom.getClassExpressionsAsList()))
            if idx is None:
                return False
            for k, i in enumerate(idx):
                for j in idx[k + 1:]:
                    rows["concept_disjointness"].append([i, j])
            return True
        if name == "SubObjectPropertyOf":
            r = self._named_property(axiom.getSubProperty())
            s = self._named_property(axiom.getSuperProperty())
            if r is None or s is None:
                return False
            rows["role_inclusion"].append([r, s])
            return True
        if name == "EquivalentObjectProperties":
            props = [self._named_property(p) for p in axiom.getProperties()]
            if any(p is None for p in props):
                return False
            for i in props:
                for j in props:
                    if i != j:
                        rows["role_inclusion"].append([i, j])
            return True
        if name == "ObjectPropertyRange":
            r = self._named_property(axiom.getProperty())
            c = self._named_class(axiom.getRange())
            if r is None or c is None:
                return False
            rows["range"].append([r, c])
            return True
        return False

    def extract_rows(self, ontology):
        """Index rows of the supported axioms of ``ontology`` grouped by kind, and the
        list of unsupported axioms. Conjunction and disjunction rows are grouped by
        arity (keys ``concept_conjunction_<k>``)."""
        rows = defaultdict(list)
        skipped = []
        for axiom in ontology.getAxioms(Imports.INCLUDED):
            if not axiom.isLogicalAxiom():
                continue
            if not self._axiom_rows(axiom, rows):
                skipped.append(axiom)
        return dict(rows), skipped

    # -- hierarchy-derived data ---------------------------------------------------------
    @staticmethod
    def _closure(edges, nodes):
        up = defaultdict(set)
        for a, b in edges:
            if a != b:
                up[a].add(b)

        def reach(n, nxt):
            seen, stack = set(), [n]
            while stack:
                for m in nxt[stack.pop()]:
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
            return seen
        down = defaultdict(set)
        for a, bs in up.items():
            for b in bs:
                down[b].add(a)
        anc = {n: reach(n, up) for n in nodes}
        desc = {n: reach(n, down) for n in nodes}
        return up, down, anc, desc

    def _sibling_pairs(self, up, nodes):
        children = defaultdict(set)
        for n in nodes:
            parents = up.get(n) or {None}  # None: the (virtual) top element
            for p in parents:
                children[p].add(n)
        pairs = set()
        for kids in children.values():
            kids = sorted(kids)
            for i, a in enumerate(kids):
                for b in kids[i + 1:]:
                    pairs.add((a, b))
        pairs = sorted(pairs)
        if len(pairs) > self.max_sibling_pairs:
            pairs = self._rng.sample(pairs, self.max_sibling_pairs)
        return pairs

    def _range_rows(self, range_rows, up, down, anc, desc, nodes):
        out = []
        roots = [n for n in nodes if not up.get(n)]
        for r, cj in range_rows:
            if cj not in anc:
                continue
            exclude = anc[cj] | desc[cj] | {cj}
            if self.range_negatives == "all":
                cand = set(nodes)
            else:
                cand = set(roots)
                for a in anc[cj]:
                    cand |= down[a]
            out.extend([r, ck] for ck in sorted(cand - exclude))
        return out

    def _load_axioms(self):
        rows, skipped = self.extract_rows(self.dataset.ontology)
        self.skipped_axioms = skipped
        special = {self._cid.get(_THING), self._cid.get(_NOTHING)}
        nodes = [i for i in self._cid.values() if i not in special]
        incl = [(a, b) for a, b in rows.get("concept_inclusion", [])
                if a not in special and b not in special]
        # The told hierarchy also contains the inclusions entailed by the definitions
        # C = D1 ⊓ ... (C ⊑ Di) and C = D1 ⊔ ... (Di ⊑ C); they are used for siblings and
        # for the non-descendants of eq. (9), not as extra losses.
        edges = list(incl)
        for key, def_rows in rows.items():
            for row in def_rows:
                if key.startswith("concept_conjunction"):
                    edges.extend((row[0], b) for b in row[1:])
                elif key.startswith("concept_disjunction"):
                    edges.extend((b, row[0]) for b in row[1:])
        edges = [(a, b) for a, b in edges if a not in special and b not in special]
        up, down, anc, desc = self._closure(edges, nodes)
        self._class_hierarchy = (up, down, anc, desc)

        role_nodes = list(self._rid.values())
        rincl = [tuple(x) for x in rows.get("role_inclusion", [])]
        rup, _, _, _ = self._closure(rincl, role_nodes)

        self.range_axioms = rows.pop("range", [])
        rows["range"] = self._range_rows(self.range_axioms, up, down, anc, desc, nodes)

        def tensor(x, width):
            return th.tensor(x, dtype=th.long, device=self.device).reshape(-1, width)

        self.training_rows = {k: tensor(v, len(v[0])) for k, v in rows.items() if v}
        self.sibling_pairs = tensor(self._sibling_pairs(up, nodes), 2)
        self.role_sibling_pairs = tensor(self._sibling_pairs(rup, role_nodes), 2)
        self.separation_pairs = tensor(incl, 2)
        self.role_separation_pairs = tensor(rincl, 2)
        self._known_ca = {tuple(x) for x in rows.get("class_assertion", [])}
        self._known_ra = {tuple(x) for x in rows.get("role_assertion", [])}
        self._class_idx = th.tensor(nodes, dtype=th.long, device=self.device)

    # -- training data and diagnostics ------------------------------
    @staticmethod
    def _kind(key):
        for kind in AXIOM_KINDS:
            if key == kind or key.startswith(kind + "_"):
                return kind
        raise KeyError(key)

    def total_loss(self):
        """Current value of the objective on all rows, without negatives (float)."""
        k, self.num_negatives = self.num_negatives, 0
        try:
            with th.no_grad():
                return float(self.compute_loss(full=True))
        finally:
            self.num_negatives = k

    def axiom_loss(self):
        """Sum over axiom kinds of the mean axiom loss (eqs. (4)-(9)) on all rows, without
        structure terms, regularisers and negatives (float)."""
        return summed_mean_cost(self.module, {self._kind(k): r
                                              for k, r in self.training_rows.items()})

    # -- reasoning and scoring: the geometry computes, the model reads the expressions ----
    def _expression_tree(self, cexpr):
        """A class expression (OWL object or class IRI) as the tree of class indices that
        :meth:`~mowl.representation.library.logic_subspace.SubspaceGeometry.indicator`
        evaluates."""
        if isinstance(cexpr, str):
            return ("class", self._cid[cexpr])
        ctype = cexpr.getClassExpressionType()
        if ctype == ClassExpressionType.OWL_CLASS:
            return ("class", self._cid[_iri(cexpr.asOWLClass())])
        if ctype == ClassExpressionType.OBJECT_INTERSECTION_OF:
            return ("and", [self._expression_tree(c) for c in cexpr.getOperandsAsList()])
        if ctype == ClassExpressionType.OBJECT_UNION_OF:
            return ("or", [self._expression_tree(c) for c in cexpr.getOperandsAsList()])
        if ctype == ClassExpressionType.OBJECT_COMPLEMENT_OF:
            return ("not", self._expression_tree(cexpr.getOperand()))
        raise NotImplementedError(f"E2R has no subspace for {ctype}.")

    def concept_indicator(self, cexpr, binary=False):
        """Indicator vector of a class expression (an OWL class expression or a class
        IRI) built from named classes with :math:`\\sqcap` (product), :math:`\\sqcup`
        (max), :math:`\\neg` (complement in the real half) -- Sec. 4, "Membership" task.

        :param binary: Round the named-class indicators at 0.5 first.
        """
        return self.module.indicator(self._expression_tree(cexpr), binary)

    def score_membership(self, cexpr, individuals=None, binary=False):
        """Residual length :math:`\\|\\overline{y} \\odot x_a\\|` of each individual for
        the class expression ``cexpr`` (lower = better; Sec. 5).

        :param individuals: IRIs; defaults to all individuals in index order.
        :rtype: numpy.ndarray
        """
        y = self.concept_indicator(cexpr, binary)
        ids = (list(range(len(self._iid))) if individuals is None
               else [self._iid[i] for i in individuals])
        return self.module.membership_scores(y, ids)

    def rank_members(self, cexpr, binary=False):
        """Individual IRIs sorted by increasing residual for ``cexpr``."""
        names = sorted(self._iid, key=self._iid.get)
        return rank_by_score(names, self.score_membership(cexpr, binary=binary))

    def score_role_membership(self, prop, pairs):
        """Residual lengths of the pair vectors of ``pairs`` (list of ``(subject,
        object)`` IRIs) for the relation ``prop`` (lower = better).

        :rtype: numpy.ndarray
        """
        return self.module.role_membership_scores(
            self._rid[prop], [self._iid[a] for a, _ in pairs], [self._iid[b] for _, b in pairs])

    def axiom_costs(self, gci_name, rows):
        """Cost of every candidate axiom in ``rows`` (model indices), lower is more
        plausible (see :meth:`~mowl.representation.library.logic_subspace.SubspaceGeometry.axiom_costs`)."""
        return self.module.axiom_costs(gci_name, rows)

    def score_subsumption(self, sub, sup, binary=False):
        """Inclusion loss of eq. (5) for class expressions ``sub``, ``sup`` (``0`` iff
        the subspace of ``sub`` lies in that of ``sup`` for 0/1 indicators)."""
        return self.module.inclusion_cost(self.concept_indicator(sub, binary),
                                          self.concept_indicator(sup, binary))

    def discretise(self, threshold=0.5):
        """0/1 indicators of all concepts and relations and the entity vectors, as numpy
        arrays ``(concepts, relations, entities)`` in index order."""
        return self.module.discretise(threshold)

    def violations(self, tol=1e-3, threshold=0.5):
        """Training axioms (as ``(kind, row)``) that the discretised embedding violates,
        decided with set operations on the axis supports and residual norms, independently
        of the loss functions. Membership holds if the residual is at most ``tol``
        times the norm of the vector (and the vector is not ``0``)."""
        yc, zr, x = self.discretise(threshold)
        rows = [(self._kind(key), rows) for key, rows in self.training_rows.items()]
        return discretised_violations(yc, zr, x, rows, self.embed_dim, tol)


E2RBase = SubspaceAxiomModel   # the name this class had in :mod:`mowl.models.e2r.common`


class ComposedSubspace(RegimeTrained, SubspaceAxiomModel):
    r"""A model of axis-parallel subspace logic put together from parts: the geometry of a
    method of :data:`mowl.representation.METHODS` that is a
    :class:`~mowl.representation.library.logic_subspace.SubspaceGeometry` (``"e2r"``), with
    any of its parts replaced, trained on the told axioms with the regimes of
    :mod:`mowl.training`. The axioms are read into rows of one kind per form of the quantum
    logic (see :meth:`extract_rows`); the terms that are not axioms (structure of the
    indicators, orthogonal siblings, separation of a class from its parents) are the
    regularisers of the geometry.

    .. code-block:: python

       model = ComposedSubspace(dataset, method="e2r", embed_dim=20)

    :param method: Key of a method whose geometry is a subspace geometry.
    :param classes: A class model (name, function ``(size, embed_dim)`` or built part) that
        replaces that of the method; so do ``relations``, ``individuals``, ``measure`` and
        ``objective`` (:func:`mowl.representation.swap.build_method`).
    :param objective_options: Options of the method's own objective that are replaced.
    :param regularizers: Penalties added to those of the method.
    :param embed_dim, learning_rate, batch_size, num_negatives, negative_weight, reg_orth,
        reg_sep, sep_gap, range_negatives, max_sibling_pairs, role_unit_length, seed:
        see :class:`mowl.models.E2R`.
    :param method_options: Further options of the method's builder.
    """

    def __init__(self, dataset, method="e2r", classes=None, relations=None, individuals=None,
                 measure=None, objective=None, objective_options=None, regularizers=(),
                 embed_dim=100, learning_rate=0.01, batch_size=4096, num_negatives=3,
                 negative_weight=1.0, reg_orth=1.0, reg_sep=1.0, sep_gap=None,
                 range_negatives="shrunk", max_sibling_pairs=100000, role_unit_length=True,
                 model_filepath=None, device="cpu", seed=0, **method_options):
        if method not in M.METHODS:
            raise ValueError(f"Unknown method {method!r}; available: {sorted(M.METHODS)}.")
        self.method = method
        self._part_choices = dict(
            classes=classes, relations=relations, individuals=individuals, measure=measure,
            objective=objective, objective_options=objective_options)
        self.extra_regularizers = tuple(regularizers)
        self.method_options = method_options
        super().__init__(dataset, embed_dim, learning_rate, batch_size, num_negatives,
                         negative_weight, reg_orth, reg_sep, sep_gap, range_negatives,
                         max_sibling_pairs, role_unit_length, model_filepath, device, seed)

    def init_module(self):
        geometry = build_method(
            self.method, len(self._cid), len(self._rid), len(self._iid),
            embed_dim=self.embed_dim, regularizers=(), top_index=self._cid.get(_THING),
            bottom_index=self._cid.get(_NOTHING), role_unit_length=self.role_unit_length,
            regularization=False,
            **{k: v for k, v in self._part_choices.items() if v is not None},
            **self.method_options)
        if not hasattr(geometry, "membership_residual"):
            raise TypeError(
                f"The geometry of {self.method!r} ({type(geometry).__name__}) is not a "
                f"subspace geometry: ComposedSubspace cannot train it.")
        reset = getattr(geometry, "reset_parameters", None)
        if reset is not None:
            reset()
        self._attach_regularizers(geometry)

    def _attach_regularizers(self, geometry):
        """Give ``geometry`` the terms of the objective that are not axioms (structure of the
        indicators, eqs. (1)-(3); siblings orthogonal, eq. (11); separation of a class from
        its parents, eq. (12); the pairs are read from the told hierarchy) and make it the
        module."""
        self.structure = SubspaceStructure()
        self.orthogonality = Orthogonality(self.reg_orth, self.sibling_pairs,
                                           self.role_sibling_pairs)
        self.separation = Separation(self.reg_sep, self.sep_gap, self.separation_pairs,
                                     self.role_separation_pairs)
        geometry.regularizers = (self.structure, self.orthogonality, self.separation,
                                 *self.extra_regularizers)
        self.module = geometry.to(self.device)

    # -- training -----------------------------------------------------------------------
    @property
    def default_regimes(self):
        """The regimes that make the training (Sec. 4 and 5 of the E2R paper): a random
        batch per axiom kind and step (:class:`~mowl.training.PerNormalFormMiniBatches`,
        an epoch has :math:`\\max_k \\lceil N_k / \\text{batch size} \\rceil` steps),
        ``num_negatives`` corruptions of every assertion that replace an individual
        (:class:`~mowl.training.EdgeCorruption`) minus the known assertions
        (:class:`~mowl.training.KnownRowMask` with ``drop``), the sum of the axiom losses and
        ``negative_weight`` times the negated-membership losses (:class:`~mowl.training.SumLoss`),
        and the projection of the indicators into :math:`[0, 1]`
        (:class:`~mowl.training.Regularization` of an
        :class:`~mowl.training.IntervalProjection`). The regularisers of eqs. (1)-(3), (11) and
        (12) are those of the module (``module.regularizers``)."""
        assertions = ("class_assertion", "role_assertion")
        parts = [partial(PerNormalFormMiniBatches, self.batch_size, "replacement")]
        n_ind = len(self._iid)
        if min(self.num_negatives, self.negative_weight, n_ind - 1) > 0:
            parts += [
                partial(EdgeCorruption, self.num_negatives, (0,), n_ind, ("class_assertion",),
                        "block"),
                partial(EdgeCorruption, self.num_negatives, (0, 2), n_ind, ("role_assertion",),
                        "block", False),
                partial(KnownRowMask, assertions, False, True)]
        parts.append(partial(SumLoss, self.negative_weight))
        parts.append(partial(Regularization, IntervalProjection(
            ("class_model.centres", "relation_model.vectors"), 0.0, 1.0)))
        return tuple(parts)

    @property
    def training_datasets(self):
        """One dataset per axiom kind and arity of the training rows (``training_rows``)."""
        return {key: Rows(rows) for key, rows in self.training_rows.items()}

    def score_rows(self, form, rows, neg=False):
        return self.module(self._kind(form), rows, neg=neg)

    def compute_loss(self, full=True):
        """The objective (eq. (10) with regularisers and negatives) on all rows as training
        computes it; without negatives when ``num_negatives`` is 0."""
        return self.batch_loss(self.training_datasets, negatives=self.num_negatives > 0)
