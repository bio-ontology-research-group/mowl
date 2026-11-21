r"""Embed2Reason (E2R): quantum-logic subspace embeddings (Garg et al., NeurIPS 2019)."""
import math
import random
from collections import defaultdict

import numpy as np
import torch as th

from mowl.base_models.model import Model
from mowl.nn.alc.e2r import losses as L
from mowl.nn.alc.e2r.module import AXIOM_KINDS, E2RModule
from mowl.owlapi import ClassExpressionType, Imports

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


def _iri(entity):
    return str(entity.toStringID())


class E2R(Model):
    r"""Embed2Reason [garg2019]_: concepts and relations as axis-parallel linear subspaces,
    entities as vectors, the logic is quantum logic (Sec. 3).

    The space :math:`\mathbb{C}^d` over the field :math:`\mathbb{R}` is simulated by
    :math:`\mathbb{R}^{2d}` (Sec. 4). A concept is the span of the axes selected by a
    learnable indicator :math:`y \in [0, 1]^{2d}` (real half only), a relation the span
    selected by :math:`z` (both halves), an entity a vector :math:`x = (x_R, 0)` and an
    ordered pair :math:`(a, b)` the vector :math:`(x_{aR}, x_{bR})`. :math:`C(a)` holds iff
    :math:`x_a \in C`, :math:`R(a, b)` iff :math:`x_{ab} \in R`. :math:`\sqcap` is
    intersection, :math:`\sqcup` the span (sum) of subspaces, :math:`\neg` the orthogonal
    complement, :math:`\sqsubseteq` inclusion. The loss is eq. (10): the sum of the terms
    of eqs. (1)-(9), each averaged over its assertions, plus the regularisers of eqs. (11)
    and (12) and the negated-membership negatives of Sec. 4 (3 per positive, Sec. 5).

    Axioms used for training (everything else is listed in :attr:`skipped_axioms`):

    ======================================================  ===========================
    OWL axiom                                               loss
    ======================================================  ===========================
    ``ClassAssertion(A, a)``                                eq. (4)
    ``ObjectPropertyAssertion(R, a, b)``                    eq. (4)
    ``SubClassOf(A, B)``, ``SubClassOf(A, B1 ⊓ … ⊓ Bk)``    eq. (5) (per conjunct)
    ``EquivalentClasses(A, B)``                             eq. (5) both directions
    ``EquivalentClasses(A, B1 ⊓ ... ⊓ Bk)``                 eq. (6)
    ``EquivalentClasses(A, B1 ⊔ ... ⊔ Bk)``                 eq. (7)
    ``EquivalentClasses(A, ¬B)``                            eq. (8) (real half)
    ``DisjointClasses(A, B)``, ``SubClassOf(A, ¬B)``        eq. (6) with A ⊓ B = ⊥ = {0}
    ``SubObjectPropertyOf(R, S)``, equivalences             eq. (5) for relations
    ``ObjectPropertyRange(R, C)``, ``⊤ ⊑ ∀R.C``             eq. (9)
    ======================================================  ===========================

    All class and property operands must be named. The paper's universal type
    restriction (Sec. 4) is the statement "every :math:`R`-successor is in :math:`C_j`",
    i.e. a range axiom; eq. (9) is summed over the non-descendants :math:`C_k` of
    :math:`C_j` chosen by the paper's shrinking rule (children of :math:`\top` and children
    of ancestors of :math:`C_j` that are not ancestors of :math:`C_j`), computed from the
    told class hierarchy. Existential restrictions have no E2R loss (the paper defines
    :math:`\exists R.C` in Sec. 2 but gives none) and are skipped. Role conjunction,
    disjunction and negation (eqs. (6)-(8) for relations) are implemented in the module,
    but OWL 2 has no syntax for them, so the model never produces them.

    Deviations from the paper (see also :mod:`mowl.nn.alc.e2r.losses`):

    * the second term of eq. (4) is omitted (as printed it is not zero in the situation
      where the text says it is); pair vectors are built from entity vectors, so the
      pair-structure terms of eq. (4) vanish identically;
    * role assertions also carry the unit-length terms of both entities;
    * concept negation (eq. (8)) is evaluated on the real half (otherwise it can never
      be zero together with eq. (2));
    * negatives: only the residual term of the membership loss is negated, the
      unit-length term is kept, so the objective is bounded below; corrupted assertions
      that are asserted in the training ontology are dropped;
    * indicators are clipped to :math:`[0, 1]` after every step; the paper does not say how
      they are kept in range, nor how parameters are initialised, how the terms are
      weighted against each other, or whether eqs. (11)-(12) were used together with
      the negatives (Sec. 4 offers the negatives as an alternative, Sec. 5 uses
      "regularization terms as described in Section 4" and negatives).

    :param dataset: mOWL dataset (training ontology with TBox and ABox).
    :param embed_dim: :math:`d` (vectors have :math:`2d` real coordinates). Defaults to
        ``100`` (Sec. 5).
    :param learning_rate: Adam learning rate (not stated in the paper). Defaults to
        ``0.01``.
    :param batch_size: Rows per axiom kind and step. Defaults to ``4096``.
    :param num_negatives: Corrupted assertions per positive membership assertion.
        Defaults to ``3`` (Sec. 5).
    :param negative_weight: Weight of the negated-membership term. Defaults to ``1``.
    :param reg_orth: Weight of :math:`\Omega_{\text{orth}}` (eq. (11)). Defaults to ``1``.
    :param reg_sep: Weight of :math:`\Omega_{\text{sep}}` (eq. (12)). Defaults to ``1``.
    :param sep_gap: Constant of eq. (12); ``None`` means :math:`\sqrt{d}` as in the paper.
    :param range_negatives: ``"shrunk"`` (the paper's reduced set of non-descendants) or
        ``"all"`` (every class that is neither an ancestor nor a descendant of
        :math:`C_j`).
    :param max_sibling_pairs: Upper bound on the sibling pairs used by eq. (11); larger
        sets are subsampled once. Defaults to ``100000``.
    :param role_unit_length: See :class:`~mowl.nn.alc.e2r.module.E2RModule`.
    :param seed: Seed of the Python random generator used for subsampling.
    """

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

    def init_module(self):
        self.module = E2RModule(
            len(self._iid), len(self._cid), len(self._rid), dim=self.embed_dim,
            top_index=self._cid.get(_THING), bottom_index=self._cid.get(_NOTHING),
            role_unit_length=self.role_unit_length).to(self.device)

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

    # -- training -----------------------------------------------------------------------
    @staticmethod
    def _kind(key):
        for kind in AXIOM_KINDS:
            if key == kind or key.startswith(kind + "_"):
                return kind
        raise KeyError(key)

    def _batch(self, rows):
        n = rows.shape[0]
        if n <= self.batch_size:
            return rows
        return rows[th.randint(n, (self.batch_size,), device=rows.device)]

    def _negatives(self, batch, kind):
        k = self.num_negatives
        n_ind = len(self._iid)
        neg = batch.repeat(k, 1)
        rand = th.randint(n_ind, (neg.shape[0],), device=neg.device)
        if kind == "class_assertion":
            neg[:, 0] = rand
            known = self._known_ca
        else:
            col = th.randint(2, (neg.shape[0],), device=neg.device) * 2  # 0 or 2
            neg[th.arange(neg.shape[0]), col] = rand
            known = self._known_ra
        keep = [tuple(r) not in known for r in neg.tolist()]
        return neg[th.tensor(keep, dtype=th.bool, device=neg.device)]

    def compute_loss(self, full=False):
        """The E2R objective (eq. (10) with regularisers and negatives) on one random
        batch per axiom kind (``full=True``: on all rows)."""
        m = self.module
        loss = m.structure_loss(self._class_idx)
        for key, rows in self.training_rows.items():
            kind = self._kind(key)
            batch = rows if full else self._batch(rows)
            loss = loss + m(kind, batch).mean()
            use_neg = min(self.num_negatives, self.negative_weight, len(self._iid) - 1) > 0
            if kind in ("class_assertion", "role_assertion") and use_neg:
                neg = self._negatives(batch, kind)
                if len(neg) > 0:
                    loss = loss + self.negative_weight * m(kind, neg, neg=True).mean()
        if self.reg_orth > 0:
            for pairs, rel in ((self.sibling_pairs, False), (self.role_sibling_pairs, True)):
                if len(pairs) > 0:
                    loss = loss + self.reg_orth * m.orthogonality_loss(pairs, rel).mean()
        if self.reg_sep > 0:
            for pairs, rel in ((self.separation_pairs, False),
                               (self.role_separation_pairs, True)):
                if len(pairs) > 0:
                    loss = loss + self.reg_sep * m.separation_loss(
                        pairs, self.sep_gap, rel).mean()
        return loss

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
        with th.no_grad():
            return float(sum(self.module(self._kind(k), r).mean()
                             for k, r in self.training_rows.items()))

    def train(self, epochs=100, verbose=False):
        """Adam on the E2R objective; indicators are clipped to :math:`[0, 1]` after each
        step. An epoch has :math:`\\max_k \\lceil N_k / \\text{batch size} \\rceil` steps.

        :param epochs: Number of epochs.
        :param verbose: Print the loss of every epoch.
        :returns: List of the mean loss of each epoch.
        """
        self.module.train()
        opt = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        sizes = [len(r) for r in self.training_rows.values()] or [1]
        steps = max(1, max(math.ceil(n / self.batch_size) for n in sizes))
        history = []
        for epoch in range(epochs):
            total = 0.0
            for _ in range(steps):
                loss = self.compute_loss()
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.module.project_()
                total += loss.item()
            history.append(total / steps)
            if verbose:
                print(f"epoch {epoch}: {history[-1]:.6f}")
        if self._model_filepath is not None:
            th.save(self.module.state_dict(), self._model_filepath)
        return history

    # -- reasoning and scoring ----------------------------------------------------------
    def concept_indicator(self, cexpr, binary=False):
        """Indicator vector of a class expression (an OWL class expression or a class
        IRI) built from named classes with :math:`\\sqcap` (product), :math:`\\sqcup`
        (max), :math:`\\neg` (complement in the real half) -- Sec. 4, "Membership" task.

        :param binary: Round the named-class indicators at 0.5 first.
        """
        d = self.embed_dim
        if isinstance(cexpr, str):
            idx = th.tensor([self._cid[cexpr]], device=self.device)
            y = self.module.concept(idx)[0].detach()
            return (y > 0.5).to(y) if binary else y
        ctype = cexpr.getClassExpressionType()
        if ctype == ClassExpressionType.OWL_CLASS:
            return self.concept_indicator(_iri(cexpr.asOWLClass()), binary)
        if ctype == ClassExpressionType.OBJECT_INTERSECTION_OF:
            return L.meet(*[self.concept_indicator(c, binary)
                            for c in cexpr.getOperandsAsList()])
        if ctype == ClassExpressionType.OBJECT_UNION_OF:
            return L.join(*[self.concept_indicator(c, binary)
                            for c in cexpr.getOperandsAsList()])
        if ctype == ClassExpressionType.OBJECT_COMPLEMENT_OF:
            y = self.concept_indicator(cexpr.getOperand(), binary)
            out = th.zeros_like(y)
            out[:d] = 1 - y[:d]
            return out
        raise NotImplementedError(f"E2R has no subspace for {ctype}.")

    def score_membership(self, cexpr, individuals=None, binary=False):
        """Residual length :math:`\\|\\overline{y} \\odot x_a\\|` of each individual for
        the class expression ``cexpr`` (lower = better; Sec. 5).

        :param individuals: IRIs; defaults to all individuals in index order.
        :rtype: numpy.ndarray
        """
        y = self.concept_indicator(cexpr, binary)
        if individuals is None:
            idx = th.arange(len(self._iid), device=self.device)
        else:
            idx = th.tensor([self._iid[i] for i in individuals], device=self.device)
        with th.no_grad():
            x = self.module.entity(idx)
            return self.module.membership_residual(y.unsqueeze(0), x).cpu().numpy()

    def rank_members(self, cexpr, binary=False):
        """Individual IRIs sorted by increasing residual for ``cexpr``."""
        names = sorted(self._iid, key=self._iid.get)
        scores = self.score_membership(cexpr, binary=binary)
        return [names[i] for i in np.argsort(scores, kind="stable")]

    def score_role_membership(self, prop, pairs):
        """Residual lengths of the pair vectors of ``pairs`` (list of ``(subject,
        object)`` IRIs) for the relation ``prop`` (lower = better).

        :rtype: numpy.ndarray
        """
        with th.no_grad():
            z = self.module.relation(th.tensor([self._rid[prop]], device=self.device))
            s = th.tensor([self._iid[a] for a, _ in pairs], device=self.device)
            o = th.tensor([self._iid[b] for _, b in pairs], device=self.device)
            xpq = L.pair_vector(self.module.entity(s), self.module.entity(o), self.embed_dim)
            return self.module.membership_residual(z, xpq).cpu().numpy()

    @th.no_grad()
    def axiom_costs(self, gci_name, rows):
        """Cost of every candidate axiom in ``rows`` (model indices), lower is more
        plausible: the inclusion loss of eq. (5) for ``gci0`` rows ``[C, D]`` and the
        residual length of Sec. 5 for ``class_assertion`` rows ``[a, C]``."""
        rows = rows.to(self.device)
        if gci_name == "gci0":
            return L.inclusion_loss(self.module.concept(rows[:, 0]),
                                    self.module.concept(rows[:, 1]))
        if gci_name == "class_assertion":
            return self.module.membership_residual(self.module.concept(rows[:, 1]),
                                                   self.module.entity(rows[:, 0]))
        from mowl.evaluation.model_scoring import UnsupportedAxiomForm
        raise UnsupportedAxiomForm(f"E2R has no score for {gci_name!r} axioms.")

    def score_subsumption(self, sub, sup, binary=False):
        """Inclusion loss of eq. (5) for class expressions ``sub``, ``sup`` (``0`` iff
        the subspace of ``sub`` lies in that of ``sup`` for 0/1 indicators)."""
        return float(L.inclusion_loss(self.concept_indicator(sub, binary),
                                      self.concept_indicator(sup, binary)))

    def discretise(self, threshold=0.5):
        """0/1 indicators of all concepts and relations and the entity vectors, as numpy
        arrays ``(concepts, relations, entities)`` in index order."""
        m = self.module
        with th.no_grad():
            c = m.concept(th.arange(m.c_embedding.weight.shape[0], device=self.device))
            r = m.r_embedding.weight
            return ((c > threshold).cpu().numpy().astype(int),
                    (r > threshold).cpu().numpy().astype(int),
                    m.e_embedding.weight.detach().cpu().numpy())

    def violations(self, tol=1e-3, threshold=0.5):
        """Training axioms (as ``(kind, row)``) that the discretised embedding violates,
        decided with set operations on the axis supports and residual norms, independently
        of the loss functions. Membership holds if the residual is at most ``tol``
        times the norm of the vector (and the vector is not ``0``)."""
        yc, zr, x = self.discretise(threshold)
        d = self.embed_dim
        supp = [frozenset(np.flatnonzero(v)) for v in yc]
        rsupp = [frozenset(np.flatnonzero(v)) for v in zr]

        def inside(s, v):
            n = np.linalg.norm(v)
            out = [t for t in range(len(v)) if t not in s]
            return n > 0 and np.linalg.norm(v[out]) <= tol * n

        bad = []
        for key, rows in self.training_rows.items():
            kind = self._kind(key)
            for row in rows.tolist():
                if kind == "class_assertion":
                    ok = inside(supp[row[1]], x[row[0]])
                elif kind == "role_assertion":
                    v = np.concatenate([x[row[0]][:d], x[row[2]][:d]])
                    ok = inside(rsupp[row[1]], v)
                elif kind == "concept_inclusion":
                    ok = supp[row[0]] <= supp[row[1]]
                elif kind == "concept_conjunction":
                    ok = supp[row[0]] == frozenset.intersection(*[supp[i] for i in row[1:]])
                elif kind == "concept_disjunction":
                    ok = supp[row[0]] == frozenset.union(*[supp[i] for i in row[1:]])
                elif kind == "concept_negation":
                    ok = supp[row[0]] == frozenset(range(d)) - supp[row[1]]
                elif kind == "concept_disjointness":
                    ok = not (supp[row[0]] & supp[row[1]])
                elif kind == "role_inclusion":
                    ok = rsupp[row[0]] <= rsupp[row[1]]
                elif kind == "range":
                    ok = not ({t - d for t in rsupp[row[0]] if t >= d} & supp[row[1]])
                else:
                    continue
                if not ok:
                    bad.append((kind, tuple(row)))
        return bad
