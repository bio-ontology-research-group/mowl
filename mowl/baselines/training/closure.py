"""Approximate deductive closure of a normalised :math:`\\mathcal{EL}^{++}` ontology with
respect to the seven normal forms, as defined by DELE [mashkova2024dele]_ (Section 5,
Algorithms 1 and 2, soundness proofs in Appendix F) and by its NeSy 2024 predecessor
[mashkova2024nesy]_ (Section 4.3, Appendix J).

What is computed
----------------

The deductive closure of an ontology is infinite, but restricted to the normal forms over
a finite signature it is finite (DELE, Section 2.3). DELE does not compute it exactly. It
computes the subsumption hierarchy between concept names with a reasoner (the "bold"
axioms of Algorithm 1, *transitive closure of the ELK inferences*) and then applies a fixed
set of sound but incomplete syntactic rules to the normalised axioms. This module
implements that approximation. For concept names :math:`A, B, E`, role names :math:`r`
and the reflexive-transitive role hierarchy :math:`\\sqsubseteq^*` of the role inclusions:

- **GCI0** :math:`A \\sqsubseteq B`: exactly the reasoner's hierarchy (complete; this
  includes :math:`A \\sqsubseteq A`, :math:`A \\sqsubseteq \\top`, :math:`\\bot \\sqsubseteq
  A`, and :math:`A \\sqsubseteq B` for every unsatisfiable :math:`A`).
- **GCI0-BOT** :math:`A \\sqsubseteq \\bot`: :math:`A` is unsatisfiable (complete).
- **GCI1** :math:`A \\sqcap B \\sqsubseteq E`: Algorithm 1 (for an asserted
  :math:`A_0 \\sqcap B_0 \\sqsubseteq E_0`, with :math:`A \\sqsubseteq A_0`, :math:`B
  \\sqsubseteq B_0`, :math:`E_0 \\sqsubseteq E`; asserted GCI1-BOT axioms are treated as
  GCI1 axioms with :math:`E_0 = \\bot`, which is Algorithm 1's rule
  :math:`A \\sqcap B \\sqsubseteq \\bot \\Rightarrow A \\sqcap B \\sqsubseteq E` combined with
  its subclass rule) and Algorithm 2 (:math:`A \\sqsubseteq E` or :math:`B \\sqsubseteq E`,
  which covers rules 8-10, 12, 13 of Appendix F, and :math:`A` or :math:`B`
  unsatisfiable, rule 9).
- **GCI1-BOT** :math:`A \\sqcap B \\sqsubseteq \\bot`: the GCI1 closure with :math:`E =
  \\bot`.
- **GCI2** :math:`A \\sqsubseteq \\exists r.B`: Algorithm 1 (for an asserted or derived
  :math:`A_0 \\sqsubseteq \\exists r_0.B_0`: :math:`A \\sqsubseteq A_0`, :math:`r_0
  \\sqsubseteq^* r`, :math:`B_0 \\sqsubseteq B`; the role-chain rule
  :math:`A \\sqsubseteq \\exists r.B, B \\sqsubseteq \\exists r'.E, r \\circ r' \\sqsubseteq s
  \\Rightarrow A \\sqsubseteq \\exists s.E` is iterated to a fixpoint) and Algorithm 2
  (:math:`A` unsatisfiable, which includes :math:`\\bot \\sqsubseteq \\exists r.B`).
- **GCI3** :math:`\\exists r.A \\sqsubseteq B`: Algorithm 1 (for an asserted
  :math:`\\exists r_0.A_0 \\sqsubseteq B_0`: :math:`r \\sqsubseteq^* r_0`, :math:`A
  \\sqsubseteq A_0`, :math:`B_0 \\sqsubseteq B`; asserted GCI3-BOT axioms count as GCI3
  axioms with :math:`B_0 = \\bot`) and Algorithm 2 (:math:`B = \\top`).
- **GCI3-BOT** :math:`\\exists r.A \\sqsubseteq \\bot`: the GCI3 closure with :math:`B =
  \\bot`.

Deviations from the papers, all of them sound (every rule follows from the semantics):
the conjunction is treated as commutative (:math:`A \\sqcap B` and :math:`B \\sqcap A`);
the role-chain rule is iterated to a fixpoint instead of being applied once; an
unsatisfiable filler makes :math:`\\exists r.A \\sqsubseteq B` true for all :math:`B`;
Algorithm 2's restriction :math:`B \\neq \\bot` in :math:`A \\sqsubseteq \\bot \\Rightarrow
A \\sqsubseteq \\exists r.B` is dropped (:math:`\\exists r.\\bot \\equiv \\bot`). The closure
is still incomplete for GCI1, GCI2 and GCI3, as in the papers.

Reasoners
---------

The subsumption hierarchy is computed over the *normalised* axioms (so it also covers the
auxiliary classes introduced by normalisation) by one of:

- ``"canonical"``: :class:`mowl.geometric.completion.Completion`, a sound and complete
  pure-Python EL++ saturation (Kazakov et al. 2014, Table 2);
- ``"elk"``: the ELK reasoner through the OWL API (as in the papers).

Both are complete for subsumption between concept names, so they give the same closure;
the test-suite checks this.
"""

import hashlib
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch as th

from mowl.ontology.normalize import ELNormalizer, extract_role_axioms
from mowl.owlapi.defaults import TOP, BOT

logger = logging.getLogger(__name__)

#: Normal forms supported by :class:`DeductiveClosure`.
NORMAL_FORMS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot")
_GCI_KEYS = NORMAL_FORMS

_ALL = None  # marker: "every class" (used for unsatisfiable classes)


def _gci_strings(gcis):
    """Yield the normalised axioms as tuples of IRIs, tagged with their normal form."""
    for a in gcis.get("gci0", []):
        yield ("gci0", a.subclass, a.superclass)
    for a in gcis.get("gci0_bot", []):
        yield ("gci0", a.subclass, BOT)
    for key in ("gci1", "gci1_bot"):
        for a in gcis.get(key, []):
            sup = a.superclass if key == "gci1" else BOT
            yield ("gci1", a.left_subclass, a.right_subclass, sup)
    for a in gcis.get("gci2", []):
        yield ("gci2", a.subclass, a.object_property, a.filler)
    for key in ("gci3", "gci3_bot"):
        for a in gcis.get(key, []):
            sup = a.superclass if key == "gci3" else BOT
            yield ("gci3", a.object_property, a.filler, sup)


class DeductiveClosure:
    """Approximate deductive closure of a normalised ontology (see the module documentation
    for exactly what is included).

    The closure lives in the index space of the given dictionaries: queries take and return
    the indices used by a model or an evaluator. Classes and object properties occurring in
    the axioms but missing from the dictionaries get internal indices beyond them (they can
    still take part in derivations, but they are never returned).

    :param normalized_gcis: Output of :meth:`mowl.ontology.normalize.ELNormalizer.normalize`.
    :type normalized_gcis: dict
    :param class_index_dict: Class IRI to index. ``owl:Thing`` and ``owl:Nothing`` should be
        in it (mOWL datasets always contain them).
    :type class_index_dict: dict
    :param object_property_index_dict: Object property IRI to index.
    :type object_property_index_dict: dict
    :param role_inclusions: :class:`~mowl.ontology.normalize.RoleInclusion` axioms.
    :type role_inclusions: list, optional
    :param role_chains: :class:`~mowl.ontology.normalize.RoleChain` axioms. Only binary chains
        are used.
    :type role_chains: list, optional
    :param reasoner: ``"canonical"`` (default) or ``"elk"``.
    :type reasoner: str, optional
    :param cache_dir: If given, the subsumption hierarchy is cached there, keyed by a hash of
        the normalised axioms and the reasoner, and reused when the same axioms come again.
    :type cache_dir: str, optional
    """

    def __init__(self, normalized_gcis, class_index_dict, object_property_index_dict,
                 role_inclusions=(), role_chains=(), reasoner="canonical", cache_dir=None):
        if reasoner not in ("canonical", "elk"):
            raise ValueError(f"reasoner must be 'canonical' or 'elk', got {reasoner!r}")
        self.reasoner = reasoner
        self.cache_dir = cache_dir
        self._gcis = normalized_gcis
        self._role_inclusions = list(role_inclusions)
        self._role_chains = [c for c in role_chains if len(c.sub_chain) == 2]
        if len(self._role_chains) < len(list(role_chains)):
            logger.warning("Role chains longer than two properties are ignored by the "
                           "deductive closure.")

        self._axioms = list(_gci_strings(normalized_gcis))

        # ---- index spaces -----------------------------------------------------------
        self.class_index_dict = dict(class_index_dict)
        self.object_property_index_dict = dict(object_property_index_dict)
        self._cls = dict(class_index_dict)
        self._rel = dict(object_property_index_dict)
        self._n_classes = len(class_index_dict)
        self._n_relations = len(object_property_index_dict)

        def cid(iri):
            if iri not in self._cls:
                self._cls[iri] = max(self._cls.values(), default=-1) + 1
            return self._cls[iri]

        def rid(iri):
            if iri not in self._rel:
                self._rel[iri] = max(self._rel.values(), default=-1) + 1
            return self._rel[iri]

        self.top = cid(TOP)
        self.bot = cid(BOT)
        for ax in self._axioms:
            if ax[0] == "gci0":
                cid(ax[1]), cid(ax[2])
            elif ax[0] == "gci1":
                cid(ax[1]), cid(ax[2]), cid(ax[3])
            elif ax[0] == "gci2":
                cid(ax[1]), rid(ax[2]), cid(ax[3])
            else:
                rid(ax[1]), cid(ax[2]), cid(ax[3])
        for ri in self._role_inclusions:
            rid(ri.sub_property), rid(ri.super_property)
        for rc in self._role_chains:
            rid(rc.sub_chain[0]), rid(rc.sub_chain[1]), rid(rc.super_property)
        self._id_to_class = {v: k for k, v in self._cls.items()}

        self._compute_role_hierarchy()
        self._compute_hierarchy()
        self._index_axioms()

        self._memo = {}

    # ------------------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------------------

    @classmethod
    def from_ontology(cls, ontology, class_index_dict=None, object_property_index_dict=None,
                      reasoner="canonical", cache_dir=None, normalized_gcis=None):
        """Normalise ``ontology`` and compute its closure.

        :param ontology: The ontology (for instance the training ontology, or the union of
            training and testing ontologies).
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        :param class_index_dict: Index space; built from the ontology's classes (sorted IRIs,
            as in :class:`mowl.datasets.Dataset`) if ``None``.
        :param object_property_index_dict: As above, for object properties.
        :param normalized_gcis: Already normalised axioms of ``ontology``, to avoid
            normalising twice.
        :rtype: DeductiveClosure
        """
        if normalized_gcis is None:
            normalized_gcis = ELNormalizer().normalize(ontology)
        if class_index_dict is None:
            classes = {str(c.toStringID()) for c in ontology.getClassesInSignature()}
            classes = sorted(classes | {TOP, BOT})
            class_index_dict = {c: i for i, c in enumerate(classes)}
        if object_property_index_dict is None:
            props = sorted(str(p.toStringID())
                           for p in ontology.getObjectPropertiesInSignature())
            object_property_index_dict = {p: i for i, p in enumerate(props)}
        inclusions, chains = extract_role_axioms(ontology)
        return cls(normalized_gcis, class_index_dict, object_property_index_dict,
                   role_inclusions=inclusions, role_chains=chains, reasoner=reasoner,
                   cache_dir=cache_dir)

    @classmethod
    def from_model(cls, model, reasoner="canonical", cache_dir=None):
        """Closure of the training ontology of an
        :class:`~mowl.base_models.elmodel.EmbeddingELModel`, in the model's index space
        (named and auxiliary classes). The model's normalisation is reused.

        :rtype: DeductiveClosure
        """
        inclusions, chains = extract_role_axioms(model.dataset.ontology)
        return cls(model._training_normalized_gcis, model.class_index_dict,
                   model.object_property_index_dict, role_inclusions=inclusions,
                   role_chains=chains, reasoner=reasoner, cache_dir=cache_dir)

    # ------------------------------------------------------------------------------
    # Hierarchies
    # ------------------------------------------------------------------------------

    def _compute_role_hierarchy(self):
        sups = defaultdict(set)
        for r in self._rel.values():
            sups[r].add(r)
        for ri in self._role_inclusions:
            sups[self._rel[ri.sub_property]].add(self._rel[ri.super_property])
        changed = True
        while changed:
            changed = False
            for r in list(sups):
                new = set()
                for s in sups[r]:
                    new |= sups[s]
                if not new <= sups[r]:
                    sups[r] |= new
                    changed = True
        self._role_sups = {r: frozenset(s) for r, s in sups.items()}
        subs = defaultdict(set)
        for r, ss in self._role_sups.items():
            for s in ss:
                subs[s].add(r)
        self._role_subs = {r: frozenset(s) for r, s in subs.items()}
        self._chains = defaultdict(set)
        for rc in self._role_chains:
            self._chains[(self._rel[rc.sub_chain[0]], self._rel[rc.sub_chain[1]])].add(
                self._rel[rc.super_property])

    def _fingerprint(self):
        h = hashlib.sha256()
        h.update(self.reasoner.encode())
        for line in sorted("\t".join(a) for a in self._axioms):
            h.update(line.encode())
            h.update(b"\n")
        for ri in sorted(f"{r.sub_property}\t{r.super_property}"
                         for r in self._role_inclusions):
            h.update(ri.encode())
        for rc in sorted(f"{'|'.join(r.sub_chain)}\t{r.super_property}"
                         for r in self._role_chains):
            h.update(rc.encode())
        h.update("\n".join(sorted(self._cls)).encode())
        return h.hexdigest()

    def _compute_hierarchy(self):
        """Fill ``self._sups`` (class id -> frozenset of superclass ids, or ``_ALL`` for an
        unsatisfiable class)."""
        sups_iri = None
        cache_file = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir,
                                      f"closure_hierarchy_{self._fingerprint()}.json")
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    sups_iri = json.load(f)
                logger.info("Loaded subsumption hierarchy from %s", cache_file)

        if sups_iri is None:
            if self.reasoner == "canonical":
                sups_iri = self._hierarchy_canonical()
            else:
                sups_iri = self._hierarchy_elk()
            if cache_file is not None:
                with open(cache_file, "w") as f:
                    json.dump(sups_iri, f)

        self._sups = {}
        self.unsatisfiable = {self.bot}
        for iri, cid in self._cls.items():
            sup = sups_iri.get(iri)
            if sup is None:  # class not occurring in the axioms
                self._sups[cid] = _ALL if cid == self.bot else frozenset({cid, self.top})
            elif BOT in sup or cid == self.bot:
                self._sups[cid] = _ALL
                self.unsatisfiable.add(cid)
            else:
                ids = {self._cls[s] for s in sup if s in self._cls}
                self._sups[cid] = frozenset(ids | {cid, self.top})

        subs = defaultdict(set)
        all_ids = list(self._cls.values())
        for cid, sup in self._sups.items():
            for s in (all_ids if sup is _ALL else sup):
                subs[s].add(cid)
        self._subs = {c: frozenset(s) for c, s in subs.items()}

    def _hierarchy_canonical(self):
        from mowl.geometric.completion import Completion
        from mowl.geometric.normal_forms import NF1, NF2, NF3, NF4, NF6, NF7
        axioms = []
        for ax in self._axioms:
            if ax[0] == "gci0":
                axioms.append(NF1(ax[1], ax[2]))
            elif ax[0] == "gci1":
                axioms.append(NF2(ax[1], ax[2], ax[3]))
            elif ax[0] == "gci2":
                axioms.append(NF3(ax[1], ax[2], ax[3]))
            else:
                axioms.append(NF4(ax[1], ax[2], ax[3]))
        for ri in self._role_inclusions:
            axioms.append(NF6(ri.sub_property, ri.super_property))
        for rc in self._role_chains:
            axioms.append(NF7(rc.sub_chain[0], rc.sub_chain[1], rc.super_property))
        result = Completion(axioms, set(self._cls), set(self._rel)).saturate()
        return {c: sorted(result.sub_super.get(c, {c, TOP})) for c in self._cls}

    def _hierarchy_elk(self):
        from java.util import HashSet
        from org.semanticweb.elk.owlapi import ElkReasonerFactory
        from mowl.owlapi import OWLAPIAdapter
        adapter = OWLAPIAdapter()
        axioms = HashSet()
        for key in _GCI_KEYS:
            for a in self._gcis.get(key, []):
                axioms.add(a.owl_axiom)
        for a in self._role_inclusions + self._role_chains:
            axioms.add(a.owl_axiom)
        ont = adapter.create_ontology("http://mowl.borg/training/closure")
        adapter.owl_manager.addAxioms(ont, axioms)
        reasoner = ElkReasonerFactory().createReasoner(ont)
        reasoner.precomputeInferences()
        out = {}
        try:
            signature = {str(c.toStringID()) for c in ont.getClassesInSignature()}
            for iri in self._cls:
                if iri not in signature and iri not in (TOP, BOT):
                    continue
                owl_class = adapter.create_class(iri)
                if iri == BOT or not reasoner.isSatisfiable(owl_class):
                    out[iri] = [BOT]
                    continue
                sups = {str(c.toStringID())
                        for c in reasoner.getSuperClasses(owl_class, False).getFlattened()}
                sups |= {str(c.toStringID())
                         for c in reasoner.getEquivalentClasses(owl_class).getEntities()}
                sups |= {iri, TOP}
                out[iri] = sorted(sups)
        finally:
            reasoner.dispose()
        return out

    # ------------------------------------------------------------------------------
    # Axiom indexes
    # ------------------------------------------------------------------------------

    def _index_axioms(self):
        c, r = self._cls, self._rel
        self._gci1_pairs = defaultdict(lambda: defaultdict(set))
        edges = set()
        self._gci3_by_filler = defaultdict(set)
        for ax in self._axioms:
            if ax[0] == "gci1":
                a, b, e = c[ax[1]], c[ax[2]], c[ax[3]]
                self._gci1_pairs[a][b].add(e)
                self._gci1_pairs[b][a].add(e)
            elif ax[0] == "gci2":
                edges.add((c[ax[1]], r[ax[2]], c[ax[3]]))
            elif ax[0] == "gci3":
                self._gci3_by_filler[c[ax[2]]].add((r[ax[1]], c[ax[3]]))
        if self._chains:
            edges = self._chain_fixpoint(edges)
        self._edges = edges
        self._edges_by_sub = defaultdict(set)
        for a, rr, b in edges:
            self._edges_by_sub[a].add((rr, b))

    def _chain_fixpoint(self, edges):
        """Algorithm 1, role-chain rule, iterated: A ⊑ ∃r.B, B' ⊑ ∃r'.E with B ⊑ B',
        r ⊑* r1, r' ⊑* r2, r1 ∘ r2 ⊑ s  ==>  A ⊑ ∃s.E."""
        edges = set(edges)
        changed = True
        while changed:
            changed = False
            by_sub = defaultdict(set)
            for a, rr, b in edges:
                by_sub[a].add((rr, b))
            new = set()
            for a, r1_, b in edges:
                sups_b = self._sups[b]
                mids = by_sub.keys() if sups_b is _ALL else sups_b
                for x in mids:
                    for r2_, e in by_sub.get(x, ()):
                        for r1 in self._role_sups.get(r1_, (r1_,)):
                            for r2 in self._role_sups.get(r2_, (r2_,)):
                                for s in self._chains.get((r1, r2), ()):
                                    if (a, s, e) not in edges:
                                        new.add((a, s, e))
            if new:
                edges |= new
                changed = True
        return edges

    # ------------------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------------------

    def is_unsatisfiable(self, c):
        """``True`` iff class index ``c`` is entailed to be unsatisfiable."""
        return c in self.unsatisfiable

    def superclasses(self, c):
        """Indices :math:`B` with :math:`c \\sqsubseteq B` entailed, or ``None`` meaning all
        classes (``c`` unsatisfiable)."""
        s = self._sups.get(c, frozenset({c, self.top}))
        return None if s is _ALL else s

    def _sup_iter(self, c):
        s = self._sups.get(c, frozenset({c, self.top}))
        return self._cls.values() if s is _ALL else s

    def _union_sups(self, ids):
        out = set()
        for i in ids:
            s = self._sups.get(i, frozenset({i, self.top}))
            if s is _ALL:
                return None
            out |= s
        return out

    def _gci1_tails(self, a, b):
        key = ("gci1", a, b)
        if key in self._memo:
            return self._memo[key]
        if a in self.unsatisfiable or b in self.unsatisfiable:
            out = None
        else:
            e0s = set()
            pairs = self._gci1_pairs
            for a0 in self._sup_iter(a):
                partners = pairs.get(a0)
                if not partners:
                    continue
                sb = self._sups.get(b, frozenset({b, self.top}))
                for b0, es in partners.items():
                    if b0 in sb:
                        e0s |= es
            out = self._union_sups(e0s | {a, b})
        self._memo[key] = out
        return out

    def _gci2_tails(self, a, r):
        key = ("gci2", a, r)
        if key in self._memo:
            return self._memo[key]
        if a in self.unsatisfiable:
            out = None
        else:
            subs_r = self._role_subs.get(r, frozenset({r}))
            b0s = set()
            for a0 in self._sup_iter(a):
                for r0, b0 in self._edges_by_sub.get(a0, ()):
                    if r0 in subs_r:
                        b0s.add(b0)
            out = self._union_sups(b0s)
        self._memo[key] = out
        return out

    def _gci3_tails(self, r, a):
        key = ("gci3", r, a)
        if key in self._memo:
            return self._memo[key]
        if a in self.unsatisfiable:
            out = None
        else:
            sups_r = self._role_sups.get(r, frozenset({r}))
            b0s = set()
            for a0 in self._sup_iter(a):
                for r0, b0 in self._gci3_by_filler.get(a0, ()):
                    if r0 in sups_r:
                        b0s.add(b0)
            out = self._union_sups(b0s)
            if out is not None:
                out.add(self.top)
        self._memo[key] = out
        return out

    def entailed_tails(self, gci_name, head, relation=None):
        """The last-column values :math:`X` that make the axiom with the given head entailed:
        superclasses for ``gci0`` (:math:`h \\sqsubseteq X`), fillers for ``gci2``
        (:math:`h \\sqsubseteq \\exists r.X`). Returns ``None`` when every class qualifies.

        :param gci_name: ``"gci0"`` or ``"gci2"``.
        :param head: Index of the subclass.
        :param relation: Index of the role (``gci2`` only).
        :rtype: set or None
        """
        if gci_name == "gci0":
            return self.superclasses(head)
        if gci_name == "gci2":
            return self._gci2_tails(head, relation)
        raise ValueError(f"entailed_tails supports 'gci0' and 'gci2', got {gci_name!r}")

    def entails(self, gci_name, row):
        """Whether the axiom given as a row of indices (in the column layout of
        :class:`mowl.datasets.el.ELDataset`) is in the closure.

        :param gci_name: One of :data:`NORMAL_FORMS`.
        :param row: Sequence of ints.
        :rtype: bool
        """
        if self._memo_size() > 5_000_000:
            self._memo.clear()
        row = [int(x) for x in row]
        if gci_name in ("gci0", "gci0_bot"):
            tails = self.superclasses(row[0])
        elif gci_name in ("gci1", "gci1_bot"):
            tails = self._gci1_tails(row[0], row[1])
        elif gci_name == "gci2":
            tails = self._gci2_tails(row[0], row[1])
        elif gci_name in ("gci3", "gci3_bot"):
            tails = self._gci3_tails(row[0], row[1])
        else:
            raise ValueError(f"Unknown normal form {gci_name!r}; expected one of "
                             f"{NORMAL_FORMS}.")
        return tails is None or row[-1] in tails

    def _memo_size(self):
        return len(self._memo)

    def contains(self, gci_name, data):
        """Vectorised :meth:`entails` over the rows of a tensor.

        :param data: Tensor of shape ``(n, k)`` in the layout of ``gci_name``.
        :type data: :class:`torch.Tensor`
        :rtype: :class:`torch.Tensor` of bool, shape ``(n,)``
        """
        rows = data.detach().cpu().tolist()
        mask = [self.entails(gci_name, r) for r in rows]
        return th.tensor(mask, dtype=th.bool, device=data.device)

    # ------------------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------------------

    def axioms(self, gci_name, heads=None, relations=None, tails=None,
               include_trivial=False):
        """Enumerate closure axioms of ``gci0``, ``gci2`` or ``gci3`` as an index tensor.

        Only indices of the given dictionaries (not internal auxiliary ones) are returned.
        The closure can be very large, so restrict it with ``heads`` / ``relations`` /
        ``tails`` when possible.

        :param heads: Allowed first-class indices (subclass for gci0/gci2, filler for gci3).
        :param relations: Allowed role indices (gci2/gci3).
        :param tails: Allowed last-column indices.
        :param include_trivial: Also return axioms that hold for every ontology or follow
            from unsatisfiability alone: :math:`A \\sqsubseteq A`, :math:`A \\sqsubseteq
            \\top`, :math:`\\exists r.A \\sqsubseteq \\top`, axioms whose subclass (or filler)
            is :math:`\\bot` or unsatisfiable. Defaults to ``False``.
        :rtype: :class:`torch.Tensor` of shape ``(n, 2)`` or ``(n, 3)``
        """
        n, m = self._n_classes, self._n_relations
        known_c = range(n)
        heads = known_c if heads is None else [int(h) for h in heads]
        tail_set = None if tails is None else {int(t) for t in tails}
        rels = range(m) if relations is None else [int(r) for r in relations]

        def keep_tail(t):
            return t < n and (tail_set is None or t in tail_set)

        rows = []
        if gci_name == "gci0":
            for h in heads:
                if h in self.unsatisfiable:
                    if include_trivial:
                        rows.extend([h, t] for t in known_c if keep_tail(t))
                    continue
                for t in self._sups.get(h, frozenset({h, self.top})):
                    if not include_trivial and (t == h or t == self.top):
                        continue
                    if keep_tail(t):
                        rows.append([h, t])
            width = 2
        elif gci_name == "gci2":
            for h in heads:
                if h in self.unsatisfiable:
                    if include_trivial:
                        rows.extend([h, r, t] for r in rels for t in known_c if keep_tail(t))
                    continue
                for r in rels:
                    tails_hr = self._gci2_tails(h, r)
                    for t in tails_hr:
                        if keep_tail(t):
                            rows.append([h, r, t])
            width = 3
        elif gci_name == "gci3":
            for a in heads:
                for r in rels:
                    tails_ra = self._gci3_tails(r, a)
                    if tails_ra is None:
                        if include_trivial:
                            rows.extend([r, a, t] for t in known_c if keep_tail(t))
                        continue
                    for t in tails_ra:
                        if not include_trivial and t == self.top:
                            continue
                        if keep_tail(t):
                            rows.append([r, a, t])
            width = 3
        else:
            raise ValueError("axioms() enumerates 'gci0', 'gci2' and 'gci3' only; use "
                             "entails()/contains() for the other normal forms.")
        if not rows:
            return th.empty((0, width), dtype=th.long)
        return th.tensor(np.asarray(rows, dtype=np.int64))

    def __repr__(self):
        return (f"DeductiveClosure(reasoner={self.reasoner!r}, classes={self._n_classes}, "
                f"relations={self._n_relations}, unsatisfiable={len(self.unsatisfiable)})")
