"""The CatE lattice of concept descriptions as an indexed structure.

:class:`CatELattice` builds the preorder :math:`(\\tilde C, \\sqsubseteq)` of
Zhapa-Camacho and Hoehndorf (arXiv:2305.07163v3, Section 3) with the existing
:class:`~mowl.projection.categorical.model.CategoricalProjector`, optionally completes and
saturates it (:mod:`mowl.projection.categorical.saturation`), and freezes it into integer
node ids and an edge list that an order embedding can be trained on. It also answers
which node represents a named class, an individual :math:`\\{a\\}` or an existential
restriction :math:`\\exists r.D`, which is what scoring queries needs.

Node names are unambiguous: a concept name or individual is its IRI, a complex concept
description is its OWL API functional-syntax string (e.g.
``ObjectIntersectionOf(<http://A> <http://B>)``), and the role-related elements of the
existential construction are ``Role(r|∃r.C)``, ``Domain(r|∃r.C)``, ``Codomain(r|∃r.C)``
and ``Not(Domain(...))``. (The projector's own string output removes parentheses, so
different descriptions can be rendered identically; it is not used.)
"""

import logging

import torch as th

from mowl.owlapi import OWLAPIAdapter
from mowl.owlapi.defaults import BOT, TOP
from mowl.projection.categorical.edge import Node
from mowl.projection.categorical.model import CategoricalProjector, Graph
from mowl.projection.categorical import saturation as sat

from org.semanticweb.owlapi.model import ClassExpressionType as CT
from org.semanticweb.owlapi.model import OWLObjectProperty

logger = logging.getLogger(__name__)


class UnseenConceptError(LookupError):
    """A concept description, individual or role is not an element of the lattice.

    CatE embeds only the elements of the lattice built from the training ontology; it
    cannot embed a concept description that is not one of them (paper, Section 5). Add
    such descriptions before training (e.g. with ``existentials`` of
    :class:`CatELattice`).
    """


def node_name(node):
    """Unambiguous string name of a lattice node (see module docstring)."""
    if node.relation is None:
        oc = node.owl_class
        if not oc.isAnonymous():
            return str(oc.asOWLClass().toStringID())
        return str(oc.toString())
    rel = str(node.relation.toStringID())
    base = rel if node.owl_class is None else f"{rel}|{node.owl_class.toString()}"
    if node.domain:
        base = f"Domain({base})"
        if node.negated_domain:
            base = f"Not({base})"
        return base
    if node.codomain:
        return f"Codomain({base})"
    return base if node.owl_class is None else f"Role({base})"


class CatELattice:
    """The lattice of concept descriptions of an ALC ontology, indexed for embedding.

    Construction (paper, Section 3) is done by the existing
    :class:`~mowl.projection.categorical.model.CategoricalProjector`; see the CatE card
    for how it relates to the paper's rules. Afterwards, in this order:

    1. every class in ``classes`` and individual in ``individuals`` that is not yet an
       element is added (with :math:`\\bot \\sqsubseteq X \\sqsubseteq \\top`), so that
       every named class can be a ranking candidate; an individual whose IRI is also a
       class (punning, as in the ORE1 data) *is* that class's element, as in the
       reference code, whose nodes are identified by strings;
    2. for every role IRI ``r`` and class IRI ``D`` in ``existentials``, the axiom
       :math:`\\exists r.D \\sqsubseteq \\top` is projected, which adds the element
       :math:`\\exists r.D` with its existential construction (the paper's treatment of
       the PPI task, Section 4.4: "we added the relationships
       :math:`\\bot \\sqsubseteq \\exists interacts.p_i` and
       :math:`\\exists interacts.p_i \\sqsubseteq \\top`");
    3. if ``complete_negation``, :func:`~mowl.projection.categorical.saturation.complete_negation`;
    4. ``saturation_steps`` iterations of the saturation rules ``rules`` (default: the
       paper's S1), then, if ``transitive_closure``, the transitive closure (Tr). The
       paper's "S1-Tr" is ``saturation_steps=1, transitive_closure=True``.

    :param ontology: Training ontology.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param classes: Class IRIs to include as elements (e.g. ``dataset.classes.as_str``).
    :param individuals: Individual IRIs to include as elements.
    :param existentials: Mapping from role IRI to an iterable of filler class IRIs.
    :param complete_negation: Add the full negation diagram (Figure 2c) for complex
        operands.
    :param saturation_steps: Iterations of the saturation rules (0: none; ``None``: until
        a fixed point).
    :param rules: Saturation rules, names from
        :data:`mowl.projection.categorical.saturation.RULES`.
    :param transitive_closure: Add the transitive closure after saturation (rule 4):
        ``True`` for the paper's Tr as the reference code computes it (paths through
        :math:`\\bot` and :math:`\\top` are not followed), ``"full"`` for the complete
        closure (see :func:`~mowl.projection.categorical.saturation.rule_transitivity`).
    :param obvious_morphisms: Keep :math:`\\bot \\sqsubseteq X` and
        :math:`X \\sqsubseteq \\top` for *complex* :math:`X` (the paper keeps them for
        every element). The released training graphs of the ORE1, GO and FoodOn
        benchmarks drop them (``remove_obvious_morphisms.py`` of the reference code);
        pass ``False`` to match those graphs element for element.
    """

    def __init__(self, ontology, classes=(), individuals=(), existentials=None,
                 complete_negation=True, saturation_steps=0, rules=sat.S1,
                 transitive_closure=False, obvious_morphisms=True):
        self.ontology = ontology
        #: IRI -> graph id of an element of the *other* kind: an individual that is
        #: also a class (punning, as in the ORE1 data) shares the class's element, as
        #: in the reference code, whose node names are strings.
        self._individual_alias = {}
        self._class_alias = {}
        self.complete_negation = complete_negation
        self.saturation_steps = saturation_steps
        self.rules = tuple(rules)
        self.transitive_closure = transitive_closure
        self.obvious_morphisms = obvious_morphisms
        self.stats = {}
        self.adapter = OWLAPIAdapter()
        self.projector = CategoricalProjector("owl")
        self.graph = Graph()
        self.projector.graph = self.graph
        self._build(classes, individuals, existentials or {})
        self._freeze()

    # ------------------------------------------------------------------ building
    def _project(self):
        # Same loop as CategoricalProjector.project, without converting the result to
        # an edge list (its "owl" output fails on class assertions, and its "str"
        # output is ambiguous, see the module docstring).
        for cls in self.ontology.getClassesInSignature():
            self.graph.add_node(Node(owl_class=cls))
        for axiom in self.ontology.getAxioms(True):
            edges = self.projector.process_axiom(axiom)
            if edges:
                self.graph.add_all_edges(*list(edges))

    def _build(self, classes, individuals, existentials):
        self._project()
        self.stats["projected_edges"] = self.num_morphisms()
        for iri in classes:
            node = Node(owl_class=self.adapter.create_class(str(iri)))
            if node not in self.graph.node_to_id:
                self.graph.add_node(node)
            elif not self.graph.id_to_node[self.graph.node_to_id[node]].is_individual:
                pass  # already a class element
            else:
                self._class_alias[str(iri)] = self.graph.node_to_id[node]
        for iri in individuals:
            node = Node(owl_class=self.adapter.create_class(str(iri)), is_individual=True)
            if node not in self.graph.node_to_id:
                self.graph.add_node(node)
            elif self.graph.id_to_node[self.graph.node_to_id[node]].is_individual:
                pass  # already an individual element
            else:
                self._individual_alias[str(iri)] = self.graph.node_to_id[node]
        if existentials:
            top = self.adapter.create_class(TOP)
            for role_iri, fillers in existentials.items():
                role = self.adapter.create_object_property(str(role_iri))
                for filler in fillers:
                    ex = self.adapter.create_object_some_values_from(
                        role, self.adapter.create_class(str(filler)))
                    axiom = self.adapter.create_subclass_of(ex, top)
                    self.graph.add_all_edges(*list(self.projector.process_axiom(axiom)))
        if not self.obvious_morphisms:
            self.stats["obvious_removed"] = self._remove_obvious()
        if self.complete_negation:
            self.stats["negation_completion_added"] = sat.complete_negation(self.graph)
        if self.saturation_steps is None or self.saturation_steps > 0:
            self.stats["saturation"] = sat.saturate(self.graph, rules=self.rules,
                                                    max_iterations=self.saturation_steps)
        if self.transitive_closure:
            rules = sat.TR_FULL if self.transitive_closure == "full" else sat.TR
            self.stats["transitive_closure"] = sat.saturate(
                self.graph, rules=rules, max_iterations=1)

    def num_morphisms(self):
        return sum(len(v) for v in self.graph.out_edges.values())

    def _remove_obvious(self):
        """Remove :math:`\\bot \\sqsubseteq X` and :math:`X \\sqsubseteq \\top` for
        complex :math:`X`, as ``remove_obvious_morphisms.py`` of the reference code
        does to produce the released training graphs."""
        bot = Node(owl_class=self.adapter.create_class(BOT))
        top = Node(owl_class=self.adapter.create_class(TOP))

        def trivial(node):
            """Everything that is not a named class or individual: a complex concept
            description or one of the role elements (their reference rendering
            contains a space, which is what remove_obvious_morphisms.py keys on)."""
            if node.relation is not None:
                return True
            return node.owl_class is not None and node.owl_class.isAnonymous()

        removed = 0
        for dst in list(self.graph.out_edges.get(bot, ())):
            if trivial(dst):
                self.graph.out_edges[bot].discard(dst)
                self.graph.in_edges[dst].discard(bot)
                removed += 1
        for src, dsts in list(self.graph.out_edges.items()):
            if trivial(src) and top in dsts:
                dsts.discard(top)
                self.graph.in_edges[top].discard(src)
                removed += 1
        return removed

    def _freeze(self):
        nodes = list(self.graph.node_to_id.keys())
        names = [node_name(n) for n in nodes]
        order = sorted(range(len(nodes)), key=lambda i: names[i])
        self.nodes = [nodes[i] for i in order]
        self.names = [names[i] for i in order]
        self.node_to_id = {n: i for i, n in enumerate(self.nodes)}
        self.name_to_id = {}
        for i, name in enumerate(self.names):
            if name in self.name_to_id:
                raise RuntimeError(f"Two lattice elements share the name {name!r}.")
            self.name_to_id[name] = i

        edges = []
        for src, dsts in self.graph.out_edges.items():
            s = self.node_to_id[src]
            for dst in dsts:
                edges.append((s, self.node_to_id[dst]))
        edges.sort()
        self.edges = th.tensor(edges, dtype=th.long).reshape(-1, 2)

        self.class_to_id, self.individual_to_id = {}, {}
        roles = {str(p.toStringID()) for p in self.ontology.getObjectPropertiesInSignature()}
        existentials = []
        for i, node in enumerate(self.nodes):
            if node.relation is not None:
                roles.add(str(node.relation.toStringID()))
                continue
            oc = node.owl_class
            if not oc.isAnonymous():
                if node.is_individual:
                    self.individual_to_id[self.names[i]] = i
                else:
                    self.class_to_id[self.names[i]] = i
            elif oc.getClassExpressionType() == CT.OBJECT_SOME_VALUES_FROM:
                prop = oc.getProperty()
                filler = self.node_to_id.get(Node(owl_class=oc.getFiller()))
                if isinstance(prop, OWLObjectProperty) and filler is not None:
                    existentials.append((str(prop.toStringID()), filler, i))
        for iri in (TOP, BOT):
            if iri not in self.class_to_id:
                self.class_to_id[iri] = self.name_to_id[iri]
        for iri, gid in self._individual_alias.items():
            self.individual_to_id.setdefault(
                str(iri), self.node_to_id[self.graph.id_to_node[gid]])
        for iri, gid in self._class_alias.items():
            self.class_to_id.setdefault(
                str(iri), self.node_to_id[self.graph.id_to_node[gid]])
        self.role_to_id = {r: i for i, r in enumerate(sorted(roles))}
        self.existential_to_id = {(self.role_to_id[r], f): i for r, f, i in existentials}
        self.stats.update(nodes=len(self.nodes), edges=len(self.edges))

    # ------------------------------------------------------------------- queries
    @property
    def bot_id(self):
        return self.class_to_id[BOT]

    @property
    def top_id(self):
        return self.class_to_id[TOP]

    def __len__(self):
        return len(self.nodes)

    def node_id(self, expression):
        """Id of the element for an OWL class expression.

        :raises UnseenConceptError: If the expression is not an element of the lattice.
        """
        idx = self.node_to_id.get(Node(owl_class=expression))
        if idx is None:
            raise UnseenConceptError(
                f"{expression} is not an element of the CatE lattice. CatE embeds only "
                f"the concept descriptions of the training ontology (and their "
                f"sub-descriptions); add it before training, e.g. with an axiom "
                f"'{expression} SubClassOf owl:Thing' or the 'existentials' option.")
        return idx

    def individual_id(self, iri):
        """Id of :math:`\\{a\\}` for an individual IRI.

        :raises UnseenConceptError: If the individual is not an element.
        """
        iri = str(iri)
        if iri not in self.individual_to_id:
            raise UnseenConceptError(f"Individual {iri} is not an element of the lattice.")
        return self.individual_to_id[iri]

    def existential_id(self, role_iri, filler_id):
        """Id of :math:`\\exists r.D` for a role IRI and the id of the filler element.

        :raises UnseenConceptError: If :math:`\\exists r.D` is not an element.
        """
        key = (self.role_to_id.get(str(role_iri), -1), int(filler_id))
        if key not in self.existential_to_id:
            raise UnseenConceptError(
                f"∃{role_iri}.{self.names[int(filler_id)]} is not an element of the "
                f"CatE lattice; add it before training with the 'existentials' option.")
        return self.existential_to_id[key]

    def has_edge(self, src, dst):
        """Whether the morphism ``src -> dst`` (ids) is in the lattice."""
        return sat.has_edge(self.graph, self.nodes[src], self.nodes[dst])

    def successors(self, idx):
        """Ids of the elements :math:`D` with a morphism ``idx -> D``."""
        return sorted(self.node_to_id[d] for d in self.graph.out_edges[self.nodes[idx]])
