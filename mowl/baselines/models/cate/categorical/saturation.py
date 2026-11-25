"""Partial saturation of the CatE lattice of concept descriptions.

Implements the saturation rules of Zhapa-Camacho and Hoehndorf, *Lattice-based ALC
ontology embeddings with saturation* (arXiv:2305.07163v3, Section 3.1), on the
:class:`~mowl.baselines.models.cate.categorical.model.Graph` built by
:class:`~mowl.baselines.models.cate.categorical.model.CategoricalProjector`. For elements
:math:`C, D, E` of the lattice and roles :math:`r`:

.. math::

   (1)\\quad & C \\sqsubseteq \\neg D \\Rightarrow D \\sqsubseteq \\neg C \\\\
   (2)\\quad & C \\sqcap D \\sqsubseteq \\bot \\Rightarrow C \\sqsubseteq \\neg D \\\\
   (3)\\quad & \\top \\sqsubseteq C \\sqcup D \\Rightarrow \\neg C \\sqsubseteq D \\\\
   (4)\\quad & C \\sqsubseteq D,\\ D \\sqsubseteq E \\Rightarrow C \\sqsubseteq E \\\\
   (5)\\quad & C \\sqsubseteq D \\Rightarrow \\exists r.C \\sqsubseteq \\exists r.D

The paper calls rules (1)-(3) together **S1** and the transitive closure (4) **Tr**
(Section 4.5); it does not use rule (5) because it introduces :math:`2|\\tilde R|` new
elements per morphism.

Deviations and readings (also stated in the CatE catalogue card):

* Rule (3) is printed with :math:`\\bot \\sqsubseteq C \\sqcup D` as premise. That premise
  holds for every :math:`C, D` and would make the rule unsound; it is read as
  :math:`\\top \\sqsubseteq C \\sqcup D`, as in the reference implementation.
* For an n-ary conjunction (disjunction) rules (2) and (3) are applied to every split of
  the operands into two non-empty groups (as the reference implementation does); above
  :data:`MAX_SPLIT_OPERANDS` operands only the splits "one operand vs. the rest" are used.
* Rule (5) is implemented only in a restricted form that does not create elements: it
  adds :math:`\\exists r.C \\sqsubseteq \\exists r.D` when both existential restrictions are
  already in the lattice. The unrestricted rule does not terminate (it can nest
  existentials indefinitely).
* Rules are applied to concept elements only (nodes of the object category). Individuals
  :math:`\\{a\\}` and role elements are not negated.

:func:`complete_negation` adds the morphisms of the negation diagram of the paper's
Figure 2c (:math:`C \\sqcap \\neg C \\to C, \\neg C` and :math:`C, \\neg C \\to C \\sqcup \\neg C`)
for complex :math:`C`; the existing projector adds them only when :math:`C` is a concept
name.

The existing module :mod:`mowl.baselines.models.cate.categorical.model` also has a ``saturate``
method (``lemma_6`` and others). It is not used here: its contraposition step adds only
the last morphism computed in a loop (a variable carried over from the previous loop), so
rule (1) is applied only partially.
"""

from itertools import combinations

import networkx as nx

from mowl.owlapi import OWLAPIAdapter
from mowl.baselines.models.cate.categorical.edge import Edge, Node
from mowl.baselines.models.cate.categorical.model import bot_node, top_node

from org.semanticweb.owlapi.model import ClassExpressionType as CT

#: Label of the morphisms added by saturation.
SATURATION_RELATION = "http://saturation"
#: Above this number of operands, n-ary conjunctions and disjunctions are only split as
#: "one operand vs. the rest" in rules (2) and (3).
MAX_SPLIT_OPERANDS = 6

_adapter = OWLAPIAdapter()


def is_concept(node):
    """Whether ``node`` is a concept element of the lattice (not an individual, not a
    role)."""
    return node.in_object_category()


def _splits(operands):
    """All ordered splits of ``operands`` into two non-empty groups."""
    n = len(operands)
    if n > MAX_SPLIT_OPERANDS:
        for i in range(n):
            rest = operands[:i] + operands[i + 1:]
            yield [operands[i]], rest
            yield rest, [operands[i]]
        return
    idx = range(n)
    for k in range(1, n):
        for left in combinations(idx, k):
            right = [operands[i] for i in idx if i not in left]
            yield [operands[i] for i in left], right


def _conjunction(ops):
    return ops[0] if len(ops) == 1 else _adapter.create_object_intersection_of(*ops)


def _disjunction(ops):
    return ops[0] if len(ops) == 1 else _adapter.create_object_union_of(*ops)


def _complement(expr):
    return _adapter.create_complement_of(expr)


def _plain(node):
    """Concept node with a class expression (not a domain/codomain element)."""
    has_class = node.relation is None and node.owl_class is not None
    return has_class and not node.is_individual


def _node_list(graph):
    return list(graph.node_to_id.keys())


def has_edge(graph, src, dst):
    """Whether the morphism ``src -> dst`` is in the graph."""
    return dst in graph.out_edges.get(src, ())


# ---------------------------------------------------------------------------------------
# Rules. Each returns a set of (src, dst) node pairs implied by the current graph.
# ---------------------------------------------------------------------------------------

def rule_contraposition(graph):
    """Rule (1): :math:`C \\sqsubseteq \\neg D \\Rightarrow D \\sqsubseteq \\neg C`."""
    out = set()
    for neg in _node_list(graph):
        if not is_concept(neg) or not neg.is_negated():
            continue
        d = neg.get_operand()
        for c in list(graph.in_edges.get(neg, ())):
            if c == neg or c == bot_node or not is_concept(c):
                continue
            out.add((d, c.negate()))
    return out


def rule_disjointness(graph):
    """Rule (2): :math:`C \\sqcap D \\sqsubseteq \\bot \\Rightarrow C \\sqsubseteq \\neg D`."""
    out = set()
    for node in _node_list(graph):
        if not _plain(node) or not node.is_intersection():
            continue
        if bot_node not in graph.out_edges.get(node, ()):
            continue
        ops = list(node.owl_class.getOperandsAsList())
        for left, right in _splits(ops):
            out.add((Node(owl_class=_conjunction(left)),
                     Node(owl_class=_complement(_conjunction(right)))))
    return out


def rule_excluded_middle(graph):
    """Rule (3), read with premise :math:`\\top \\sqsubseteq C \\sqcup D`:
    :math:`\\neg C \\sqsubseteq D`."""
    out = set()
    for node in _node_list(graph):
        if not _plain(node) or not node.is_union():
            continue
        if top_node not in graph.in_edges.get(node, ()):
            continue
        ops = list(node.owl_class.getOperandsAsList())
        for left, right in _splits(ops):
            out.add((Node(owl_class=_complement(_disjunction(left))),
                     Node(owl_class=_disjunction(right))))
    return out


def rule_transitivity(graph, through_bounds=False):
    """Rule (4): the transitive closure of the morphisms.

    With ``through_bounds=False`` (the paper's Tr as run by the reference code) paths are
    not followed through :math:`\\bot` and :math:`\\top`: an element below
    :math:`\\bot` (e.g. :math:`C \\sqcap \\neg C`) does not get morphisms to every
    element, nor an element above :math:`\\top` from every element. These morphisms are
    numerous (number of such elements times the size of the lattice).
    ``through_bounds=True`` computes the full closure.
    """
    g = nx.DiGraph()
    for src, dsts in graph.out_edges.items():
        for dst in dsts:
            if dst == src:
                continue
            if not through_bounds and (src == bot_node or dst == top_node):
                continue
            g.add_edge(src, dst)
    closure = nx.transitive_closure(g, reflexive=False)
    return {(s, d) for s, d in closure.edges() if s != d}


def rule_full_transitivity(graph):
    """Rule (4) with paths through :math:`\\bot` and :math:`\\top`
    (see :func:`rule_transitivity`)."""
    return rule_transitivity(graph, through_bounds=True)


def rule_existential_monotonicity(graph):
    """Rule (5), restricted to existing elements: if :math:`C \\sqsubseteq D` and both
    :math:`\\exists r.C` and :math:`\\exists r.D` are in the lattice, add
    :math:`\\exists r.C \\sqsubseteq \\exists r.D`."""
    by_role = {}
    for node in _node_list(graph):
        if _plain(node) and node.owl_class.getClassExpressionType() == CT.OBJECT_SOME_VALUES_FROM:
            prop = node.owl_class.getProperty()
            filler = Node(owl_class=node.owl_class.getFiller())
            by_role.setdefault(prop, {})[filler] = node
    out = set()
    for fillers in by_role.values():
        for c, ex_c in fillers.items():
            for d in graph.out_edges.get(c, ()):
                if d != c and d in fillers:
                    out.add((ex_c, fillers[d]))
    return out


RULES = {
    "contraposition": rule_contraposition,
    "disjointness": rule_disjointness,
    "excluded_middle": rule_excluded_middle,
    "transitivity": rule_transitivity,
    "full_transitivity": rule_full_transitivity,
    "existential_monotonicity": rule_existential_monotonicity,
}

#: Rule groups of the paper (Section 4.5).
S1 = ("contraposition", "disjointness", "excluded_middle")
TR = ("transitivity",)
#: Transitive closure including paths through ⊥ and ⊤.
TR_FULL = ("full_transitivity",)


def add_morphisms(graph, pairs, relation=SATURATION_RELATION):
    """Add the morphisms ``pairs`` that are not yet in ``graph``.

    :return: Number of morphisms added.
    """
    added = 0
    for src, dst in pairs:
        if src == dst or has_edge(graph, src, dst):
            continue
        graph.add_edge(Edge(src, relation, dst))
        added += 1
    return added


def saturate(graph, rules=S1, max_iterations=1):
    """Apply saturation rules to ``graph`` in place.

    One iteration computes the consequences of all ``rules`` on the current graph and
    adds them. Iteration stops after ``max_iterations`` iterations or at a fixed point.
    Rules (1)-(4) and the restricted rule (5) create at most the negations of existing
    elements and conjunctions/disjunctions of operands of existing ones, so a fixed
    point exists; ``max_iterations=None`` iterates until it is reached.

    :param graph: The graph of a :class:`CategoricalProjector`.
    :type graph: :class:`mowl.baselines.models.cate.categorical.model.Graph`
    :param rules: Names of rules in :data:`RULES` (default: the paper's S1).
    :type rules: tuple of str
    :param max_iterations: Maximum number of iterations, or ``None``.
    :type max_iterations: int or None
    :return: ``{"iterations": ..., "added": [...], "fixed_point": bool, "nodes": ...,
        "edges": ...}`` where ``added`` lists the morphisms added per iteration.
    :rtype: dict
    """
    unknown = [r for r in rules if r not in RULES]
    if unknown:
        raise ValueError(f"Unknown saturation rules {unknown}; known: {sorted(RULES)}.")
    added = []
    fixed_point = False
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        iteration += 1
        new = set()
        for name in rules:
            new |= RULES[name](graph)
        n = add_morphisms(graph, new)
        added.append(n)
        if n == 0:
            fixed_point = True
            break
    return {"iterations": iteration, "added": added, "fixed_point": fixed_point,
            "nodes": len(graph.node_to_id),
            "edges": sum(len(v) for v in graph.out_edges.values())}


def complete_negation(graph):
    """Add the negation diagram of the paper's Figure 2c for every negated element
    :math:`\\neg C` whose operand is a complex concept description:
    :math:`C \\sqcap \\neg C \\sqsubseteq C, \\neg C, \\bot` and
    :math:`C, \\neg C, \\top \\sqsubseteq C \\sqcup \\neg C`.

    The projector adds these morphisms itself when :math:`C` is a concept name, and only
    :math:`C \\sqcap \\neg C \\sqsubseteq \\bot`, :math:`\\top \\sqsubseteq C \\sqcup \\neg C`
    otherwise.

    :return: Number of morphisms added.
    """
    pairs = set()
    for neg in _node_list(graph):
        if not _plain(neg) or not neg.is_negated():
            continue
        operand = neg.owl_class.getOperand()
        if operand.getClassExpressionType() == CT.OWL_CLASS:
            continue
        c = Node(owl_class=operand)
        meet = Node(owl_class=_adapter.create_object_intersection_of(operand, neg.owl_class))
        join = Node(owl_class=_adapter.create_object_union_of(operand, neg.owl_class))
        pairs |= {(meet, c), (meet, neg), (meet, bot_node),
                  (c, join), (neg, join), (top_node, join)}
    return add_morphisms(graph, pairs, relation="http://negation")
