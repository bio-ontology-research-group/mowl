"""Scoring axioms as the triples they are projected to.

A model of the triples of a projected graph has a cost for a triple ``(head, relation, tail)``
of graph ids (``module.triple_cost``). An axiom is scored through the edge the projection turns
it into, so any such model, whatever its entity model, relation model and interaction are, is
evaluated on axioms by one part: :class:`AxiomTriples`.

* ``gci0`` rows ``[C, D]``: the edge ``C -subclassof-> D``;
* ``gci2`` rows ``[C, r, D]``: the edge ``C -r-> D``;
* ``class_assertion`` rows ``[a, C]``: the edge ``a -type-> C``;
* ``object_property_assertion`` rows ``[a, r, b]``: the edge ``a -r-> b``.
"""

import torch as th

__all__ = ["AxiomTriples"]

SCORED_FORMS = ("gci0", "gci2", "class_assertion", "object_property_assertion")


def _index_map(index_dict, size, to_id):
    out = th.full((max(1, size),), -1, dtype=th.long)
    for iri, i in index_dict.items():
        out[i] = to_id.get(iri, -1)
    return out


class AxiomTriples:
    """The cost of axioms (rows of indices of the dataset's classes, object properties and
    individuals) as the cost of the triples of the graph they are projected to.

    :param classes, individuals, roles: IRI -> index of the dataset.
    :param sizes: ``(number of classes, individuals, roles)`` of the dataset.
    :param nodes, relations: IRI of a node (label of an edge) -> id in the graph.
    :param subclass, type_label: Labels of the edges of ``C ⊑ D`` and of ``C(a)``.
    :param inverse: Whether relation ``r`` of the graph is ``2r`` in the model (a model with
        inverse triples).
    """

    def __init__(self, classes, individuals, roles, sizes, nodes, relations,
                 subclass, type_label, inverse=False):
        self.class_map = _index_map(classes, sizes[0], nodes)
        self.individual_map = _index_map(individuals, sizes[1], nodes)
        self.role_map = _index_map(roles, sizes[2], relations)
        self.subclass = relations.get(subclass, -1)
        self.type = relations.get(type_label, -1)
        self.inverse = bool(inverse)

    def triples(self, gci_name, rows):
        """Heads, relations and tails (graph ids, ``-1`` where unknown) of the edges of
        ``rows``."""
        if gci_name not in SCORED_FORMS:
            from mowl.evaluation.model_scoring import UnsupportedAxiomForm
            raise UnsupportedAxiomForm(f"No triple score for {gci_name!r} axioms.")
        rows = rows.cpu()
        if gci_name == "gci0":
            head, tail = self.class_map[rows[:, 0]], self.class_map[rows[:, 1]]
            relation = th.full_like(head, self.subclass)
        elif gci_name == "gci2":
            head, tail = self.class_map[rows[:, 0]], self.class_map[rows[:, 2]]
            relation = self.role_map[rows[:, 1]]
        elif gci_name == "class_assertion":
            head, tail = self.individual_map[rows[:, 0]], self.class_map[rows[:, 1]]
            relation = th.full_like(head, self.type)
        else:
            head, tail = self.individual_map[rows[:, 0]], self.individual_map[rows[:, 2]]
            relation = self.role_map[rows[:, 1]]
        if self.inverse:
            relation = th.where(relation >= 0, 2 * relation, relation)
        return head, relation, tail

    @th.no_grad()
    def costs(self, module, gci_name, rows, device="cpu", chunk=2 ** 16):
        """Cost of every axiom in ``rows``, lower is more plausible: minus the score the
        model gives to the edge the axiom is projected to. An axiom whose role is not an edge
        label of the graph (and a class assertion when the projection has no type edges) gets
        an infinite cost. A class or individual that occurs in no edge is a node with an
        untrained vector."""
        head, relation, tail = self.triples(gci_name, rows)
        known = (head >= 0) & (tail >= 0) & (relation >= 0)
        out = th.full((len(head),), float("inf"))
        if known.any():
            triples = th.stack([head[known], relation[known], tail[known]], dim=1)
            module.eval()
            costs = []
            for s in range(0, len(triples), chunk):
                part = triples[s:s + chunk].to(device)
                costs.append(module.triple_cost(part[:, 0], part[:, 1], part[:, 2])
                             .reshape(-1).cpu())
            out[known] = th.cat(costs).float()
        return out.to(rows.device)
