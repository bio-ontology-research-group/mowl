"""Order embedding of the CatE lattice (Zhapa-Camacho and Hoehndorf,
arXiv:2305.07163v3, Section 3.2)."""

import torch as th
import torch.nn as nn


class CatEModule(nn.Module):
    """One vector :math:`f_\\theta(X) \\in \\mathbb{R}^n` per element of the lattice,
    ordered by the product order (:math:`a \\preceq b` iff :math:`a_i \\le b_i` for all
    :math:`i`, the paper's Theorem 1).

    The score of a morphism :math:`A \\sqsubseteq B` is the order violation of Eq. 6,

    .. math:: s(A, B) = \\lVert \\max(0, f_\\theta(A) - f_\\theta(B)) \\rVert_p ,

    which is zero iff :math:`f_\\theta(A) \\preceq f_\\theta(B)` (the paper uses
    :math:`p = 2`). Lower scores mean "more likely".

    Calling the module with ``(rows, gci_name)`` follows the conventions of
    :mod:`mowl.evaluation.el_scoring`, with lattice element ids in place of class
    indices: ``"gci0"`` rows ``[C, D]`` and ``"edge"`` rows ``[A, B]`` are scored as
    :math:`s(C, D)`; ``"class_assertion"`` rows ``[a, C]`` as :math:`s(\\{a\\}, C)`;
    ``"gci2"`` rows ``[C, r, D]`` (``r`` a role index) as :math:`s(C, \\exists r.D)`,
    which needs the element :math:`\\exists r.D` (registered with
    :meth:`set_existentials`).

    :param num_nodes: Number of lattice elements.
    :param embed_dim: Dimension :math:`n`.
    :param p: Order of the norm in Eq. 6.
    """

    def __init__(self, num_nodes, embed_dim, p=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.p = p
        self.node_embedding = nn.Embedding(num_nodes, embed_dim)
        # Xavier-uniform initialisation, the PyKEEN TransE default used by the reference
        # implementation.
        nn.init.xavier_uniform_(self.node_embedding.weight)
        self.register_buffer("existential_keys", th.zeros(0, dtype=th.long))
        self.register_buffer("existential_nodes", th.zeros(0, dtype=th.long))

    def set_existentials(self, role_ids, filler_ids, node_ids):
        """Register the elements :math:`\\exists r.D`: ``node_ids[i]`` is the element for
        role ``role_ids[i]`` and filler element ``filler_ids[i]``."""
        role_ids = th.as_tensor(role_ids, dtype=th.long).reshape(-1)
        filler_ids = th.as_tensor(filler_ids, dtype=th.long).reshape(-1)
        node_ids = th.as_tensor(node_ids, dtype=th.long).reshape(-1)
        keys = role_ids * self.num_nodes + filler_ids
        keys, order = th.sort(keys)
        dev = self.node_embedding.weight.device
        self.existential_keys = keys.to(dev)
        self.existential_nodes = node_ids[order].to(dev)

    def existential_ids(self, role_ids, filler_ids):
        """Element ids of :math:`\\exists r.D`; ``-1`` where it is not an element."""
        keys = role_ids.long() * self.num_nodes + filler_ids.long()
        if len(self.existential_keys) == 0:
            return th.full_like(keys, -1)
        last = len(self.existential_keys) - 1
        pos = th.searchsorted(self.existential_keys, keys).clamp(max=last)
        found = self.existential_keys[pos] == keys
        return th.where(found, self.existential_nodes[pos], th.full_like(keys, -1))

    def embed(self, ids):
        return self.node_embedding(ids)

    def order_violation(self, sub, sup):
        """:math:`s(A, B)` of Eq. 6 for element ids ``sub`` (:math:`A`) and ``sup``
        (:math:`B`) of the same shape."""
        diff = self.node_embedding(sub) - self.node_embedding(sup)
        return th.linalg.vector_norm(th.relu(diff), ord=self.p, dim=-1)

    def forward(self, rows, gci_name="edge"):
        rows = rows.long()
        if gci_name in ("edge", "gci0", "class_assertion"):
            return self.order_violation(rows[:, 0], rows[:, 1])
        if gci_name == "gci2":
            ex = self.existential_ids(rows[:, 1], rows[:, 2])
            if (ex < 0).any():
                from mowl.projection.categorical.lattice import UnseenConceptError
                bad = rows[ex < 0][0].tolist()
                raise UnseenConceptError(
                    f"The existential restriction of row {bad} (role index {bad[1]}, "
                    f"filler element {bad[2]}) is not an element of the CatE lattice. "
                    f"CatE scores C ⊑ ∃r.D only when ∃r.D was added before training "
                    f"(model option 'existentials').")
            return self.order_violation(rows[:, 0], ex)
        raise NotImplementedError(
            f"CatE scores 'edge', 'gci0', 'class_assertion' and 'gci2' rows, not "
            f"{gci_name!r}. Other axioms are scored through elements of the lattice "
            f"(CatE.score).")
