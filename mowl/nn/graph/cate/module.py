"""Order embedding of the CatE lattice (Zhapa-Camacho and Hoehndorf,
arXiv:2305.07163v3, Section 3.2), built from parts."""

from mowl.representation.library.hierarchy_geometry import ExistentialElements, LatticeGeometry
from mowl.representation.library.hierarchy_methods import cate
from mowl.representation.library.hierarchy_modules import adopt


class CatEModule(ExistentialElements, LatticeGeometry):
    """One vector :math:`f_\\theta(X) \\in \\mathbb{R}^n` per element of the lattice,
    ordered by the product order (:math:`a \\preceq b` iff :math:`a_i \\le b_i` for all
    :math:`i`, the paper's Theorem 1): the combination that
    :func:`mowl.representation.library.hierarchy_methods.cate` names (read
    ``module.parts``). The class model is
    :class:`~mowl.representation.library.hierarchy_classes.OrderPoints`; the compound
    classes :math:`\\exists r.D` and :math:`C \\sqcap D` are elements with embeddings of
    their own, looked up (:class:`~mowl.representation.library.hierarchy_geometry.
    LatticeGeometry`); the measure is the order violation of Eq. 6,

    .. math:: s(A, B) = \\lVert \\max(0, f_\\theta(A) - f_\\theta(B)) \\rVert_p ,

    which is zero iff :math:`f_\\theta(A) \\preceq f_\\theta(B)` (the paper uses
    :math:`p = 2`). Lower scores mean "more likely".

    Calling the module with ``(rows, gci_name)`` follows the conventions of
    :mod:`mowl.evaluation.el_scoring`, with lattice element ids in place of class
    indices, and returns one score per row, shape ``(B,)``: ``"gci0"`` rows ``[C, D]`` and
    ``"edge"`` rows ``[A, B]`` are scored as :math:`s(C, D)`; ``"class_assertion"`` rows
    ``[a, C]`` as :math:`s(\\{a\\}, C)`; ``"gci2"`` rows ``[C, r, D]`` (``r`` a role index) as
    :math:`s(C, \\exists r.D)`, which needs the element :math:`\\exists r.D` (registered with
    :meth:`set_existentials`). With ``neg=True`` the cost of a corruption is the hinge
    :math:`\\max(0, \\gamma - s)` (``margin`` :math:`\\gamma`).

    The earlier hand-written module is kept for comparison as
    ``mowl.baselines.nn.graph.cate.CatEModule``.

    :param num_nodes: Number of lattice elements.
    :param embed_dim: Dimension :math:`n`.
    :param p: Order of the norm in Eq. 6.
    :param margin: The margin :math:`\\gamma` of the cost of a corruption (CatE's loss has
        it in the model; default 1).
    """

    def __init__(self, num_nodes, embed_dim, p=2, margin=1.0):
        adopt(self, cate(num_nodes, embed_dim=embed_dim, margin=margin, p=p))
        self.embed_dim = embed_dim
        self.p = p
        self._init_existentials(num_nodes)

    @property
    def node_embedding(self):
        """The table of the vectors of the elements."""
        return self.class_embed

    def embed(self, ids):
        return self.node_embedding(ids)

    def order_violation(self, sub, sup):
        """:math:`s(A, B)` of Eq. 6 for element ids ``sub`` (:math:`A`) and ``sup``
        (:math:`B`) of the same shape; the result has that shape."""
        return self.measure.energy(self.classes(sub), self.classes(sup)).squeeze(-1)

    def forward(self, rows, gci_name="edge", neg=False):
        rows = rows.long()
        if gci_name in ("edge", "gci0", "class_assertion"):
            return self.cost("gci0", rows[:, :2], neg).reshape(-1)
        if gci_name == "gci2":
            if neg:
                raise NotImplementedError("CatE has no negative cost of 'gci2' rows.")
            region = self.exists(rows[:, 1], self.classes(rows[:, 2]))
            return self.inside(self.classes(rows[:, 0]), region).reshape(-1)
        raise NotImplementedError(
            f"CatE scores 'edge', 'gci0', 'class_assertion' and 'gci2' rows, not "
            f"{gci_name!r}. Other axioms are scored through elements of the lattice "
            f"(CatE.score).")
