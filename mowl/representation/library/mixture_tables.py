"""The constructed part of the canonical mixture: the canonical model as sparse element sets.

The canonical model of an EL++ ontology is *constructed* (by completion) and read, without
ever building a dense matrix over its elements, as

========================  ======================================================================
source                    :class:`CanonicalCompletionSource` (``"canonical completion"``): the
                          canonical model as the completion builds it
                          (:class:`~mowl.geometric.completion.CanonicalModel`), not expanded
                          into a :class:`~mowl.representation.construction.FiniteModel`
reading                   :class:`MembershipReading` (``"membership"``): the elements of the
                          satisfiable named classes and of the individuals (the support
                          :math:`S`), every class as the sparse set of its elements, every role
                          as a sparse relation over :math:`S`
class model               :class:`MembershipClasses`: a class is its set of elements
                          (:class:`SparseElementSets`, no parameters)
relation model            :class:`CanonicalRoles`: a role is its set of pairs of elements
========================  ======================================================================

The sparse tables (:class:`MembershipTable`) are read off a
:class:`~mowl.geometric.membership.MembershipStructure`, the neutral restriction of the
canonical model to :math:`S` (``M[x, C]`` iff the element ``x`` belongs to ``C``). The trained
parts of the canonical mixture (:mod:`~mowl.representation.library.mixture_individuals`) put
distributions over the same elements.

Class regions are compared exactly (costs 0.0 or 1.0); inclusion of element sets is
entailment for satisfiable named classes (whose elements are in :math:`S`). The top class has
no element of its own in :math:`S`, so over :math:`S` it can lie inside a class that covers
every element of :math:`S` without the ontology entailing it. :math:`\\exists r.C` is the
preimage over :math:`S` (pairs that leave :math:`S`, through auxiliary elements, are not
read).
"""

import torch as th
import torch.nn as nn

from mowl.representation.class_models import Classes
from mowl.representation.construction import (FiniteModel, Guarantee, ReadModel, Reading,
                                              register_reading, register_source)
from mowl.representation.library.construct_sources import (CanonicalModelSource,
                                                           canonical_model_of)
from mowl.representation.regions import ConstructorUnavailable, Region
from mowl.representation.relation_models import Relations

__all__ = ["MembershipTable", "SparseElementSets", "MembershipClasses", "CanonicalRoles",
           "CanonicalCompletionSource", "MembershipReading", "MembershipRead",
           "membership_from_sizes"]


def _sparse_float(indices, size):
    values = th.ones((indices.shape[1],), dtype=th.float32)
    return th.sparse_coo_tensor(indices, values, size=size, dtype=th.float32).coalesce()


def _offsets(counts):
    out = th.zeros((counts.numel() + 1,), dtype=th.long)
    out[1:] = th.cumsum(counts, dim=0)
    return out


class MembershipTable(nn.Module):
    """The membership relation ``M`` (elements × classes) of a constructed model, sparse.

    Holds ``M`` by rows (the classes of every element, ``crow`` and ``types``), by columns
    (the elements of every class, ``ccol`` and ``members``), as sorted keys
    ``element * n_classes + class`` (for membership tests by binary search) and as a sparse
    float matrix ``(n_classes, |S|)``. Nothing of size ``|S| x |S|`` or
    ``|S| x n_classes`` is dense. The buffers are constructed, not learned: they are not
    part of the ``state_dict``.

    :param structure: A :class:`~mowl.geometric.membership.MembershipStructure`.
    """

    def __init__(self, structure):
        super().__init__()
        self.structure = structure
        n_elements, n_classes = structure.M.shape
        self.n_elements, self.n_classes = int(n_elements), int(n_classes)
        m = structure.M.coalesce()
        keep = m.values().bool().cpu()
        rows = m.indices()[0].cpu()[keep]
        cols = m.indices()[1].cpu()[keep]
        by_row = th.argsort(rows * self.n_classes + cols, stable=True)
        by_col = th.argsort(cols * self.n_elements + rows, stable=True)
        self.register_buffer("keys", (rows * self.n_classes + cols)[by_row], persistent=False)
        self.register_buffer("crow", _offsets(th.bincount(rows, minlength=self.n_elements)),
                             persistent=False)
        self.register_buffer("types", cols[by_row], persistent=False)
        self.register_buffer("ccol", _offsets(th.bincount(cols, minlength=self.n_classes)),
                             persistent=False)
        self.register_buffer("members", rows[by_col], persistent=False)
        self.register_buffer("matrix_t", _sparse_float(
            th.stack([cols[by_col], rows[by_col]]) if cols.numel()
            else th.empty((2, 0), dtype=th.long), (self.n_classes, self.n_elements)),
            persistent=False)

    def contains(self, elements, classes):
        """``(B,)`` boolean: ``M[elements, classes]`` (False for ids out of range)."""
        valid = ((elements >= 0) & (elements < self.n_elements) & (classes >= 0)
                 & (classes < self.n_classes))
        if self.keys.numel() == 0 or not bool(valid.any()):
            return th.zeros_like(valid, dtype=th.bool)
        key = (elements.clamp(0, max(self.n_elements - 1, 0)) * self.n_classes
               + classes.clamp(0, max(self.n_classes - 1, 0)))
        pos = th.searchsorted(self.keys, key)
        last = self.keys.numel() - 1
        return valid & (pos <= last) & (self.keys.index_select(0, pos.clamp(max=last)) == key)

    def types_of(self, element):
        """The classes of one element (a 1-D long tensor)."""
        return self.types[int(self.crow[element]):int(self.crow[element + 1])]

    def members_of(self, cls):
        """The elements of one class, sorted (a 1-D long tensor)."""
        return self.members[int(self.ccol[cls]):int(self.ccol[cls + 1])]

    def column_sizes(self, classes):
        """``(B,)`` number of elements of every class."""
        return self.ccol.index_select(0, classes + 1) - self.ccol.index_select(0, classes)

    def columns_t(self, classes):
        """Sparse float ``(len(classes), |S|)``: the element sets of ``classes`` (rows in
        the order given)."""
        lengths = self.column_sizes(classes)
        total = int(lengths.sum())
        device = self.members.device
        if total == 0:
            return th.sparse_coo_tensor(th.empty((2, 0), dtype=th.long, device=device),
                                        th.empty((0,), device=device),
                                        size=(len(classes), self.n_elements)).coalesce()
        row = th.repeat_interleave(th.arange(len(classes), device=device), lengths)
        starts = self.ccol.index_select(0, classes)
        flat = (th.arange(total, device=device)
                - th.repeat_interleave(th.cumsum(lengths, 0) - lengths, lengths)
                + th.repeat_interleave(starts, lengths))
        elements = self.members.index_select(0, flat)
        return th.sparse_coo_tensor(th.stack([row, elements]),
                                    th.ones((total,), device=device),
                                    size=(len(classes), self.n_elements)).coalesce()

    def dense_columns(self, classes):
        """``(B, |S|)`` float 0/1: the element sets of ``classes`` (dense, for compound
        classes of a batch)."""
        return self.columns_t(classes).to_dense()


class SparseElementSets(Region):
    """Sets of elements of a constructed model: named classes by id (read sparsely from the
    :class:`MembershipTable`) or, for compound classes, a dense 0/1 ``mask`` ``(B, |S|)``.

    Closed under intersection, union, complement (within :math:`S`) and preimages of the
    roles; all exact."""

    def __init__(self, table, ids=None, mask=None):
        if (ids is None) == (mask is None):
            raise ValueError("Give the ids of named classes or a mask.")
        self.table, self.ids, self._mask = table, ids, mask

    def __len__(self):
        return len(self.ids) if self.ids is not None else self._mask.shape[0]

    @property
    def named(self):
        return self.ids is not None

    @property
    def mask(self):
        """``(B, |S|)`` float 0/1."""
        if self._mask is None:
            self._mask = self.table.dense_columns(self.ids)
        return self._mask

    def take(self, ids):
        if self.ids is not None:
            return SparseElementSets(self.table, ids=self.ids[ids])
        return SparseElementSets(self.table, mask=self.mask[ids])

    def with_mask(self, mask):
        return SparseElementSets(self.table, mask=mask)

    def intersect(self, other):
        mask = self.mask * other.mask
        return self.with_mask(mask), (mask.sum(dim=1, keepdim=True) == 0).to(mask.dtype)

    def union(self, other):
        return self.with_mask(th.maximum(self.mask, other.mask))

    def complement(self):
        return self.with_mask(1 - self.mask)

    def is_empty(self):
        return self.mask.sum(dim=1) == 0


class MembershipClasses(Classes):
    """A class is the set of elements of the constructed model that belong to it
    (:class:`SparseElementSets`); nothing is learned.

    :param table: The :class:`MembershipTable` of the constructed model.

    Part (vocabulary value): ``fixed by reasoning``. Used by: ``canonical_mixture``.
    """

    region = SparseElementSets
    part = "fixed by reasoning"
    exact_conjunction = True

    def __init__(self, table):
        super().__init__()
        self.table = table

    def __len__(self):
        return self.table.n_classes

    @property
    def extra_parts(self):
        # complement and union are set operations within the support, not the negation and
        # disjunction of a logic the canonical model is a model of: they are not named
        return {"existential": "exact"}

    def forward(self, ids, detach_size=False):
        return SparseElementSets(self.table, ids=ids.to(self.table.keys.device))

    def tables(self):
        return {}


class CanonicalRoles(Relations):
    """A role is its set of pairs of elements of the constructed model, as a sparse
    ``(|S|, |S|)`` matrix :math:`E_r` per role (``E_r[x, y] = 1`` iff :math:`(x, y) \\in r`).

    The preimage of a set of elements (:math:`\\exists r.C`) is exact; the preimage of an
    individual's distribution is the region of the elements related to it
    (:class:`~mowl.representation.library.mixture_individuals.RelatedMixture`), read by the
    probability-mass measure as :math:`p_a \\cdot E_r p_b`.

    :param roles: ``{role id: sparse (|S|, |S|) bool tensor}``.
    :param n_elements: :math:`|S|`.
    :param n_roles: Number of role ids (default: one more than the largest key).

    Part (vocabulary value): ``fixed by reasoning``. Used by: ``canonical_mixture``.
    """

    part = "fixed by reasoning"

    def __init__(self, roles, n_elements, n_roles=None):
        super().__init__()
        self.n_elements = int(n_elements)
        self.keys = sorted(int(k) for k in roles)
        self.n_roles = int(n_roles) if n_roles is not None else (
            max(self.keys) + 1 if self.keys else 0)
        for key in self.keys:
            r = roles[key].coalesce()
            keep = r.values().bool()
            edges = r.indices()[:, keep].to(th.long) if keep.numel() else \
                th.empty((2, 0), dtype=th.long)
            self.register_buffer(f"role_{key}", _sparse_float(
                edges, (self.n_elements, self.n_elements)), persistent=False)

    def __len__(self):
        return self.n_roles

    def matrix(self, key):
        """The sparse float :math:`E_r` of role ``key``, or ``None`` if it has no pairs."""
        name = f"role_{int(key)}"
        if not hasattr(self, name):
            return None
        e = getattr(self, name)
        return e if e._nnz() > 0 else None

    def tables(self):
        return {}

    def preimage(self, ids, region):
        from mowl.representation.library.mixture_individuals import (
            ElementMixture, RelatedMixture)
        if isinstance(region, ElementMixture):
            return RelatedMixture(self, ids, region)
        if not isinstance(region, SparseElementSets):
            raise ConstructorUnavailable(
                f"CanonicalRoles maps sets of elements and distributions, not "
                f"{type(region).__name__}.")
        mask = region.mask
        out = th.zeros_like(mask)
        for key in th.unique(ids).tolist():
            e = self.matrix(key)
            rows = th.nonzero(ids == key, as_tuple=False).reshape(-1)
            if e is None:
                continue
            reach = th.sparse.mm(e, mask.index_select(0, rows).t()).t()
            out[rows] = (reach > 0).to(mask.dtype)
        return region.with_mask(out)


@register_source
class CanonicalCompletionSource(CanonicalModelSource):
    """The canonical model of a normalised EL++ ontology as the completion builds it (a
    :class:`~mowl.geometric.completion.CanonicalModel`), for readings that restrict it
    sparsely (:class:`MembershipReading`). The same construction as the ``canonical model``
    source, without expanding the extension of every name into a finite model (which takes
    time quadratic in the number of names)."""

    key = "canonical completion"

    @property
    def produces(self):
        from mowl.geometric.completion import CanonicalModel
        return CanonicalModel

    def build(self, data):
        return canonical_model_of(data)

    def default_names(self, built):
        from mowl.geometric.normal_forms import BOTTOM, TOP, is_nominal_name
        names = sorted(n for n in built.concept_names
                       if n not in (TOP, BOTTOM) and not is_nominal_name(n))
        return names, sorted(built.role_names), sorted(built.individuals)


class MembershipRead(ReadModel):
    """What :class:`MembershipReading` returns: a :class:`ReadModel
    <mowl.representation.construction.ReadModel>` with the restriction of the canonical model
    to the support (``structure``, a :class:`~mowl.geometric.membership.MembershipStructure`)
    and its :class:`MembershipTable` (``table``)."""

    def __init__(self, structure, table, classes, relations, names, roles, individuals):
        n = table.n_elements
        super().__init__(classes, relations, None,
                         SparseElementSets(table, mask=th.ones(1, n)),
                         SparseElementSets(table, mask=th.zeros(1, n)), list(names),
                         list(roles), list(individuals))
        self.structure = structure
        self.table = table


@register_reading
class MembershipReading(Reading):
    """The canonical model restricted to the elements of the satisfiable named classes and
    of the individuals (the support :math:`S` of the canonical mixture), read sparsely: a
    class is the set of its elements (:class:`SparseElementSets`), a role a sparse relation
    over :math:`S` (:class:`CanonicalRoles`). Individuals get no region here: the trained
    individual model of the canonical mixture places them.

    Reads a :class:`~mowl.geometric.completion.CanonicalModel` (source
    ``"canonical completion"``) or a finite model built from one (source
    ``"canonical model"``).

    :param allow_absent_individuals: Individuals of the given names that the canonical
        model does not know (they occur only in validation or test data) get no element
        (``individual_element`` ``-1``) instead of an error.
    """

    key = "membership"
    region = SparseElementSets
    class_part = "fixed by reasoning"
    relation_part = "fixed by reasoning"
    individual_part = "canonical element"
    constructors = {"conjunction": "exact", "existential": "exact", "negation": "exact",
                    "disjunction": "exact"}
    guarantees = (Guarantee(
        "soundness", "A ⊑ B holds between the element sets of satisfiable named classes "
        "other than ⊤ iff the ontology entails it; the membership M[x, C] is entailment of "
        "C by the name of x.", "normalised EL++ (named classes)"),)

    def __init__(self, allow_absent_individuals=True):
        self.allow_absent_individuals = bool(allow_absent_individuals)

    @property
    def accepts(self):
        from mowl.geometric.completion import CanonicalModel
        return (CanonicalModel, FiniteModel)

    def check(self, built):
        super().check(built)
        if isinstance(built, FiniteModel) and type(built.origin).__name__ != "CanonicalModel":
            raise ConstructorUnavailable("The membership reading reads a canonical model "
                                         "(or a finite model built from one).")

    def structure(self, built, names, roles, individuals):
        """The :class:`~mowl.geometric.membership.MembershipStructure` of ``built`` for
        class, role and individual names in the order of their ids."""
        from mowl.geometric.membership import MembershipStructure
        cm = built.origin if isinstance(built, FiniteModel) else built
        return MembershipStructure.from_canonical_model(
            cm, {n: i for i, n in enumerate(names)},
            {n: i for i, n in enumerate(individuals)}, {n: i for i, n in enumerate(roles)},
            allow_absent_individuals=self.allow_absent_individuals)

    def read(self, built, names, roles, individuals, source):
        names = [source.name_of(n) for n in names]
        return self.read_structure(self.structure(built, names, roles, individuals),
                                   names, roles, individuals)

    @staticmethod
    def read_structure(structure, names=(), roles=(), individuals=()):
        """The :class:`MembershipRead` of a structure already built (or loaded)."""
        table = MembershipTable(structure)
        n_roles = max(len(roles), max(structure.roles, default=-1) + 1)
        relations = CanonicalRoles(structure.roles, table.n_elements, n_roles)
        classes = SparseElementSets(table, ids=th.arange(table.n_classes))
        return MembershipRead(structure, table, classes, relations, names, roles,
                              individuals)


def membership_from_sizes(nb_classes, nb_relations, nb_individuals):
    """A constructed model for a geometry built from sizes only: one element per class
    (belonging to that class only) and one element per individual (belonging to no class),
    no pairs of any role."""
    from mowl.geometric.membership import MembershipStructure
    n = nb_classes + nb_individuals
    idx = th.arange(nb_classes)
    m = th.sparse_coo_tensor(th.stack([idx, idx]), th.ones(nb_classes, dtype=th.bool),
                             size=(n, nb_classes), dtype=th.bool).coalesce()
    empty = th.sparse_coo_tensor(th.empty((2, 0), dtype=th.long),
                                 th.empty((0,), dtype=th.bool), size=(n, n)).coalesce()
    names = [f"c{c}" for c in range(nb_classes)] + [f"i{i}" for i in range(nb_individuals)]
    return MembershipStructure(
        element_names=names, element_index={x: k for k, x in enumerate(names)}, M=m,
        roles={r: empty for r in range(nb_relations)},
        individual_element=th.arange(nb_classes, n, dtype=th.long),
        class_index_dict={f"c{c}": c for c in range(nb_classes)},
        individual_index_dict={f"i{i}": i for i in range(nb_individuals)},
        role_index_dict={f"r{r}": r for r in range(nb_relations)})
