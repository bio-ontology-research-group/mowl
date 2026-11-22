"""MultiBoxEL and MultiELBE with the objectives other than ``"axiom"``, from parts, and the
geometry that carries a definition table.

=============  ======================================================================
``reading``    parts
=============  ======================================================================
``axiom``      (:mod:`multibox_methods`) union of boxes, head and tail boxes, bounds
``structural`` slots of boxes (:class:`SlotClasses`) wrapped in :class:`DefinedClasses`,
               :class:`SlotMeasure`, :class:`StructuralReading`, penalties
               (:class:`ExistencePenalty`, :class:`SizePenalty`, :class:`WitnessPenalty`)
``point``,     union of boxes, the axiom measure, :class:`PointReading` with
``mixed``,     :class:`PointTerms` (soft membership, :class:`SoftMembership`)
``hybrid``
=============  ======================================================================

What is **not** a part: *re-seeding* during training (moving components that cover nothing
to where they are needed) changes the structure of the model between optimisation steps; it
is a training regime (:class:`mowl.training.structure.IdleComponentReseeding`). The geometry
offers the hook the regime calls (:meth:`StructuralGeometry.reseed_components`); the
relocation rule is :func:`~mowl.representation.library.structural_reseed.reseed_components`,
applied to the tables of the geometry: it is about the layout of the tables of unions of
boxes, not about an axiom.
"""

import numpy as np
import torch as th
import torch.nn as nn

from mowl.nn.el.multibox_utils import effective_components
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.structural_measure import SlotMeasure
from mowl.representation.library.structural_points import (PointReading, PointTerms,
                                                           SoftMembership)
from mowl.representation.library.structural_reading import (ExistencePenalty, SizePenalty,
                                                            StructuralReading, WitnessPenalty)
from mowl.representation.library.structural_regions import DefinedClasses, SlotClasses

__all__ = ["StructuralGeometry"]


def _rows(model):
    """The classes the size penalty covers: not defined, not the top class."""
    classes = model.class_model
    if isinstance(classes, DefinedClasses):
        return classes.primitive_mask(model.top_idx)
    mask = th.ones(len(classes), dtype=th.bool, device=model.device)
    if model.top_idx is not None and 0 <= model.top_idx < len(classes):
        mask[model.top_idx] = False
    return mask


class StructuralGeometry(ComposedGeometry):
    """A :class:`~mowl.representation.ComposedGeometry` of slots of boxes with the
    structural tables: :meth:`set_structural_tables` wraps the class model in
    :class:`DefinedClasses` and adds the existence and witness penalties.

    :param base_regularizers: The regularisers that are always there (bump norm, component
        penalties).
    :param existence_weight, size_reg, witness_weight: The weights of the three penalties.
    :param size_reg_without_tables: The size penalty has to know which class is ``owl:Thing``
        (it must not shrink it) and which classes are defined, so by default it is zero until
        :meth:`set_structural_tables` has been called. ``True``: it is applied to every class
        before that too (the earlier behaviour, which shrinks ``owl:Thing``).
    :param size_tables, size_scales: ``geometry -> [raw half-width tensors]`` and one factor
        per tensor for the size penalty (default: the offsets of a union of boxes, factor 2;
        a prism: the inradius and the box offsets, factors 1 and 2).
    """

    def __init__(self, classes, relations, measure, objective, individuals=None,
                 base_regularizers=(), existence_weight=1.0, size_reg=1e-3,
                 witness_weight=0.0, size_tables=None, size_scales=(2.0,),
                 size_reg_without_tables=False, **options):
        super().__init__(classes, relations, measure, objective, individuals=individuals,
                         regularizers=(), **options)
        self.base_regularizers = tuple(base_regularizers)
        self.existence_weight, self.size_reg = float(existence_weight), float(size_reg)
        self.witness_weight = float(witness_weight)
        self.size_reg_without_tables = bool(size_reg_without_tables)
        self.tables_installed = False
        self.size_tables = size_tables if size_tables is not None else \
            (lambda m: [m._slot_classes.offsets.weight])
        self.size_scales = tuple(size_scales)
        self.definition_table = None
        self.existence_targets = None
        self.top_idx = None
        self.witness_embed = None
        self.last_reseed_count = 0
        self.reseed_options = {"mode": "dead", "size": 0.15, "batch": 1}
        self._set_penalties()

    # -- tables ---------------------------------------------------------------------------
    @property
    def wants_structural_tables(self):
        """Whether the geometry has no definition table and no existence targets yet: a
        model that hosts it (:class:`~mowl.models.ComposedEL`) builds them from its
        dataset (:func:`~mowl.nn.el.structural.build_structural_tables`, with
        :attr:`structural_table_options`) and calls :meth:`set_structural_tables`."""
        return self.definition_table is None and self.existence_targets is None

    @property
    def structural_table_options(self):
        """The arguments of :func:`~mowl.nn.el.structural.build_structural_tables` that the
        geometry decides: ``k`` (components per class), ``max_components`` (slots) and
        ``enabled`` (the definitions of the objective). Prisms (one box, one triangle) have
        ``k = 1`` and no bound on the slots."""
        base = self._slot_classes
        components = int(getattr(base, "components", 1))
        slots = getattr(base, "max_components", 2 ** 30)
        return {"k": components, "max_components": int(slots),
                "enabled": bool(getattr(self.objective, "definitions", True))}

    @property
    def _slot_classes(self):
        classes = self.class_model
        return classes.base if isinstance(classes, DefinedClasses) else classes

    def _set_penalties(self):
        penalties = []
        targets = self.existence_targets
        margin = lambda: self.objective.margin  # noqa: E731
        if targets is not None:
            penalties.append(ExistencePenalty(targets.classes, targets.pairs, margin,
                                              lambda: self.existence_weight))
        penalties.append(SizePenalty(self.size_tables, self.size_scales, _rows,
                                     weight=self._size_weight))
        if targets is not None and targets.witness is not None \
                and self.witness_embed is not None:
            penalties.append(WitnessPenalty(targets.witness, margin,
                                            lambda: self.witness_weight))
        self.regularizers = self.base_regularizers + tuple(penalties)

    def _size_weight(self):
        known = self.tables_installed or self.size_reg_without_tables
        return self.size_reg if known else 0.0

    def set_structural_tables(self, table, targets, top_idx=None):
        """Attach a :class:`~mowl.nn.el.structural.DefinitionTable` and
        :class:`~mowl.nn.el.structural.ExistenceTargets` (as the module's method of the same
        name): defined classes become the intersections of their definers, and the
        existence (and, with ``witness_weight > 0`` and witness targets, the witness) penalties
        read ``targets``."""
        base = self._slot_classes
        self.class_model = DefinedClasses(base, table) if table is not None else base
        self.definition_table, self.existence_targets, self.top_idx = table, targets, top_idx
        self.tables_installed = True
        witness = None if targets is None else targets.witness
        if witness is not None and self.witness_weight > 0.0:
            n = int(witness.classes.numel())
            if self.witness_embed is None or self.witness_embed.num_embeddings != n:
                embed = nn.Embedding(n, base.embed_dim)
                with th.no_grad():
                    nn.init.uniform_(embed.weight, a=-1.0, b=1.0)
                    if n > 0:
                        embed.weight.div_(th.linalg.norm(embed.weight, dim=-1, keepdim=True))
                self.witness_embed = embed.to(base.centres.weight.device)
        self._set_penalties()

    def witness_points(self):
        """Detached CPU witness coordinates ``(W, d)``, or ``None`` when there are none."""
        if self.witness_embed is None:
            return None
        return self.witness_embed.weight.detach().cpu().clone()

    def effective_components(self, eps=1e-3):
        """Per class the number of components that are not degenerate and not nested in
        another (:func:`mowl.nn.el.multibox_utils.effective_components`), on the raw
        tables."""
        base = self._slot_classes
        shape = (len(base), base.components, base.embed_dim)
        return effective_components(base.centres.weight.view(shape),
                                    base.offsets.weight.view(shape), eps=eps)

    # -- re-seeding: a regime ---------------------------------------------------------------
    def reseed(self, rng):
        """Relocate dead or idle components (see the class docstring); returns the number of
        moves."""
        from mowl.representation.library.structural_reseed import reseed_components
        return reseed_components(self, rng)

    @property
    def reseed_components(self):
        """The hook of :class:`~mowl.training.structure.IdleComponentReseeding` (the
        regime): ``hook(epoch)`` re-seeds with a generator seeded by ``epoch`` and returns the
        number of moves. Only geometries of slots of boxes have it (the relocation rule is
        about unions of boxes; a prism does not)."""
        if not isinstance(self._slot_classes, SlotClasses):
            raise AttributeError(
                f"{type(self._slot_classes).__name__} classes have no components to "
                f"re-seed: only slots of boxes do.")
        return lambda epoch=0: self.reseed(np.random.default_rng(int(epoch)))


# ------------------------------------------------------------------------------------
# the point objectives
# ------------------------------------------------------------------------------------

def point_reading(mode, margin, terms, **options):
    """:class:`PointReading` with the options of the axiom objective of MultiBoxEL."""
    return PointReading(terms, mode, margin, **options)


__all__ += ["point_reading", "SlotClasses", "SlotMeasure", "StructuralReading",
            "PointTerms", "SoftMembership"]
