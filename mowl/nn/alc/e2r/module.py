r"""Neural module of Embed2Reason (E2R; Garg et al., NeurIPS 2019), put together from parts.

Concepts and relations are axis-parallel linear subspaces of :math:`\mathbb{C}^d` simulated as
:math:`\mathbb{R}^{2d}` (Sec. 4), given by learnable indicator vectors; entities are learnable
vectors. The module is the geometry of the method
:func:`~mowl.representation.library.logic_subspace.e2r`: class model
:class:`~mowl.representation.library.logic_subspace.SubspaceClasses`, relation model
:class:`~mowl.representation.library.logic_subspace.SubspaceRelations`, individual model
:class:`~mowl.representation.library.logic_subspace.PointIndividuals`, measure
:class:`~mowl.representation.library.logic_subspace.ResidualMeasure`, objective
:class:`~mowl.representation.library.logic_subspace.SubspaceObjective`; see
:mod:`mowl.representation.library.logic_subspace_algebra` for the loss terms. The hand-written
module that it replaces is kept for comparison as :mod:`mowl.baselines.nn.alc.e2r`.

The module also provides the exact linear-algebra helpers for *arbitrary* (oblique) subspaces
given by orthogonal projectors (:func:`projector`, :func:`meet_projector`,
:func:`join_projector`, :func:`orthocomplement`, :func:`is_subspace`). They implement the
quantum-logic operations of Sec. 3 in general and are used to check the axis-parallel case
against ground truth and to exhibit non-distributivity (which the axis-parallel restriction
removes, supplementary material Theorem 11).
"""
from types import SimpleNamespace

import torch.nn as nn

from mowl.representation.library.logic_subspace import SubspaceGeometry, e2r_parts, AXIOM_KINDS
from mowl.representation.library.logic_subspace_algebra import (  # noqa: F401
    projector, indicator_projector, rank, orthocomplement, join_projector, meet_projector,
    is_subspace, residual_length)

__all__ = ["E2RModule", "AXIOM_KINDS", "projector", "indicator_projector", "rank",
           "orthocomplement", "join_projector", "meet_projector", "is_subspace",
           "residual_length"]


class E2RModule(SubspaceGeometry):
    r"""Parameters and per-axiom losses of E2R.

    * entity :math:`i`: vector :math:`x_i \in \mathbb{R}^{2d}` (imaginary half pushed to
      ``0`` by eq. (1));
    * concept :math:`C`: indicator :math:`y_C \in [0, 1]^{2d}` (imaginary half pushed to
      ``0`` by eq. (2), values pushed to :math:`\{0, 1\}` by eq. (3));
    * relation :math:`R`: indicator :math:`z_R \in [0, 1]^{2d}`.

    ``owl:Thing`` and ``owl:Nothing`` (if their indices are given) are fixed to the concept
    universe :math:`(1_d, 0_d)` and to :math:`\{0\}`.

    The paper optimises the indicators directly (eq. (10)); it does not say how they are
    kept in :math:`[0, 1]`. Here :meth:`project_` clips them after every optimiser step
    (projected gradient descent).

    :meth:`forward` dispatches on an axiom kind (one of ``AXIOM_KINDS``) and a batch of
    index rows, returning one loss value per row. :meth:`forward_fs` of
    :class:`~mowl.nn.alc.module.ALCModule` (fuzzy-set semantics) does not apply.

    :param num_entities: Number of individuals.
    :param num_classes: Number of classes.
    :param num_relations: Number of object properties.
    :param dim: :math:`d`; vectors have :math:`2d` real coordinates. Defaults to ``100``
        (Sec. 5).
    :param top_index: Index of ``owl:Thing`` or ``None``.
    :param bottom_index: Index of ``owl:Nothing`` or ``None``.
    :param role_unit_length: Add the unit-length terms of the two entities to role assertion
        losses (see
        :func:`~mowl.representation.library.logic_subspace_algebra.role_membership_loss`).
    """

    def __init__(self, num_entities, num_classes, num_relations, dim=100, top_index=None,
                 bottom_index=None, role_unit_length=True):
        # The tables are created in the order of the earlier module (entities, concepts,
        # relations) and initialised by :meth:`reset_parameters`, so that a seed gives the
        # numbers it always gave.
        tables = SimpleNamespace(
            e_embedding=nn.Embedding(max(num_entities, 1), 2 * dim),
            c_embedding=nn.Embedding(max(num_classes, 1), 2 * dim),
            r_embedding=nn.Embedding(max(num_relations, 1), 2 * dim),
            top_index=top_index, bottom_index=bottom_index, role_unit_length=role_unit_length)
        super().__init__(**e2r_parts(None, None, regularization=False, tables=tables))
        self.role_unit_length = role_unit_length
        self.reset_parameters()
