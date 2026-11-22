"""The canonical mixture as a combination of parts: a constructed canonical model and a
trained individual model.

========================  ======================================================================
source                    ``"canonical completion"`` (:class:`~mowl.representation.library.
                          mixture_tables.CanonicalCompletionSource`): the canonical model of
                          the EL++ ontology, by completion
reading                   ``"membership"`` (:class:`~mowl.representation.library.
                          mixture_tables.MembershipReading`): the elements of the named
                          classes and the individuals, sparse
class model               :class:`~mowl.representation.library.mixture_tables.MembershipClasses`
                          (fixed by reasoning: a class is its set of elements)
relation model            :class:`~mowl.representation.library.mixture_tables.CanonicalRoles`
                          (fixed by reasoning: a role is its set of pairs)
individual model          :class:`~mowl.representation.library.mixture_individuals.
                          MixtureIndividuals` (learned: a softmax distribution over the
                          admissible support, low-rank logits)
measure                   :class:`~mowl.representation.library.mixture_measures.ProbabilityMass`
                          (the mass on a set of elements; binary cross entropy as the loss)
objective                 the default reading, with negatives of the two assertion forms
regularisers              :class:`~mowl.representation.library.mixture_measures.EntropyPenalty`
                          (``entropy_reg > 0``)
training regime           :class:`~mowl.representation.library.mixture_training.IndividualBatches`
                          (negatives: ``NonEntailedClasses``, ``UniformTails``)
========================  ======================================================================

:func:`canonical_mixture` is the named combination (``METHODS["canonical_mixture"]``); every
part can be replaced through its arguments. :class:`mowl.models.CanonicalMixture` builds the
constructed model of a dataset and calls it.
"""

import torch as th

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.mixture_individuals import (MAX_SUPPORT_ENTRIES,
                                                             MixtureIndividuals)
from mowl.representation.library.mixture_measures import EntropyPenalty, ProbabilityMass
from mowl.representation.library.mixture_tables import (MembershipClasses, MembershipReading,
                                                        membership_from_sizes)
from mowl.representation.methods import register_method
from mowl.representation.objectives import FORMS, Objective

__all__ = ["MixtureGeometry", "canonical_mixture", "mixture_objective"]

#: The forms with a negative loss of their own.
NEGATIVE_FORMS = ("class_assertion", "object_property_assertion")


def mixture_objective(**options):
    """The objective of the canonical mixture: the default reading (an individual is a
    member of a class, ``apart`` for a negative), with negatives of the assertion forms."""
    options.setdefault("negatives", NEGATIVE_FORMS)
    return Objective(**options)


class MixtureGeometry(ComposedGeometry):
    """The canonical mixture as a combination of parts (module docstring).

    Besides the interface of every geometry it answers the questions of the method:
    :meth:`class_assertion_score`, :meth:`object_property_assertion_score`,
    :meth:`rank_classes`, :meth:`log_p`, and it draws the negatives of its regime
    (:meth:`sample_class_negatives`, :meth:`sample_role_negatives`). ``structure`` is the
    constructed model (a :class:`~mowl.geometric.membership.MembershipStructure`).

    ``member`` is the exact score :math:`-\\log P(a \\in C)` (``inf`` for probability zero);
    the losses of the objective are those of the measure (binary cross entropy).
    """

    MAX_SUPPORT_ENTRIES = MAX_SUPPORT_ENTRIES
    role_axiom_capable = False

    def __init__(self, classes, relations, individuals, measure, objective, structure=None,
                 regularizers=(), source="canonical completion", reading="membership",
                 class_negatives=None, role_negatives=None):
        super().__init__(classes, relations, measure, objective, individuals=individuals,
                         regularizers=regularizers, supported_forms=FORMS)
        self.margin_semantics = None
        self.structure = structure
        self.source, self.reading = source, reading
        from mowl.representation.library.mixture_training import (NonEntailedClasses,
                                                                   UniformTails)
        self.class_negatives = class_negatives or NonEntailedClasses()
        self.role_negatives = role_negatives or UniformTails()

    # -- no margin ---------------------------------------------------------------------------
    @property
    def margin(self):
        """``None``: the probability-mass measure has no margin."""
        return None

    @margin.setter
    def margin(self, value):
        if value:
            raise ValueError("The canonical mixture has no margin.")

    # -- the tables of the method --------------------------------------------------------------
    @property
    def U(self):
        return self.individual_model.U

    @property
    def V(self):
        return self.individual_model.V

    @property
    def b(self):
        return self.individual_model.b

    @property
    def rank(self):
        return self.individual_model.rank

    @property
    def init_self_mass(self):
        return self.individual_model.init_self_mass

    @property
    def entropy_reg(self):
        return sum(r.weight for r in self.regularizers if isinstance(r, EntropyPenalty))

    @property
    def n_elements(self):
        return self.individual_model.n_elements

    @property
    def n_classes(self):
        return len(self.class_model)

    @property
    def n_ind(self):
        return len(self.individual_model)

    def __getattr__(self, name):
        if name in ("support_offsets", "support_indices", "own_logit_offset",
                    "is_full_support", "support_lengths", "individual_element"):
            return getattr(self.individual_model, name)
        return super().__getattr__(name)

    # -- scores ---------------------------------------------------------------------------------
    def member(self, individual_ids, region):
        return self.measure.score(self.individuals(individual_ids), region)

    def _rows(self, data, width):
        data = th.as_tensor(data, dtype=th.long, device=self.U.device)
        return data.reshape(-1, width)

    def class_assertion_score(self, data):
        """``(B,)``: :math:`-\\log P(a \\in C)` for rows ``(a, C)``."""
        data = self._rows(data, 2)
        return self.member(data[:, 0], self.classes(data[:, 1])).reshape(-1)

    def object_property_assertion_score(self, data):
        """``(B,)``: :math:`-\\log (p_a \\cdot E_r p_b)` for rows ``(a, r, b)``."""
        data = self._rows(data, 3)
        target = self.exists(data[:, 1], self.individuals(data[:, 2]))
        return self.member(data[:, 0], target).reshape(-1)

    def log_p(self, ids):
        """``(B, |S|)``: the log-probabilities of the individuals, ``-inf`` outside their
        supports."""
        return self.individual_model.log_p(ids)

    def rank_classes(self, ids):
        """``(B, n_classes)``: :math:`-\\log P(a \\in C)` of every class. Exact where the
        probability is 0 or 1 (by the counts of support elements in every class); in between
        computed as the total mass minus the mass in :math:`C`
        (:meth:`class_assertion_score` is the exact path)."""
        model, table = self.individual_model, self.class_model.table
        ids = th.as_tensor(ids, dtype=th.long, device=self.U.device).reshape(-1)
        step = model.row_chunk(len(ids))
        if len(ids) == 0:
            return th.empty((0, self.n_classes), dtype=self.U.dtype, device=self.U.device)
        if len(ids) > step:
            return th.cat([self.rank_classes(ids[s:s + step]) for s in range(0, len(ids), step)])
        d = model.distribution(ids)
        probs = th.zeros((len(ids), self.n_elements), dtype=self.U.dtype, device=self.U.device)
        support = th.zeros_like(probs)
        probs[d.dense_pos] = th.exp(d.dense_logp)
        support[d.dense_pos] = 1.0
        rows = d.part_pos.index_select(0, d.rows)
        probs[rows, d.elements] = d.probs
        support[rows, d.elements] = 1.0
        inside = th.sparse.mm(table.matrix_t.to(probs.dtype), probs.t()).t()
        counts = th.sparse.mm(table.matrix_t.to(probs.dtype), support.t()).t()
        lengths = model.support_lengths.index_select(0, ids).to(probs.dtype).unsqueeze(1)
        outside = (probs.sum(dim=1, keepdim=True) - inside).clamp(0.0, 1.0)
        value = (1.0 - outside).clamp(0.0, 1.0)
        value = th.where(counts >= lengths - 0.5, th.ones_like(value), value)
        value = th.where(counts < 0.5, th.zeros_like(value), value)
        return -th.log(value) + 0.0

    # -- negatives and the batch regulariser ------------------------------------------------------
    def sample_class_negatives(self, rows):
        return self.class_negatives.sample(self, self._rows(rows, 2))

    def sample_role_negatives(self, rows):
        return self.role_negatives.sample(self, self._rows(rows, 3))

    def regularization_loss(self, ind_idx=None):
        """The regularisers, those that read a batch on the individuals ``ind_idx`` (default:
        all individuals)."""
        total = self.U.sum() * 0.0
        for regularizer in self.regularizers:
            if getattr(regularizer, "batch_penalty", False):
                term = regularizer.penalty(self, None if ind_idx is None else
                                           th.as_tensor(ind_idx, dtype=th.long,
                                                        device=self.U.device).reshape(-1))
            else:
                term = regularizer.penalty(self)
            if term is not None:
                total = total + term
        return total

    @property
    def parts(self):
        parts = super().parts
        parts.update({"individuals": self.individual_model.part,
                      "source": getattr(self.source, "key", self.source),
                      "reading": getattr(self.reading, "key", self.reading)})
        return parts


@register_method("canonical_mixture")
def canonical_mixture(nb_classes, nb_relations, nb_individuals=0, embed_dim=50,
                      structure=None, entropy_reg=0.0, init_self_mass=0.9,
                      training_class_assertions=None, loss="bce", eps=1e-7,
                      classes=None, relations=None, individuals=None, measure=None,
                      objective=None, regularizers=None, class_negatives=None,
                      role_negatives=None, tables=None, geometry=MixtureGeometry):
    """The canonical mixture from its parts (module docstring).

    :param structure: The constructed model (a
        :class:`~mowl.geometric.membership.MembershipStructure`, the reading ``"membership"``
        of the canonical model; :class:`mowl.models.CanonicalMixture` builds it). Without it
        (a geometry built from sizes) every class has one element of its own and every
        individual an element of no class.
    :param embed_dim: The rank :math:`r` of the logits.
    :param entropy_reg: Weight of the entropy regulariser (none if 0).
    :param init_self_mass: Initial mass of an individual's own element.
    :param training_class_assertions: Checked to be entailed by the constructed model.
    :param loss, eps: Options of the measure (:class:`ProbabilityMass`).
    :param classes, relations, individuals, measure, objective, regularizers: Replace a part
        (an instance).
    :param class_negatives, role_negatives: The negative samplers.
    :param tables: Not used (the method has no module of its own to share tables with).
    """
    if tables is not None:
        raise TypeError("The canonical mixture has no module whose tables could be shared.")
    if structure is None:
        structure = membership_from_sizes(nb_classes, nb_relations, max(nb_individuals, 1))
    read = MembershipReading.read_structure(structure)
    if classes is None:
        classes = MembershipClasses(read.table)
    if relations is None:
        relations = read.relations
    if individuals is None:
        individuals = MixtureIndividuals(read.table, structure.individual_element,
                                         rank=embed_dim, init_self_mass=init_self_mass,
                                         training_class_assertions=training_class_assertions)
    if float(entropy_reg) < 0.0:
        raise ValueError(f"entropy_reg must be >= 0, got {entropy_reg!r}")
    if regularizers is None:
        regularizers = (EntropyPenalty(entropy_reg),) if float(entropy_reg) > 0.0 else ()
    return geometry(classes, relations, individuals,
                    measure if measure is not None else ProbabilityMass(loss, eps),
                    objective if objective is not None else mixture_objective(),
                    structure=structure, regularizers=regularizers,
                    class_negatives=class_negatives, role_negatives=role_negatives)
