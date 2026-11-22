"""What CatE models share, whatever trains them: the lattice of the training ontology, the
validation queries, the scoring and the evaluation adapters. :class:`mowl.models.CatE` trains
with the regimes of :mod:`mowl.training`; the direct implementation in
:mod:`mowl.baselines.models.cate` keeps its own loop for the comparison in ``tests/baselines``."""

import logging
import os

import torch as th

from mowl.base_models.model import Model
from mowl.baselines.models.cate.lattice_ranking import order_mean_reciprocal_rank, score_columns
from mowl.baselines.models.cate.categorical import saturation as sat
from mowl.baselines.models.cate.categorical.lattice import CatELattice, UnseenConceptError

logger = logging.getLogger(__name__)

__all__ = ["CatEBase", "UnseenConceptError"]


class CatEBase(Model):
    """The lattice, the validation, the scoring and the evaluation of CatE models (see
    :class:`mowl.models.CatE`). A subclass makes ``init_module`` and trains."""


    axiom_scoring = "module"
    scored_forms = ("gci0", "gci2", "class_assertion")

    def __init__(self, dataset, embed_dim=200, margin=1.0, num_negs=4,
                 learning_rate=1e-4, batch_size=32768, p=2, weight_decay=1e-4,
                 cyclic_lr=True, negative_aggregation="per_negative", filter_negatives=True,
                 saturation_steps=0,
                 saturation_rules=sat.S1, transitive_closure=False, complete_negation=True,
                 existentials="auto", interaction_loss=False, obvious_morphisms=True,
                 early_stopping_patience=10, model_filepath=None, device="cpu"):
        super().__init__(dataset, model_filepath=model_filepath)
        if negative_aggregation not in ("per_negative", "mean"):
            raise ValueError("negative_aggregation must be 'per_negative' or 'mean'.")
        self.embed_dim = embed_dim
        self.margin = margin
        self.num_negs = num_negs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.p = p
        self.weight_decay = weight_decay
        self.cyclic_lr = cyclic_lr
        self.negative_aggregation = negative_aggregation
        self.filter_negatives = filter_negatives
        self.saturation_steps = saturation_steps
        self.saturation_rules = tuple(saturation_rules)
        self.transitive_closure = transitive_closure
        self.complete_negation = complete_negation
        self.existentials = existentials
        self.interaction_loss = interaction_loss
        self.obvious_morphisms = obvious_morphisms
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self._lattice = None
        self._module = None
        self._validation = None
        self._keys = None
        self._interaction = None

    # ---------------------------------------------------------------- lattice
    def _resolve_existentials(self):
        spec = self.existentials
        ds = self.dataset
        if spec is None:
            return {}
        if spec == "auto":
            prop = getattr(ds, "get_evaluation_property", None)
            if prop is None:
                return {}
            try:
                spec = {prop(): "evaluation_classes"}
            except (NotImplementedError, AttributeError, TypeError):
                return {}
        out = {}
        for role, fillers in dict(spec).items():
            if fillers == "classes":
                fillers = list(ds.classes.as_str)
            elif fillers == "evaluation_classes":
                fillers = list(ds.evaluation_classes[1].as_str)
            out[str(role)] = [str(f) for f in fillers]
        return out

    @property
    def lattice(self):
        """The :class:`~mowl.baselines.models.cate.categorical.lattice.CatELattice` of the training
        ontology (built on first access)."""
        if self._lattice is None:
            ds = self.dataset
            self._lattice = CatELattice(
                ds.ontology, classes=ds.classes.as_str, individuals=ds.individuals.as_str,
                existentials=self._resolve_existentials(),
                complete_negation=self.complete_negation,
                saturation_steps=self.saturation_steps, rules=self.saturation_rules,
                transitive_closure=self.transitive_closure,
                obvious_morphisms=self.obvious_morphisms)
            logger.info("CatE lattice: %s", self._lattice.stats)
        return self._lattice

    @property
    def class_index_dict(self):
        """Class IRI to lattice element id (including ``owl:Thing`` and ``owl:Nothing``).

        :rtype: dict
        """
        return self.lattice.class_to_id

    @property
    def individual_index_dict(self):
        """Individual IRI to the lattice element id of :math:`\\{a\\}`.

        :rtype: dict
        """
        return self.lattice.individual_to_id

    @property
    def object_property_index_dict(self):
        """Role IRI to role index (used in ``gci2`` rows).

        :rtype: dict
        """
        return self.lattice.role_to_id

    @property
    def module(self):
        """The order-embedding module; ``module(rows, gci_name)`` scores rows of element
        ids (see :class:`~mowl.nn.graph.cate.CatEModule`)."""
        if self._module is None:
            self.init_module()
        return self._module

    @property
    def evaluation_model(self):
        return self.module

    def _interaction_ids(self, model=None):
        """(head ids, existential ids) of the interaction loss: the lattice elements of
        the dataset's head evaluation classes and the elements :math:`\\exists r.p` for
        the dataset's evaluation property."""
        if self._interaction is not None and self._interaction[0].device == th.device(self.device):
            return self._interaction
        ds = self.dataset
        prop = getattr(ds, "get_evaluation_property", None)
        if prop is None:
            raise ValueError(
                "interaction_loss needs a dataset with get_evaluation_property() "
                "(the PPI and GDA datasets).")
        role = str(prop())
        heads, tails = ds.evaluation_classes
        lat = self.lattice
        head_ids = [lat.class_to_id[c] for c in heads.as_str if c in lat.class_to_id]
        pairs = [(lat.role_to_id[role], lat.class_to_id[t]) for t in tails.as_str
                 if t in lat.class_to_id]
        ex_ids = [lat.existential_to_id[k] for k in pairs if k in lat.existential_to_id]
        if not head_ids or not ex_ids:
            raise ValueError(
                "interaction_loss found no interaction elements for role "
                f"{role}: the model needs existentials='auto' (or a matching "
                "'existentials' argument).")
        self._interaction = (
            th.tensor(head_ids, dtype=th.long, device=self.device),
            th.tensor(ex_ids, dtype=th.long, device=self.device))
        return self._interaction

    # ------------------------------------------------------------- validation
    def validation_queries(self):
        """Validation queries ``(head_id, true_id, candidate_ids)``: the released
        validation subsumptions of the dataset if it defines
        ``cate_validation_query_pairs`` (the CatE completion benchmarks), otherwise the
        named subsumptions :math:`C \\sqsubseteq D` and memberships :math:`D(a)` of
        the validation ontology (candidates: all named classes) and
        :math:`C \\sqsubseteq \\exists r.D` (candidates: the elements
        :math:`\\exists r.D'`). Axioms with elements outside the lattice are skipped."""
        if self._validation is not None:
            return self._validation
        lat = self.lattice
        bounds = {lat.top_id, lat.bot_id}
        classes = th.tensor(sorted(set(lat.class_to_id.values()) - bounds), dtype=th.long)
        pairs = getattr(self.dataset, "cate_validation_query_pairs", None)
        if pairs:
            out = []
            for head_iri, tail_iri in pairs:
                head_iri, tail_iri = str(head_iri), str(tail_iri)
                if head_iri in lat.class_to_id and tail_iri in lat.class_to_id:
                    out.append((lat.class_to_id[head_iri], lat.class_to_id[tail_iri],
                                classes))
            self._validation = out
            return out
        by_role = {}
        for (r, _), node in lat.existential_to_id.items():
            by_role.setdefault(r, []).append(node)
        by_role = {r: th.tensor(sorted(v), dtype=th.long) for r, v in by_role.items()}
        out = []
        onto = self.dataset.validation
        if onto is not None:
            from org.semanticweb.owlapi.model import AxiomType
            from org.semanticweb.owlapi.model import ClassExpressionType as CT
            for ax in onto.getAxioms():
                try:
                    if ax.getAxiomType() == AxiomType.SUBCLASS_OF:
                        sub, sup = ax.getSubClass(), ax.getSuperClass()
                        if sub.isAnonymous():
                            continue
                        h = lat.node_id(sub)
                        if not sup.isAnonymous():
                            out.append((h, lat.node_id(sup), classes))
                        elif sup.getClassExpressionType() == CT.OBJECT_SOME_VALUES_FROM \
                                and not sup.getFiller().isAnonymous():
                            r = lat.role_to_id.get(str(sup.getProperty().toStringID()))
                            if r in by_role:
                                out.append((h, lat.node_id(sup), by_role[r]))
                    elif ax.getAxiomType() == AxiomType.CLASS_ASSERTION:
                        cls = ax.getClassExpression()
                        if cls.isAnonymous():
                            continue
                        a = lat.individual_id(ax.getIndividual().toStringID())
                        out.append((a, lat.node_id(cls), classes))
                except UnseenConceptError:
                    continue
        self._validation = out
        return out

    def validation_mrr(self):
        """Raw mean reciprocal rank of the validation queries (ties: average rank; see
        :func:`mowl.baselines.models.cate.lattice_ranking.order_mean_reciprocal_rank`)."""
        return order_mean_reciprocal_rank(self.module, self.validation_queries(), self.device)

    # ---------------------------------------------------------------- scoring
    def _tensor(self, ids):
        return th.as_tensor(ids, dtype=th.long, device=self.device).reshape(-1)

    @th.no_grad()
    def score_edges(self, src_ids, dst_ids):
        """:math:`s(A, B)` (Eq. 6) for element ids; lower is better."""
        return self.module.order_violation(self._tensor(src_ids), self._tensor(dst_ids))

    def score_subsumption(self, sub_ids, sup_ids):
        """Score of :math:`C \\sqsubseteq D` for element ids (e.g. from
        :attr:`class_index_dict`); lower is better."""
        return self.score_edges(sub_ids, sup_ids)

    def score_membership(self, individual_ids, class_ids):
        """Score of :math:`C(a)` as :math:`s(\\{a\\}, C)` for ids from
        :attr:`individual_index_dict` and :attr:`class_index_dict`; lower is better."""
        return self.score_edges(individual_ids, class_ids)

    @th.no_grad()
    def score_existential(self, sub_ids, role_ids, filler_ids):
        """Score of :math:`C \\sqsubseteq \\exists r.D` as :math:`s(C, \\exists r.D)`.

        :raises UnseenConceptError: If some :math:`\\exists r.D` is not an element.
        """
        return score_columns(self.module, "gci2", (sub_ids, role_ids, filler_ids), self.device)

    def score(self, axiom):
        """Score of a ``SubClassOf``, ``ClassAssertion`` or ``ObjectPropertyAssertion``
        axiom (:math:`r(a, b)` as :math:`\\{a\\} \\sqsubseteq \\exists r.\\{b\\}`, as the
        projector encodes it). Lower is better; 0 means the order holds exactly.

        :raises UnseenConceptError: If a concept description of the axiom is not an
            element of the lattice.
        :rtype: float
        """
        from org.semanticweb.owlapi.model import AxiomType
        lat = self.lattice
        t = axiom.getAxiomType()
        if t == AxiomType.SUBCLASS_OF:
            pair = (lat.node_id(axiom.getSubClass()), lat.node_id(axiom.getSuperClass()))
        elif t == AxiomType.CLASS_ASSERTION:
            pair = (lat.individual_id(axiom.getIndividual().toStringID()),
                    lat.node_id(axiom.getClassExpression()))
        elif t == AxiomType.OBJECT_PROPERTY_ASSERTION:
            from mowl.owlapi import OWLAPIAdapter
            ad = OWLAPIAdapter()
            obj = ad.create_class(str(axiom.getObject().toStringID()))
            ex = ad.create_object_some_values_from(axiom.getProperty(), obj)
            pair = (lat.individual_id(axiom.getSubject().toStringID()), lat.node_id(ex))
        else:
            raise TypeError(f"CatE.score does not score {t} axioms.")
        return self.score_edges([pair[0]], [pair[1]]).item()

    # ---------------------------------------------------------- persistence
    def load_best_model(self):
        """Load the parameters saved by :meth:`train`."""
        self.module.load_state_dict(th.load(self.model_filepath, weights_only=True))
        self.module.eval()

    def from_pretrained(self, file_name):
        """Load parameters from ``file_name``."""
        if not isinstance(file_name, str):
            raise TypeError("Parameter file_name must be a string.")
        if not os.path.exists(file_name):
            raise FileNotFoundError("Pretrained model path does not exist")
        self.module.load_state_dict(th.load(file_name, weights_only=True))

    # -------------------------------------------------------------- embeddings
    def _vectors(self, mapping):
        w = self.module.node_embedding.weight.detach().cpu().numpy()
        return {name: w[i] for name, i in mapping.items()}

    @property
    def class_embeddings(self):
        """Vector of every named class.

        :rtype: dict
        """
        return self._vectors(self.lattice.class_to_id)

    @property
    def individual_embeddings(self):
        """Vector of every :math:`\\{a\\}`.

        :rtype: dict
        """
        return self._vectors(self.lattice.individual_to_id)

    @property
    def node_embeddings(self):
        """Vector of every lattice element, by element name.

        :rtype: dict
        """
        return self._vectors(self.lattice.name_to_id)
