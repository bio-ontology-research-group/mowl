"""What binary-code box models share, whatever trains them: the projected graph, the scoring
and the evaluation adapters. :class:`mowl.models.BinaryCodeBox` trains with the regimes of
:mod:`mowl.training`; the direct implementation in :mod:`mowl.baselines.models.bcbox` keeps
its own loop for the comparison in ``tests/baselines``."""

import os

import torch as th
import torch.nn as nn

from mowl.base_models.graph_model import GraphModel


class _SubsumptionScorer(nn.Module):
    """Adapter for :class:`mowl.evaluation.SubsumptionEvaluator`.

    The evaluator calls ``model(batch, "gci0")`` with columns ``(sub, super)`` given as
    indices of ``dataset.classes`` and ranks ascending; this returns the energy of the
    corresponding graph edge ``sub -> super`` (lower is better).
    """

    def __init__(self, module, class_to_node):
        super().__init__()
        self.module = module
        self.register_buffer("class_to_node", class_to_node)

    def forward(self, batch, gci_name="gci0"):
        if gci_name != "gci0":
            raise ValueError("BinaryCodeBox only scores atomic subsumptions ('gci0').")
        sub = self.class_to_node[batch[:, 0]]
        sup = self.class_to_node[batch[:, -1]]
        return self.module.energy(sup, sub)


class BinaryCodeBoxBase(GraphModel):
    """The graph, the scoring and the evaluation of binary code box models (see
    :class:`mowl.models.BinaryCodeBox`). A subclass makes ``init_module`` and trains."""


    axiom_scoring = "evaluation_model"
    scored_forms = ("gci0",)

    def __init__(self, dataset, embed_dim=32, variant="vbc", num_codes=2, code_dims=None,
                 intersection_temp=0.01, volume_temp=1.0, code_reg=0.0, code_init=0.0,
                 num_negs=32, self_adversarial=True, adversarial_temperature=1.0,
                 learning_rate=0.01, batch_size=1024, relations=None, projector=None,
                 model_filepath=None, device="cpu"):
        super().__init__(dataset, model_filepath=model_filepath)
        if projector is None:
            from mowl.projection import TaxonomyProjector
            projector = TaxonomyProjector()
        self.set_projector(projector)

        self.embed_dim = embed_dim
        self.variant = variant
        self.num_codes = num_codes
        self.code_dims = code_dims
        self.intersection_temp = intersection_temp
        self.volume_temp = volume_temp
        self.code_reg = code_reg
        self.code_init = code_init
        self.num_negs = num_negs
        self.self_adversarial = self_adversarial
        self.adversarial_temperature = adversarial_temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.relations = None if relations is None else set(relations)
        self.device = device

        self._edge_index = None
        self._module = None

    # ------------------------------------------------------------------ graph
    @property
    def edge_index(self):
        """Training edges in the paper's direction, shape ``(m, 2)``: row ``(u, v)`` means
        the box of ``v`` should lie in the box of ``u``. For a projected edge ``src -> dst``
        this is ``(dst, src)``.

        :rtype: :class:`torch.Tensor`
        """
        if self._edge_index is None:
            node_to_id = self.graph_node_to_id
            pairs = set()
            for edge in self.edges:
                if self.relations is not None and edge.rel not in self.relations:
                    continue
                if edge.src == edge.dst:
                    continue
                pairs.add((node_to_id[edge.dst], node_to_id[edge.src]))
            if not pairs:
                raise ValueError("The projected graph has no edges to train on.")
            self._edge_index = th.tensor(sorted(pairs), dtype=th.long)
        return self._edge_index

    @property
    def num_nodes(self):
        return len(self.graph_node_to_id)

    # ----------------------------------------------------------------- module
    @property
    def module(self):
        if self._module is None:
            self.init_module()
        return self._module

    def from_pretrained(self, file_name):
        """Load a state dict saved by :meth:`train`.

        :param file_name: Path to the saved state dict.
        :type file_name: str
        """
        if not isinstance(file_name, str):
            raise TypeError("Parameter file_name must be a string.")
        if not os.path.exists(file_name):
            raise FileNotFoundError("Pretrained model path does not exist")
        self.module.load_state_dict(th.load(file_name, weights_only=True))

    # ---------------------------------------------------------------- scoring
    def _ids(self, names):
        node_to_id = self.graph_node_to_id
        return th.tensor([node_to_id[str(x)] for x in names], dtype=th.long,
                         device=self.device)

    @th.no_grad()
    def edge_energy(self, src, dst):
        """Energy of the edges ``src -> dst`` in mOWL direction (``src`` below ``dst``).

        :param src: Node names (IRIs) or a tensor of node ids.
        :param dst: Node names (IRIs) or a tensor of node ids.
        :rtype: :class:`torch.Tensor`
        """
        if not th.is_tensor(src):
            src = self._ids(src)
        if not th.is_tensor(dst):
            dst = self._ids(dst)
        return self.module.energy(dst.to(self.device), src.to(self.device))

    @th.no_grad()
    def score_edges(self, src, dst):
        """Edge probability :math:`P = e^{-E}` of ``src -> dst`` (mOWL direction)."""
        return self.module.measure.probability(self.edge_energy(src, dst))

    @th.no_grad()
    def energy_matrix(self):
        """Energies of all ordered vertex pairs in mOWL direction: entry ``[a, b]`` is the
        energy of the edge ``a -> b``. Self-pairs are set to ``inf`` (paper, footnote 2).

        :rtype: :class:`torch.Tensor`
        """
        n = self.num_nodes
        a = th.arange(n, device=self.device)
        src = a.repeat_interleave(n)
        dst = a.repeat(n)
        energies = self.edge_energy(src, dst).view(n, n)
        energies.fill_diagonal_(float("inf"))
        return energies

    def score(self, axiom):
        """Probability of an atomic subsumption ``SubClassOf(C, D)`` between named classes.

        :param axiom: The axiom.
        :type axiom: :class:`org.semanticweb.owlapi.model.OWLSubClassOfAxiom`
        :rtype: float
        """
        from org.semanticweb.owlapi.model import OWLSubClassOfAxiom
        if not isinstance(axiom, OWLSubClassOfAxiom):
            raise TypeError("BinaryCodeBox scores only SubClassOf axioms.")
        sub, sup = axiom.getSubClass(), axiom.getSuperClass()
        if sub.isAnonymous() or sup.isAnonymous():
            raise ValueError("BinaryCodeBox scores only subsumptions between named classes.")
        sub = str(sub.asOWLClass().toStringID())
        sup = str(sup.asOWLClass().toStringID())
        return self.score_edges([sub], [sup]).item()

    @property
    def evaluation_model(self):
        """Adapter for :class:`mowl.evaluation.SubsumptionEvaluator` (scores ``gci0`` pairs of
        dataset class indices by their edge energy)."""
        if self._evaluation_model is None:
            node_to_id = self.graph_node_to_id
            mapping = [node_to_id.get(c, 0) for c in self.dataset.classes.as_str]
            self._evaluation_model = _SubsumptionScorer(
                self.module, th.tensor(mapping, dtype=th.long)).to(self.device)
        return self._evaluation_model

    @property
    def class_embeddings(self):
        """Box of every class as ``concat(box_min, box_max)`` (and, for ``"vbc"``, the
        vertex code appended).

        :rtype: dict
        """
        node_to_id = self.graph_node_to_id
        names = [c for c in self.dataset.classes.as_str if c in node_to_id]
        ids = th.tensor([node_to_id[c] for c in names], dtype=th.long,
                        device=self.device)
        vectors = self.module.class_model.vectors(ids).detach().cpu().numpy()
        return dict(zip(names, vectors))
