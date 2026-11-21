"""Binary code box embeddings of projected ontology graphs [zhang2022bcbox]_."""

import os

import numpy as np
import torch as th
import torch.nn as nn

from mowl.base_models.graph_model import GraphModel
from mowl.nn.graph.bcbox.losses import bce_negative_sampling_loss
from mowl.nn.graph.bcbox.module import BinaryCodeBoxModule


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


class BinaryCodeBox(GraphModel):
    """Binary code box embeddings (GBC-Box / VBC-Box) of a projected ontology graph.

    Reimplementation of Zhang, Boratko, Musco and McCallum, *Modeling Transitivity and
    Cyclicity in Directed Graphs via Binary Code Box Embeddings*, NeurIPS 2022
    [zhang2022bcbox]_. The geometry, energies and regulariser are in
    :class:`mowl.nn.graph.bcbox.BinaryCodeBoxModule`.

    **Why a** :class:`~mowl.base_models.GraphModel` **and not an**
    :class:`~mowl.base_models.EmbeddingELModel`. The paper models one simple directed graph
    :math:`G = (V, E)` and nothing else: there is no conjunction, no existential
    restriction and no notion of a role. An ``EmbeddingELModel`` restricted to ``gci0``
    would carry seven unused normal forms, the EL normaliser and its auxiliary classes, and
    would tie the method to the atomic subsumption graph. As a ``GraphModel`` it embeds the
    edges produced by any :class:`~mowl.projection.base.ProjectionModel`: by default the
    :class:`~mowl.projection.TaxonomyProjector` (the asserted atomic subsumption graph,
    i.e. the ``gci0`` axioms between named classes), but equally a relation projection
    whose edges are not transitive and may form cycles, which is the case the paper was
    designed for.

    **From projected edges to the paper's graph.** All projected edges (optionally only
    those whose label is in ``relations``) are merged into one simple directed graph:
    labels are dropped, duplicates and self-loops are removed (paper, footnote 2). In the
    paper an edge :math:`(u, v)` has low energy when the box of :math:`v` lies inside the
    box of :math:`u` (Eq. 3 normalises by the volume of :math:`v`), so edges point from the
    general to the specific vertex. mOWL edges point the other way (``C -> D`` for
    :math:`C \\sqsubseteq D`), so a projected edge ``src -> dst`` is trained as the paper's
    edge ``(dst, src)``: the box of ``src`` should lie in the box of ``dst``.
    :meth:`score_edges` and :meth:`edge_energy` take edges in mOWL direction.

    **Training (Section 4.4).** Binary cross-entropy of Eq. (10) over minibatches of
    positive edges; every positive gets ``num_negs`` negatives (32 in the paper) obtained by
    replacing its source or its target by a uniformly drawn vertex; corruptions that hit a
    training edge or a self-loop are discarded (masked). Negatives are weighted
    self-adversarially (Sun et al., 2019) as the paper states. The loss adds
    :math:`w_r \\lVert \\mathbf{1} - B \\rVert_{\\ell_1}` (Section 4.5) with
    :math:`w_r` = ``code_reg``. Adam is used.

    **Deviations / unspecified details.** (i) The paper does not say how codes are kept in
    :math:`[0, 1]`; mOWL uses a sigmoid of a free logit (see the module). (ii) The paper does
    not give the form of the self-adversarial weights; mOWL uses RotatE's softmax over the
    negatives' logits with inverse temperature ``adversarial_temperature`` and no gradient
    through the weights, which makes the negative term a weighted average rather than the
    plain sum of Eq. (10) (set ``self_adversarial=False`` for the plain sum). (iii) For
    VBC-Box the lasso is summed over the codes of all vertices at every step. (iv) The
    paper's G-Box parameters are the corner means directly (Eq. 3, Appendix A); no Euler
    constant offset is applied.

    :param dataset: mOWL dataset; its training ontology is projected.
    :type dataset: :class:`mowl.datasets.Dataset`
    :param embed_dim: Box dimension :math:`d`.
    :type embed_dim: int
    :param variant: ``"vbc"`` (per-vertex codes, Eq. 9), ``"gbc"`` (global codes, Eq. 8) or
        ``"gbox"`` (plain Gumbel boxes, Eq. 3, the baseline without codes).
    :type variant: str
    :param num_codes: Number :math:`k` of global codes (``"gbc"`` only).
    :type num_codes: int
    :param code_dims: :math:`d_{bin}`: number of dimensions with trainable codes (the last
        ones); ``None`` means all.
    :type code_dims: int, optional
    :param intersection_temp: :math:`\\tau`.
    :type intersection_temp: float
    :param volume_temp: :math:`\\nu`.
    :type volume_temp: float
    :param code_reg: Lasso weight :math:`w_r`.
    :type code_reg: float
    :param code_init: Mean of the initial code logits.
    :type code_init: float
    :param num_negs: Negatives per positive edge.
    :type num_negs: int
    :param self_adversarial: Use self-adversarial negative weights.
    :type self_adversarial: bool
    :param adversarial_temperature: Inverse temperature of the self-adversarial weights.
    :type adversarial_temperature: float
    :param learning_rate: Adam learning rate.
    :type learning_rate: float
    :param batch_size: Number of positive edges per minibatch.
    :type batch_size: int
    :param relations: If given, keep only projected edges with these labels.
    :type relations: list of str, optional
    :param projector: Projection of the ontology to a graph. Defaults to
        :class:`~mowl.projection.TaxonomyProjector`.
    :type projector: :class:`~mowl.projection.base.ProjectionModel`, optional
    :param model_filepath: Where :meth:`train` saves the module's state dict.
    :type model_filepath: str, optional
    :param device: Torch device.
    :type device: str
    """

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
    def init_module(self):
        """Create the :class:`~mowl.nn.graph.bcbox.BinaryCodeBoxModule`."""
        self._module = BinaryCodeBoxModule(
            self.num_nodes, self.embed_dim, variant=self.variant, num_codes=self.num_codes,
            code_dims=self.code_dims, intersection_temp=self.intersection_temp,
            volume_temp=self.volume_temp, code_init=self.code_init).to(self.device)

    @property
    def module(self):
        if self._module is None:
            self.init_module()
        return self._module

    def sample_negatives(self, pos):
        """Corrupt the source or the target of every positive edge ``num_negs`` times.

        :param pos: Positive edges (paper direction), shape ``(b, 2)``.
        :type pos: :class:`torch.Tensor`
        :returns: ``(neg, mask)`` with ``neg`` of shape ``(b, num_negs, 2)`` and a boolean
            ``mask`` that is ``False`` where the corruption is a training edge or a self-loop.
        :rtype: tuple
        """
        b, n = pos.shape[0], self.num_nodes
        neg = pos.unsqueeze(1).repeat(1, self.num_negs, 1)
        column = th.randint(0, 2, (b, self.num_negs), device=pos.device)
        random_nodes = th.randint(0, n, (b, self.num_negs), device=pos.device)
        neg[..., 0] = th.where(column == 0, random_nodes, neg[..., 0])
        neg[..., 1] = th.where(column == 1, random_nodes, neg[..., 1])
        keys = neg[..., 0] * n + neg[..., 1]
        known = self.edge_index.to(pos.device)
        known_keys = known[:, 0] * n + known[:, 1]
        mask = ~th.isin(keys, known_keys) & (neg[..., 0] != neg[..., 1])
        return neg, mask

    def compute_loss(self, pos):
        """Loss of Eq. (10) plus the code lasso for one minibatch of positive edges."""
        neg, mask = self.sample_negatives(pos)
        pos_energy = self.module(pos)
        neg_energy = self.module(neg)
        loss = bce_negative_sampling_loss(
            pos_energy, neg_energy, neg_mask=mask, self_adversarial=self.self_adversarial,
            adversarial_temperature=self.adversarial_temperature).mean()
        if self.code_reg > 0:
            loss = loss + self.code_reg * self.module.code_regularization()
        return loss

    def train(self, epochs=100, verbose=False):
        """Train with Adam on minibatches of positive edges.

        :param epochs: Number of passes over the edges.
        :type epochs: int
        :param verbose: Print the loss of every epoch.
        :type verbose: bool
        :returns: The mean training loss of every epoch.
        :rtype: list of float
        """
        module = self.module
        edges = self.edge_index.to(self.device)
        optimizer = th.optim.Adam(module.parameters(), lr=self.learning_rate)
        history = []
        module.train()
        for epoch in range(epochs):
            perm = th.randperm(len(edges), device=self.device)
            total, batches = 0.0, 0
            for start in range(0, len(edges), self.batch_size):
                pos = edges[perm[start:start + self.batch_size]]
                optimizer.zero_grad()
                loss = self.compute_loss(pos)
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
            history.append(total / batches)
            if verbose:
                print(f"Epoch {epoch + 1}: loss {history[-1]:.4f}")
        module.eval()
        th.save(module.state_dict(), self.model_filepath)
        return history

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
        return th.exp(-self.edge_energy(src, dst))

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
        module = self.module
        out = {}
        for cls in self.dataset.classes.as_str:
            if cls not in node_to_id:
                continue
            i = node_to_id[cls]
            parts = [module.box_min[i], module.box_max[i]]
            if module.variant == "vbc":
                parts.append(module.vertex_codes(th.tensor(i, device=module.box_min.device)))
            out[cls] = th.cat(parts).detach().cpu().numpy()
        return out


def best_f1(energies, adjacency):
    """Best F1 over all global thresholds :math:`T` of the rule :math:`E < T`.

    This is the reconstruction measure of the paper (Section 5.1: "optimal F1 score").
    Infinite energies (for example the diagonal) are never predicted as edges.

    :param energies: Energy of every candidate pair, any shape.
    :type energies: :class:`torch.Tensor`
    :param adjacency: Boolean tensor of the same shape marking the true edges.
    :type adjacency: :class:`torch.Tensor`
    :returns: ``(f1, threshold)``.
    :rtype: tuple
    """
    e = energies.flatten().double()
    y = adjacency.flatten().bool()
    finite = th.isfinite(e)
    e, y_f = e[finite], y[finite]
    positives = int(y.sum())
    if positives == 0 or len(e) == 0:
        return 0.0, float("nan")
    order = th.argsort(e)
    e, y_f = e[order], y_f[order]
    tp = th.cumsum(y_f.double(), 0)
    k = th.arange(1, len(e) + 1, dtype=th.double)
    f1 = 2 * tp / (k + positives)
    # only cut between distinct energy values
    distinct = th.ones_like(y_f)
    distinct[:-1] = e[1:] > e[:-1]
    f1 = th.where(distinct, f1, th.zeros_like(f1))
    best = int(th.argmax(f1))
    upper = e[best + 1].item() if best + 1 < len(e) else float("inf")
    threshold = (e[best].item() + upper) / 2 if np.isfinite(upper) else e[best].item() + 1.0
    return float(f1[best]), threshold
