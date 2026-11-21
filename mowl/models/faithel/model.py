"""FaithEL: strongly TBox faithful embeddings with learned ABox parameters.

Follows the authors' implementation (github.com/victorlacerdab/FaithEL_method, which
accompanies Lacerda, Ozaki & Guimarães, RuleML+RR 2024; the paper text itself was not
available, only that code and the TGDK article it builds on).

The TBox is not learned. The geometry of concepts and roles is the convex interpretation
of the Definition 26 canonical model (:class:`~mowl.geometric.faithfulness.
StronglyFaithfulModel`, region family ``"convex"``), so every TBox axiom of the supported
forms holds exactly when it is entailed. Only the ABox is learned: one vector per
individual, and per concept a "moving" parameter, per role a head and a tail parameter,
all initialised at the centroid of the region and kept within ``radius`` of it.
"""
import logging

import numpy as np
import torch as th
from torch import nn

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.geometric.faithfulness import StronglyFaithfulModel
from mowl.geometric.normal_forms import (BOTTOM, NF1, NF2, NF3, NF4, TOP, NormalisedOntology,
                                         is_nominal_name, normalise)

logger = logging.getLogger(__name__)


def _tbox_only(no):
    """The normalised ontology without ABox axioms (those mentioning a nominal name)."""
    def nominal(ax):
        return any(isinstance(v, str) and is_nominal_name(v) for v in vars(ax).values())

    return NormalisedOntology(
        axioms=[ax for ax in no.axioms if not nominal(ax)],
        concept_names={c for c in no.concept_names if not is_nominal_name(c)},
        role_names=set(no.role_names), dropped_axioms=no.dropped_axioms)


class FaithEL(EmbeddingELModel):
    """FaithEL: a fixed strongly faithful convex TBox model and learned ABox embeddings.

    Individuals live in the space of the canonical-model simplex (dimension ``n - 1`` for
    ``n`` canonical elements, so ``embed_dim`` is not a free parameter). The loss, as in
    the authors' code, is a mean squared error: individuals are pulled to the centroid of
    the region of each concept they belong to (and role pairs to the centroid of the role
    region), the moving concept and role parameters are pulled to the individuals and to
    the centroid (weights ``gamma`` and ``phi``), and a uniformly drawn individual is
    pushed away with weight ``psi`` (``neg_sampling``).

    :param dataset: Dataset with individuals; its ontology gives the TBox.
    :param radius: Maximal distance of a moving parameter from its region centroid
        (default ``0.6``, the code's ``SCALE_FACTOR / 2 + 0.1``).
    :param phi: Weight of the parameter-to-centroid term.
    :param gamma: Weight of the parameter-to-individual terms.
    :param psi: Weight of the negative term.
    :param neg_sampling: Use the negative term (the code's default).
    :param centroid_score: Score an assertion by the distance to the parameter plus the
        distance to the centroid (the code's default) instead of the parameter only.
    :param batch_size: Assertions per step.
    :param learning_rate: Adam step size.
    :param max_elements: Refuse ontologies whose Definition 26 canonical model would have
        more elements than this (it has one per pair of concept names).

    Without individuals the model has nothing to train; its TBox scores (``gci0``-``gci3``)
    are still available.
    """

    axiom_scoring = "method"
    scored_forms = ("gci0", "gci1", "gci2", "gci3", "class_assertion",
                    "object_property_assertion")

    def __init__(self, dataset, radius=0.6, phi=1.0, gamma=1.0, psi=0.7, neg_sampling=True,
                 centroid_score=True, batch_size=512, learning_rate=0.001,
                 model_filepath=None, device="cpu", seed=None, max_elements=3000):
        no = _tbox_only(normalise(dataset.ontology))
        n = len([c for c in no.concept_names if c not in (TOP, BOTTOM)])
        size = 1 + n + n * (n - 1) // 2 + len(no.role_names) * (n + 1)
        if size > max_elements:
            raise ValueError(
                f"The canonical model of Definition 26 has about {size} elements (one per "
                f"pair of concept names and per role and concept), more than max_elements="
                f"{max_elements}; the dimension of FaithEL grows with it. Use a smaller "
                f"ontology or raise max_elements.")
        self.tbox = StronglyFaithfulModel(no, regions="convex")
        d = self.tbox.dimension
        super().__init__(dataset, embed_dim=max(d, 1), batch_size=batch_size,
                         extended=True, model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate)
        self.radius, self.phi, self.gamma, self.psi = radius, phi, gamma, psi
        self.neg_sampling = neg_sampling
        self.centroid_score = centroid_score
        self.seed = seed
        self._cost_cache = {}
        self.init_module()

    # -- geometry ------------------------------------------------------------------------
    def _centroids(self):
        eta, d = self.tbox.eta, self.tbox.dimension
        cents = np.zeros((len(self.class_index_dict), d), dtype=np.float32)
        for name, idx in self.class_index_dict.items():
            pts = eta.class_region(name).vertices
            if len(pts):
                cents[idx] = pts.mean(axis=0)
        rcents = np.zeros((len(self.object_property_index_dict), 2 * d), dtype=np.float32)
        for name, idx in self.object_property_index_dict.items():
            pts = eta.role_region(name).points
            if len(pts):
                rcents[idx] = pts.mean(axis=0)
        return cents, rcents

    def init_module(self):
        cents, rcents = self._centroids()
        d = self.tbox.dimension
        if self.seed is not None:
            th.manual_seed(self.seed)
        module = nn.Module()
        module.register_buffer("concept_centroid", th.tensor(cents))
        module.register_buffer("role_centroid", th.tensor(rcents))
        module.individuals = nn.Embedding(len(self.individual_index_dict), d)
        module.concepts = nn.Embedding.from_pretrained(th.tensor(cents), freeze=False)
        module.role_head = nn.Embedding.from_pretrained(th.tensor(rcents[:, :d]), freeze=False)
        module.role_tail = nn.Embedding.from_pretrained(th.tensor(rcents[:, d:]), freeze=False)
        with th.no_grad():
            module.individuals.weight.normal_(mean=0.5, std=0.7)
        self.module = module.to(self.device)

    @property
    def individual_vectors(self):
        return self.module.individuals.weight.detach().cpu().numpy()

    def _project(self):
        """Keep each moving parameter within ``radius`` of the centroid of its region."""
        m = self.module
        with th.no_grad():
            for emb, cen in ((m.concepts, m.concept_centroid),):
                diff = emb.weight - cen
                dist = diff.norm(dim=1, keepdim=True).clamp_min(1e-12)
                emb.weight.copy_(cen + diff * th.clamp(self.radius / dist, max=1.0))
            both = th.cat([m.role_head.weight, m.role_tail.weight], dim=1)
            diff = both - m.role_centroid
            dist = diff.norm(dim=1, keepdim=True).clamp_min(1e-12)
            both = m.role_centroid + diff * th.clamp(self.radius / dist, max=1.0)
            d = self.tbox.dimension
            m.role_head.weight.copy_(both[:, :d])
            m.role_tail.weight.copy_(both[:, d:])

    # -- loss ----------------------------------------------------------------------------
    def _concept_loss(self, rows):
        m, mse = self.module, nn.functional.mse_loss
        x, p, c = m.individuals(rows[:, 0]), m.concepts(rows[:, 1]), m.concept_centroid[rows[:, 1]]
        loss = mse(x, c) + self.gamma * mse(p, x) + self.phi * mse(p, c)
        if self.neg_sampling:
            neg = m.individuals(th.randint(0, m.individuals.num_embeddings, (len(rows),),
                                           device=rows.device))
            loss = loss - self.psi * mse(x.detach(), neg)
        return loss

    def _role_loss(self, rows):
        m, mse = self.module, nn.functional.mse_loss
        xs, xo = m.individuals(rows[:, 0]), m.individuals(rows[:, 2])
        ph, pt, c = m.role_head(rows[:, 1]), m.role_tail(rows[:, 1]), m.role_centroid[rows[:, 1]]
        pair = th.cat([xs, xo], dim=1)
        loss = mse(pair, c)
        loss = loss + self.gamma * mse(th.cat([ph, xo], dim=1), pair)
        loss = loss + self.gamma * mse(th.cat([xs, pt], dim=1), pair)
        loss = loss + self.phi * mse(th.cat([ph, pt], dim=1), c)
        if self.neg_sampling:
            neg = m.individuals(th.randint(0, m.individuals.num_embeddings, (len(rows),),
                                           device=rows.device))
            loss = loss - self.psi * mse(pair.detach(), th.cat([xs.detach(), neg], dim=1))
        return loss

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train the individual embeddings and the moving parameters for ``epochs`` epochs.

        Concept and role assertions are alternated per step as in the authors' code; the
        projection to the radius is applied after every step. ``self.loss_history`` holds
        the mean loss of each epoch.
        """
        data = {}
        for name, width in (("class_assertion", 2), ("object_property_assertion", 3)):
            ds = self.training_datasets.get(name)
            data[name] = ds[:].to(self.device) if ds is not None and len(ds) else \
                th.empty((0, width), dtype=th.long, device=self.device)
        opt = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        self.loss_history = []
        for epoch in range(epochs):
            total, steps = 0.0, 0
            for name, fn in (("class_assertion", self._concept_loss),
                             ("object_property_assertion", self._role_loss)):
                rows = data[name]
                perm = th.randperm(len(rows), device=self.device)
                for s in range(0, len(rows), self.batch_size):
                    opt.zero_grad()
                    loss = fn(rows[perm[s:s + self.batch_size]])
                    loss.backward()
                    opt.step()
                    self._project()
                    total += float(loss.detach())
                    steps += 1
            self.loss_history.append(total / max(steps, 1))
            if epoch_callback is not None:
                epoch_callback(epoch, self)
        if self.model_filepath:
            th.save(self.module.state_dict(), self.model_filepath)

    # -- scores --------------------------------------------------------------------------
    def class_assertion_costs(self, rows):
        """Cost of ``C(a)`` for rows ``(a, C)``: distance to the concept parameter, plus (with
        ``centroid_score``) the distance to the region centroid. Lower is more plausible."""
        m = self.module
        x = m.individuals(rows[:, 0])
        cost = (x - m.concepts(rows[:, 1])).norm(dim=1)
        if self.centroid_score:
            cost = cost + (x - m.concept_centroid[rows[:, 1]]).norm(dim=1)
        return cost

    def role_assertion_costs(self, rows):
        """Cost of ``r(a, b)`` for rows ``(a, r, b)`` (same scheme, on concatenated vectors)."""
        m = self.module
        pair = th.cat([m.individuals(rows[:, 0]), m.individuals(rows[:, 2])], dim=1)
        par = th.cat([m.role_head(rows[:, 1]), m.role_tail(rows[:, 1])], dim=1)
        cost = (pair - par).norm(dim=1)
        if self.centroid_score:
            cost = cost + (pair - m.role_centroid[rows[:, 1]]).norm(dim=1)
        return cost

    def tbox_costs(self, gci_name, rows):
        """0 for an axiom that holds in the fixed convex TBox model (hence exactly the
        entailed ones, over the signature), 1 otherwise. An entity unknown to the TBox gets 1."""
        inv_c = {i: n for n, i in self.class_index_dict.items()}
        inv_r = {i: n for n, i in self.object_property_index_dict.items()}
        sig, rsig = self.tbox.signature, self.tbox.role_signature
        known = sig | {TOP}
        out = []
        for row in rows.tolist():
            key = (gci_name, tuple(row))
            if key not in self._cost_cache:
                if gci_name == "gci0":
                    a, b = inv_c[row[0]], inv_c[row[1]]
                    ok = a in known and b in known
                    ax = NF1(a, b)
                elif gci_name == "gci1":
                    a, b, c = (inv_c[i] for i in row)
                    ok = all(n in known for n in (a, b, c))
                    ax = NF2(a, b, c)
                elif gci_name == "gci2":
                    a, r, b = inv_c[row[0]], inv_r[row[1]], inv_c[row[2]]
                    ok = a in known and b in known and r in rsig
                    ax = NF3(a, r, b)
                else:
                    r, a, b = inv_r[row[0]], inv_c[row[1]], inv_c[row[2]]
                    ok = a in known and b in known and r in rsig
                    ax = NF4(r, a, b)
                self._cost_cache[key] = 0.0 if ok and self.tbox.satisfies(ax) else 1.0
            out.append(self._cost_cache[key])
        return th.tensor(out, dtype=th.float32, device=rows.device)

    def axiom_costs(self, gci_name, rows):
        """Costs of candidate axioms (rows in mOWL's normal-form column layout)."""
        with th.no_grad():
            if gci_name == "class_assertion":
                return self.class_assertion_costs(rows)
            if gci_name == "object_property_assertion":
                return self.role_assertion_costs(rows)
            if gci_name in ("gci0", "gci1", "gci2", "gci3"):
                return self.tbox_costs(gci_name, rows)
        from mowl.evaluation.model_scoring import UnsupportedAxiomForm
        raise UnsupportedAxiomForm(f"FaithEL has no score for {gci_name}")
