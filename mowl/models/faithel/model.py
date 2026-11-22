"""FaithEL: strongly TBox faithful embeddings with learned ABox parameters.

Follows the authors' implementation (github.com/victorlacerdab/FaithEL_method, which
accompanies Lacerda, Ozaki & Guimarães, RuleML+RR 2024; the paper text itself was not
available, only that code and the TBox article it builds on).

The model is a combination of parts (:mod:`mowl.representation.library.solved_faithel`): the
TBox is not learned but constructed, the convex interpretation of the Definition 26 canonical
model (:class:`~mowl.geometric.faithfulness.StronglyFaithfulModel`, region family
``"convex"``), so every TBox axiom of the supported forms holds exactly when it is entailed.
Only the ABox is learned: one vector per individual, and per concept a "moving" parameter, per
role a head and a tail parameter, all initialised at the centroid of the region and kept
within ``radius`` of it.
"""
import logging

import torch as th

from functools import partial

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.geometric.faithfulness import StronglyFaithfulModel
from mowl.geometric.normal_forms import (BOTTOM, TOP, NormalisedOntology, is_nominal_name,
                                         normalise)
from mowl.representation.library.solved_faithel import (FaithfulTBox, faith_centroids,
                                                        faithel as _faithel)
from mowl.training.alternating import AlternatingMinibatches
from mowl.training.loop import RegimeTrained
from mowl.training.regularization import ModuleProjection, Regularization

logger = logging.getLogger(__name__)

__all__ = ["FaithEL"]


def _tbox_only(no):
    """The normalised ontology without ABox axioms (those mentioning a nominal name)."""
    def nominal(ax):
        return any(isinstance(v, str) and is_nominal_name(v) for v in vars(ax).values())

    return NormalisedOntology(
        axioms=[ax for ax in no.axioms if not nominal(ax)],
        concept_names={c for c in no.concept_names if not is_nominal_name(c)},
        role_names=set(no.role_names), dropped_axioms=no.dropped_axioms)


class FaithEL(RegimeTrained, EmbeddingELModel):
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
    :param negative_term: ``"unbounded"`` (default of this named method): the code's
        :math:`-\\psi\\,\\mathrm{mse}(x_a, x_{neg})`, which has no lower bound, so with
        ``neg_sampling`` the loss can go to :math:`-\\infty` as individuals drift apart.
        ``"margin"``: the bounded hinge of the part
        (:class:`~mowl.representation.library.solved_faithel.AnchoredObjective`, the default of
        the part).
    :param neg_margin: The margin of ``"margin"``.
    :param centroid_score: Score an assertion by the distance to the parameter plus the
        distance to the centroid (the code's default) instead of the parameter only.
    :param batch_size: Assertions per step.
    :param learning_rate: Adam step size.
    :param max_elements: Refuse ontologies whose Definition 26 canonical model would have
        more elements than this (it has one per pair of concept names).
    :param training: The batching of the training (default: the authors' alternating
        minibatches, :class:`~mowl.training.AlternatingMinibatches`; :class:`~mowl.training.
        FullBatch` takes one step per epoch). The training of the method is this regime and
        the projection of the parameters into the balls around the centroids
        (:class:`~mowl.training.Regularization` of a :class:`~mowl.training.ModuleProjection`),
        read ``model.training_regime``.

    Without individuals the model has nothing to train; its TBox scores (``gci0``-``gci3``)
    are still available.
    """

    save_always = True

    axiom_scoring = "method"
    scored_forms = ("gci0", "gci1", "gci2", "gci3", "class_assertion",
                    "object_property_assertion")

    def __init__(self, dataset, radius=0.6, phi=1.0, gamma=1.0, psi=0.7, neg_sampling=True,
                 centroid_score=True, batch_size=512, learning_rate=0.001,
                 model_filepath=None, device="cpu", seed=None, max_elements=3000,
                 training=None, negative_term="unbounded", neg_margin=1.0):
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
        self.negative_term, self.neg_margin = negative_term, neg_margin
        self.centroid_score = centroid_score
        self.seed = seed
        #: The batching of the training (:class:`~mowl.training.AlternatingMinibatches`: the
        #: authors' loop; :class:`~mowl.training.FullBatch`).
        self.training = training if training is not None else AlternatingMinibatches()
        self.init_module()

    # -- the parts -----------------------------------------------------------------------
    def _centroids(self):
        return faith_centroids(self.tbox, self.class_index_dict,
                               self.object_property_index_dict)

    def init_module(self):
        cents, rcents = self._centroids()
        d = self.tbox.dimension
        if self.seed is not None:
            th.manual_seed(self.seed)
        names = {i: n for n, i in self.class_index_dict.items()}
        roles = {i: n for n, i in self.object_property_index_dict.items()}
        individuals = {i: n for n, i in self.individual_index_dict.items()}
        tbox = FaithfulTBox(self.tbox, [names[i] for i in sorted(names)],
                            [roles[i] for i in sorted(roles)],
                            [individuals[i] for i in sorted(individuals)])
        module = _faithel(
            len(self.class_index_dict), len(self.object_property_index_dict),
            len(self.individual_index_dict), embed_dim=d, radius=self.radius, phi=self.phi,
            gamma=self.gamma, psi=self.psi, neg_sampling=self.neg_sampling,
            centroid_score=self.centroid_score, tbox=tbox,
            negative_term=self.negative_term, neg_margin=self.neg_margin, centroids=cents,
            role_centroids=rcents)
        self.module = module.to(self.device)

    @property
    def individual_vectors(self):
        return self.module.individual_model.centres.weight.detach().cpu().numpy()

    def _project(self):
        """Keep each moving parameter within ``radius`` of the centroid of its region."""
        self.module.project()

    @property
    def default_regimes(self):
        """The training of the method: the batching of :attr:`training` and the projection of
        the moving parameters into the balls around the centroids after every step."""
        return (lambda: self.training, partial(Regularization, ModuleProjection()))

    @property
    def loss_history(self):
        """The mean loss of each epoch of the last call of :meth:`train`."""
        return getattr(self, "training_losses", [])

    # -- scores --------------------------------------------------------------------------
    def class_assertion_costs(self, rows):
        """Cost of ``C(a)`` for rows ``(a, C)``: distance to the concept parameter, plus (with
        ``centroid_score``) the distance to the region centroid. Lower is more plausible."""
        return self.module.class_assertion_costs(rows)

    def role_assertion_costs(self, rows):
        """Cost of ``r(a, b)`` for rows ``(a, r, b)`` (same scheme, on concatenated vectors)."""
        return self.module.role_assertion_costs(rows)

    def tbox_costs(self, gci_name, rows):
        """0 for an axiom that holds in the fixed convex TBox model (hence exactly the
        entailed ones, over the signature), 1 otherwise. An entity unknown to the TBox gets 1."""
        return self.module.tbox.cost(gci_name, rows.cpu()).reshape(-1).to(rows.device)

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
