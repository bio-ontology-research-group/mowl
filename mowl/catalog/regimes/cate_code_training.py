"""Regime card: the training design of the CatE reference code."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import ZHAPA2023

REGIMES = (
    RegimeCard(
        key="cate-code-training",
        assumption="closed world over stated axioms",
        name="Training design of the CatE code",
        kind=RegimeKind.TRAINING,
        summary=(
            "One epoch is a pass over all morphisms of the lattice in random order, in "
            "minibatches; every morphism gets num_negs corruptions of its "
            "superconcept, drawn uniformly over all lattice elements and not filtered "
            "even when they are morphisms; the loss is Eq. 7 of the paper as the code "
            "computes it (positive violations plus a hinge on the mean violation of "
            "the corruptions); Adam with weight decay 1e-4 and a triangular cyclic "
            "learning rate between lr/10 and lr (20 steps per half cycle, stepped per "
            "batch); selection by the raw validation MRR every 100 epochs with early "
            "stopping. Optionally the PPI term of the code is added: for the "
            "protein-headed morphisms of every batch, the mean violation plus a hinge "
            "on one corruption whose tail is drawn among the elements ∃r.p."),
        components=("batching", "negative sampling", "loss", "schedule",
                    "model selection"),
        applies_to="mowl.models.cate.CatE",
        references=(ZHAPA2023,),
        standalone=False,
        introduced_with=("cate",),
        used_by=("cate",),
        implementation="mowl.training.cate.CateCodeTraining",
        limitations=(
            "Corruptions that are morphisms of the lattice are kept as negatives, so "
            "the loss cannot reach zero: every element has the morphism X ⊑ ⊤, whose "
            "violation is 0 and whose hinge is γ. Theorem 2 of the paper (L = 0 "
            "implies order preservation) is therefore not applicable to the trained "
            "models.",
            "The reference code offers a second loss (log-sigmoid, loss_type='bpr') "
            "whose sign convention differs between src/cat.py and "
            "src/cat_ppi_evaluator.py; which loss produced the published numbers is "
            "not recorded, and only the paper's Eq. 7 is offered.",
            "Validation ranks over the named classes with the head not excluded, ties "
            "averaged, without filtering; it selects on the same quantity for every "
            "task even when the test metric is filtered.",
        ),
        notes=(
            "From src/cat.py, src/cat_completion.py, src/cat_membership.py and "
            "src/cat_ppi_evaluator.py of bio-ontology-research-group/catE (the data "
            "release of arXiv:2305.07163v3). The paper states only 'Adam with a Cyclic "
            "Learning Rate scheduler' (Section 4.1) and the hyperparameters of "
            "Appendix A, Table 6; bounds, step size, weight decay, initialisation and "
            "model selection are taken from the code. mOWL differences: epochs are "
            "counted from 1 (the code validates at epochs 0, 100, 200, ... and resets "
            "its early-stopping tolerance also when only the validation mean rank "
            "improves), and the code's released run script for the PPI task passes its "
            "arguments in the wrong order (test_batch_size lands in the norm p), so "
            "the norm used there is unknown; mOWL uses the paper's Eq. 6 norm 2."),
    ),
)
