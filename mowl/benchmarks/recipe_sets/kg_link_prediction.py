"""Recipes for the knowledge graph completion benchmarks (``kg-*-link-prediction``):
the published hyperparameters of BoxE (Abboud et al. 2020, Appendix, Table 6,
self-adversarial setting) and of ExpressivE (Pavlović and Sallinger 2023, Table 11)."""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import ABBOUD2020, PAVLOVIC2023

_NOTE = ("Trained with PyKEEN's sLCWA loop on the triples of the projected ontology "
         "(h ⊑ ∃r.t gives the triple back), Adam, the self-adversarial negative sampling "
         "loss. The publication selects the checkpoint with the best validation MRR every "
         "100 epochs; this recipe trains for the full number of epochs.")


def _nssa(margin, temperature):
    return {"loss": "nssa", "loss_kwargs": {"margin": float(margin),
                                            "adversarial_temperature": float(temperature)}}


RECIPES = (
    Recipe(benchmark="kg-fb15k237-link-prediction", method="boxe", source=ABBOUD2020,
           model_kwargs={"embed_dim": 1000, "learning_rate": 5e-5, "batch_size": 1024,
                         "num_negs_per_pos": 100, "kge_kwargs": {"p": 2, **_nssa(3, 4)}},
           epochs=1000, locator="Appendix, Table 6, FB15k-237(a)", notes=_NOTE),
    Recipe(benchmark="kg-wn18rr-link-prediction", method="boxe", source=ABBOUD2020,
           model_kwargs={"embed_dim": 500, "learning_rate": 1e-3, "batch_size": 512,
                         "num_negs_per_pos": 100, "kge_kwargs": {"p": 2, **_nssa(3, 2)}},
           epochs=1000, locator="Appendix, Table 6, WN18RR(a)", notes=_NOTE),
    Recipe(benchmark="kg-fb15k237-link-prediction", method="expressive",
           source=PAVLOVIC2023,
           model_kwargs={"embed_dim": 1000, "learning_rate": 1e-4, "batch_size": 1024,
                         "num_negs_per_pos": 150, "min_denom": 0.5,
                         "kge_kwargs": _nssa(4, 4)},
           epochs=1000, locator="Appendix, Table 11, FB15k-237",
           notes=_NOTE + " The paper does not state the number of epochs; 1000 as BoxE."),
    Recipe(benchmark="kg-wn18rr-link-prediction", method="expressive", source=PAVLOVIC2023,
           model_kwargs={"embed_dim": 500, "learning_rate": 1e-3, "batch_size": 512,
                         "num_negs_per_pos": 100, "min_denom": 0.0,
                         "kge_kwargs": _nssa(3, 2)},
           epochs=1000, locator="Appendix, Table 11, WN18RR",
           notes=_NOTE + " The paper does not state the number of epochs; 1000 as BoxE."),
)
