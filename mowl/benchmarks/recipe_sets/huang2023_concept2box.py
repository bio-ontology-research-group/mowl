"""Recipes of Concept2Box (Huang et al. 2023, Appendix B "Implementation Details") on the
DBpedia benchmarks.

Stated in the paper: Adam, the TransE instance-view score with margin 0.3, cross-view margin
0.15, instance-view vectors of dimension 256 and ontology-view boxes of dimension 512 (read
here as 256 for the centre and 256 for the offset), batch size 128, learning rate 0.005.
Not stated: the number of epochs, the weights lambda_1 and lambda_2 of the loss, the Gumbel
scale and the alpha of the distance. mOWL's defaults are used for those and the epoch number
is mOWL's choice. The paper alternates the three losses with the learning rates eta,
lambda_1 eta and lambda_2 eta (Algorithm 1); mOWL sums them (Equation 11) and replaces the
BERT encoder of the concept and relation texts by learned embeddings (see the method card).
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import HUANG2023

_NOTES = ("Adam on mOWL's generic loop with the three losses summed; no BERT encoder; the "
          "number of epochs is not stated in the paper (100 is mOWL's choice), nor are the "
          "loss weights, the Gumbel scale and alpha (mOWL defaults). gci2_reading='triple': "
          "the paper's triple (h, r, t) puts the tail box inside the head box.")

RECIPES = tuple(
    Recipe(benchmark=benchmark, method="concept2box", source=HUANG2023,
           model_kwargs={"embed_dim": 256, "kge_margin": 0.3, "cross_margin": 0.15,
                         "learning_rate": 0.005, "batch_size": 128,
                         "gci2_reading": "triple"},
           epochs=100, locator="Appendix B, Implementation Details", notes=_NOTES)
    for benchmark in ("joie-dbpedia-instance-view-completion",
                      "joie-dbpedia-ontology-view-completion", "joie-dbpedia-concept-linking"))
