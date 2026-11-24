"""Recipes of Concept2Box (Huang et al. 2023, Appendix B "Implementation Details") on the
DBpedia benchmarks.

Stated in the paper: Adam, the TransE instance-view score with margin 0.3, cross-view margin
0.15, instance-view vectors of dimension 256 and ontology-view boxes of dimension 512 (read
here as 256 for the centre and 256 for the offset), batch size 128, learning rate 0.005.
Not stated: the number of epochs, the weights lambda_1 and lambda_2 of the loss, the Gumbel
scale, the initialisation and the alpha of the distance. mOWL's defaults are used for those and
the epoch number is mOWL's choice. The paper alternates the three losses with the learning rates eta,
lambda_1 eta and lambda_2 eta (Algorithm 1); mOWL sums them (Equation 11) and replaces the
BERT encoder of the concept and relation texts by learned embeddings (see the method card).

What the recipe reaches, and what was tried (MRR, DBpedia, one seed; the paper: instance view
0.51, ontology view 0.65, concept linking 0.87):

=======================================================  ========  ========  =======
Training                                                 instance  ontology  linking
=======================================================  ========  ========  =======
this recipe (1000 steps on all axioms)                   0.02      0.30      0.72
mini-batches of 128, 60,000 steps                        0.005     0.29      0.01
mini-batches of 4096, 60,000 steps                       0.04      0.31      0.45
pairwise ranking loss, 1000 steps on all axioms          0.02      0.30      0.25
pairwise ranking loss, mini-batches of 128               0.03      0.27      0.17
pairwise ranking loss, mini-batches of 4096              0.07      0.28      0.62
=======================================================  ========  ========  =======

(mini-batches: the regime ``per-normal-form-minibatch``; pairwise ranking loss: the regime
``pairwise-margin-ranking`` on the role assertions with margin 0.3 and on the class assertions
with margin 0.15, the paper's hinge on a pair, where the method's own losses are the two halves
of the hinge taken apart.) None of them reaches the paper on the instance view. The numbers of
the paper depend on what it does not state (the number of passes, the loss weights, the
alternation of the three losses) and on the BERT encoder, which gives similar concepts similar
boxes from the start.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import HUANG2023

_NOTES = ("Adam on mOWL's generic loop with the three losses summed: one step per epoch on "
          "all axioms, so batch_size (the paper's 128) has no effect; no BERT encoder; the "
          "number of epochs is not stated in the paper (1000 steps is mOWL's choice), nor "
          "are the loss weights, the Gumbel scale, the initialisation and alpha (mOWL "
          "defaults: Gumbel scale 0.25 / d and initial scale 1 / d, without which nothing "
          "is learned at d = 256). gci2_reading='triple': the paper's triple (h, r, t) puts "
          "the tail box inside the head box.")

RECIPES = tuple(
    Recipe(benchmark=benchmark, method="concept2box", source=HUANG2023,
           model_kwargs={"embed_dim": 256, "kge_margin": 0.3, "cross_margin": 0.15,
                         "learning_rate": 0.005, "batch_size": 128,
                         "gci2_reading": "triple"},
           epochs=1000, locator="Appendix B, Implementation Details", notes=_NOTES)
    for benchmark in ("joie-dbpedia-instance-view-completion",
                      "joie-dbpedia-ontology-view-completion", "joie-dbpedia-concept-linking"))
