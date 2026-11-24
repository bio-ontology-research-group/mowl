"""Recipes of EIKE (Wang et al. 2024) on YAGO39K and M-YAGO39K.

The settings are the best configurations of the paper's Appendix (``Appendix.pdf`` of
https://github.com/keyu-wang-2002/EIKE, "Best configurations"; the main text gives only the
grids: margins from {0.1, 0.3, 0.4, 0.5, 1, 2}, alpha from {0.1, 0.5, 1, 2}, 1000, 1500 or
2000 epochs, dimension 100, SGD with learning rate 0.001, chosen on the validation set), with
the three margins in the paper's order gamma_l (relational), gamma_e (InstanceOf) and
gamma_c (SubClassOf); the identification of gamma_e and gamma_c is from the code (``margin_ins``,
``margin_sub``). The Appendix gives one configuration per task and dataset and model (UNP/PRE,
EYE/MAT), the same for uniform and Bernoulli negatives. The configurations of the triple
classification apply to the three triple classification benchmarks of a dataset, and those
of the link prediction to ``yago39k-link-prediction``.

What mOWL does differently from the paper's code (https://github.com/keyu-wang-2002/EIKE,
``code/eike-pretrain-eye`` and its siblings), and the recipes cannot change:

* mOWL trains with Adam on the generic loop, one epoch is a pass over every kind of triple;
  the code uses SGD, 100 mini-batches per epoch of 8,220 triples of the three kinds mixed
  (sampled with replacement), re-normalises the entity, relation and intensional vectors and
  the radii after every step, and uses a schedule that changes the share of hard negatives
  within each block of 1000 epochs (``in_rate``, ``hrt_cut``...). ``batch_size`` here is the
  size of the paper's mixed batch.
* The relational score is the squared L2 distance of the paper's formula; the Appendix picks
  the L1 norm for most configurations (all but PRE-EYE in triple classification), which
  mOWL does not have. The Bernoulli corruption ("bern") is not available either; negatives
  are uniform ("unif"), corrupting head or tail.
* PRE: the paper encodes the concept names with Sentence-BERT (the code:
  ``all-MiniLM-L6-v2``, 384 dimensions, frozen) and maps them to 100 dimensions with a
  trained linear layer. mOWL takes vectors of the embedding dimension: give
  ``intensional_vectors`` (see below). The PRE recipes cannot carry the vectors, so the
  run needs ``--param intensional_vectors=<file>``; without it a PRE run trains the UNP model.
* The authors' code lists other margins for link prediction (``margin_ins`` 0.3 in
  ``main.py`` of ``eike-pretrain-eye``) than the Appendix; the Appendix is followed.
* The checkpoint is taken at 1000, 1500 or 2000 epochs by validation accuracy or Hits@10 in
  the paper; the recipes train 2000 epochs and keep the last.

What the authors' code gives, and which of its ingredients matter (UNP-EYE, the triple
classification configuration, YAGO39K, 2000 epochs, seed 0; F1 in percent of InstanceOf /
SubClassOf / relational triples at the last epoch):

=====================================================  ==========  ==========  ==========
run of the authors' code                               InstanceOf  SubClassOf  relational
=====================================================  ==========  ==========  ==========
as published, Bernoulli negatives                      86.7        80.8        93.5
as published, uniform negatives                        87.1        80.9        92.9
no hard (sibling) negatives                            84.4        81.9        92.7
squared L2 instead of L1 in the relational score       87.9        79.8        92.0
no normalisation after the steps                       86.5        80.7        92.2
no alternating phases after epoch 1000                 86.5        81.0        92.5
Adam instead of SGD                                    82.1        34.0        90.7
all five of the above together                         81.2        78.6        90.7
=====================================================  ==========  ==========  ==========

No single ingredient of the training loop explains the distance to mOWL (68 / 67 / 85 with
the same configuration): with all of them replaced by what mOWL does the authors' code still
reaches 81 / 79 / 91. Adam alone breaks SubClassOf together with the hard negatives and the
normalisation (68 at epoch 1000, 34 at 2000). Two differences outside the loop remain: the
code starts from vectors of norm about 0.1 (normal, standard deviation 1/d) and radii
uniform in (0, 1), mOWL from uniform(-0.5, 0.5) and radius 1; and the code does not decide
SubClassOf by a threshold on the trained score but by TransC's rule on the two radii and the
distance of the centres (``check_sub_class`` of ``classification_test_isA.py``).

Supplying vectors:``EIKE.encode_class_texts(EIKE.class_texts(), encoder)`` with a sentence
encoder of your choice (``SentenceTransformer("all-MiniLM-L6-v2").encode``), then
``EIKE.pca_vectors(vectors, 100)`` and ``EIKE.save_intensional_vectors(path, vectors)``.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import WANG2024EIKE

# name -> (gamma_l, gamma_e, gamma_c, alpha, norm)
_TC = {"pre-eye": (1.0, 0.4, 0.3, 1.0, "L2"), "unp-eye": (1.0, 0.4, 0.1, 0.1, "L1"),
       "pre-mat": (1.0, 1.0, 0.4, 0.5, "L1"), "unp-mat": (1.0, 1.0, 0.3, 0.1, "L1")}
_LP = {"pre-eye": (1.0, 0.1, 0.1, 0.1, "L1"), "unp-eye": (1.0, 0.4, 0.3, 2.0, "L1"),
       "pre-mat": (1.0, 0.1, 0.1, 2.0, "L1"), "unp-mat": (1.0, 0.3, 0.1, 0.5, "L1")}

_TC_BENCHMARKS = ("yago39k-instanceof-classification", "yago39k-subclassof-classification",
                  "yago39k-relational-classification", "m-yago39k-instanceof-classification",
                  "m-yago39k-subclassof-classification")


def _recipe(benchmark, name, config, task):
    gamma_l, gamma_e, gamma_c, alpha, norm = config
    notes = ("Adam on mOWL's generic loop, uniform negatives, no per-step normalisation: not "
             "the paper's SGD loop (see the module docstring).")
    if norm == "L1":
        notes += (" The paper's best configuration uses the L1 norm in the relational score; "
                  "mOWL has the squared L2 distance.")
    if name.startswith("pre"):
        notes += (" Needs --param intensional_vectors=<file> (Sentence-BERT vectors of the "
                  "class names reduced to 100 dimensions); the paper trains a linear map "
                  "from frozen 384-dimensional all-MiniLM-L6-v2 vectors.")
    return Recipe(
        benchmark=benchmark, method="eike", source=WANG2024EIKE, name=name,
        model_kwargs={"embed_dim": 100, "alpha": alpha,
                      "margins": [gamma_c, gamma_e, gamma_l],
                      "instance_map": "eye" if name.endswith("eye") else "matrix",
                      "learning_rate": 0.001, "batch_size": 8220},
        epochs=2000, locator=f"Appendix (Appendix.pdf of the authors' repository), "
                             f"Best configurations, {task}, {name.upper()}", notes=notes)


RECIPES = tuple(
    [_recipe(b, name, config, "Triple Classification")
     for b in _TC_BENCHMARKS for name, config in _TC.items()]
    + [_recipe("yago39k-link-prediction", name, config, "Link Prediction")
       for name, config in _LP.items()])
