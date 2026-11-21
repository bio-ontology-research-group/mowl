"""Settings of the FaithEL code (``main.ipynb`` of victorlacerdab/FaithEL_method) for the
family knowledge base. The RuleML+RR paper could not be read; the recipe is the code's."""

from mowl.benchmarks.recipes import Recipe
from mowl.catalog.card import Reference

FAITHEL_CODE = Reference(
    "faithel-code",
    "Code of FaithEL, https://github.com/victorlacerdab/FaithEL_method (main.ipynb, "
    "create_pytorch_dataset.py), commit 0f642bd.",
    "https://github.com/victorlacerdab/FaithEL_method/tree/"
    "0f642bdf9eb50cdb90399620806101db516e3b62")

RECIPES = (
    Recipe(
        benchmark="faithel-family-assertions", method="faithel", source=FAITHEL_CODE,
        model_kwargs={"learning_rate": 0.001, "phi": 1.0, "gamma": 1.0, "psi": 0.7,
                      "radius": 0.6, "neg_sampling": True, "centroid_score": True,
                      "batch_size": 512},
        epochs=200, locator="main.ipynb (LR, PHI, GAMMA, PSI, RADIUS, EPOCHS, "
                            "NEG_SAMPLING, CENTROID_SCORE), create_pytorch_dataset.py "
                            "(BATCH_SIZE)",
        notes=("The notebook sets EMB_DIM = 602 for its hard-coded canonical model of the "
               "family ontology; in mOWL the dimension is that of the canonical model "
               "(99 for this knowledge base) and is not a parameter. radius = "
               "SCALE_FACTOR / 2 + 0.1 with the code's SCALE_FACTOR of 1; the radius "
               "default of the model is that value.")),
)
