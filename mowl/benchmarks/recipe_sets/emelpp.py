"""Recipes of EmEL++ and its ELEmbeddings baseline on the EmEL++ benchmarks (Mondal,
Bhatia, Mutharaju, AAAI-MAKE 2021), from the paper and the reference code
(kracr/EmELpp, commit 267f2c7)."""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import MONDAL2021

# ontology -> (embed_dim, margin, steps per epoch of the code)
# Dimension and margin: Table 2 of the paper, and the values Evaluating_HITS.py loads for
# both models. Steps per epoch: ceil(N / 256) with N the largest input table of the
# Keras generator (GALEN: 2 x 28,118 negatives; GO: 76,933 nf1 axioms; ANATOMY:
# 2 x 152,289 negatives), counted with the code's parser on *_norm_mod.owl.
_SETTINGS = {"galen": (50, 0.0, 220), "go": (100, -0.1, 301), "anatomy": (200, -0.1, 1190)}
_EPOCHS = 1000
_LOC = ("experiments/*/models/*_EmEL.py and *_EL.py (batch_size 256, learning_rate 3e-4, "
        "epochs 1000) and Evaluating_HITS.py of kracr/EmELpp; Table 2 of the paper")
_NOTE = ("One mOWL epoch is one optimisation step, so the recipe's epochs are the code's "
         "1,000 epochs times its steps per epoch, and the model is selected after every "
         "code epoch. The margin is passed with the code's sign: a negative margin makes "
         "inclusions strict. The code trains on the validation pairs (see the regime "
         "'validation-axioms-as-positives').")


def _build():
    out = []
    for onto, (dim, margin, steps) in _SETTINGS.items():
        for method in ("emelpp", "elem"):
            model_kwargs = {"embed_dim": dim, "margin": margin, "learning_rate": 3e-4}
            if method == "emelpp":
                model_kwargs["load_role_axioms"] = True
            out.append(Recipe(
                benchmark=f"emelpp-{onto}-prediction", method=method, source=MONDAL2021,
                model_kwargs=model_kwargs,
                training_regime="emelpp-code-training",
                regime_kwargs={"batch_size": 256, "validate_every": steps},
                epochs=_EPOCHS * steps, validate_every=steps, locator=_LOC, notes=_NOTE))
    return tuple(out)


RECIPES = _build()
