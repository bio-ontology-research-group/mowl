"""Recipes of the baselines that Yang et al. (WWW 2025) re-ran for the TransBox paper
(Tables 7, 8, 11-13): Box2EL, ELBE and BoxEL on the ``transbox-*-prediction``
benchmarks.

* Box2EL: the TransBox code does not train Box2EL (``baseline_sweep.py`` has no Box2EL
  entry); its Box2EL rows agree with the Box2EL paper except where the evaluation of
  C ⊓ D ⊑ E differs, so the settings of the Box2EL code (``configs.json`` of
  KRR-Oxford/BoxSquaredEL) are used.
* ELBE: ``model/Elbe.py`` of the TransBox code, which is the Box2EL code's ELBE with the
  loss of ∃r.C ⊑ D corrected to an inclusion, trained by ``baseline_sweep.py`` (batch 512,
  5,000 epochs) with the ``elbe`` section of ``configs_our.json`` (margin 0.15).
* BoxEL: ``model/BoxEL.py`` (the Box2EL code's BoxEL) with the ``boxel`` section of
  ``configs_our.json``, 5,000 epochs.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import YANG2025
from mowl.benchmarks.recipe_sets.box2el_transbox import _BOX2EL

_CODE = "baseline_sweep.py, configs_our.json and model/*.py of HuiYang1997/TransBox"

# ontology -> learning rate of ELBE (configs_our.json, section "elbe"; margin 0.15, d = 200)
_ELBE_LR = {"galen": 0.005, "go": 0.005, "anatomy": 0.01}
# ontology -> (embedding dimension, learning rate) of BoxEL (section "boxel")
_BOXEL = {"galen": (50, 0.001), "go": (25, 0.01), "anatomy": (25, 0.001)}


def _build():
    out = []
    for onto in ("galen", "go", "anatomy"):
        margin, lr, delta, num_neg, reg, milestone, epochs = _BOX2EL[onto]["prediction"]
        kwargs = {"num_negatives": num_neg, "margin": margin}
        if milestone:
            kwargs["lr_milestones"] = [milestone]
        out.append(Recipe(
            benchmark=f"transbox-{onto}-prediction", method="box2el", source=YANG2025,
            model_kwargs={"embed_dim": 200, "learning_rate": lr, "delta": delta,
                          "reg_factor": reg, "load_role_axioms": True},
            training_regime="box2el-code-training", regime_kwargs=kwargs,
            epochs=epochs, validate_every=100,
            locator="configs.json of KRR-Oxford/BoxSquaredEL (not part of the TransBox "
                    "code)",
            notes=("The TransBox paper states 5,000 epochs for all models (Section 6.2) "
                   "but its Box2EL rows match the Box2EL paper's, so the Box2EL code's "
                   "own settings are used.")))
        out.append(Recipe(
            benchmark=f"transbox-{onto}-prediction", method="elbe", source=YANG2025,
            model_kwargs={"embed_dim": 200, "learning_rate": _ELBE_LR[onto],
                          "gci3_form": "inclusion", "init": "joint"},
            training_regime="box2el-code-training",
            regime_kwargs={"num_negatives": 1, "margin": 0.15, "neg_target": 2.0},
            epochs=5000, validate_every=100, locator=_CODE,
            notes=("The margin is a slack margin. Unlike the Box2EL code, the TransBox "
                   "code trains ∃r.C ⊑ D as an inclusion (gci3_form='inclusion').")))
        dim, lr = _BOXEL[onto]
        out.append(Recipe(
            benchmark=f"transbox-{onto}-prediction", method="boxel", source=YANG2025,
            model_kwargs={"embed_dim": dim, "learning_rate": lr},
            training_regime="box2el-code-training", regime_kwargs={"num_negatives": 1},
            epochs=5000, validate_every=100, locator=_CODE,
            notes=("The published BoxEL numbers for ∃r.C ⊑ D (and the combined numbers "
                   "that include them) come from compute_nf4_ranks_boxel, which ranks "
                   "the class whose index equals the role index; they are not "
                   "reproducible by a correct evaluation.")))
    return tuple(out)


RECIPES = _build()
