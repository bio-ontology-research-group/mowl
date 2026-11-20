"""Recipes of Box2EL and its baselines (Jackermeier et al. 2024) and of TransBox (Yang
et al. 2025), from the papers and their reference code."""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import JACKERMEIER2024, YANG2025

_B2E_LOC = "configs.json of KRR-Oxford/BoxSquaredEL (the values of the paper's Table 10)"
_B2E_NOTE = ("The margin is a slack margin (see the regime 'slack-inclusion-margin'). "
             "The paper states a maximum of 10,000 epochs (Section 4.2).")

# ontology -> task -> (margin, lr, delta, num_neg, reg_factor, lr milestone, epochs)
_BOX2EL = {
    "galen": {"prediction": (0.15, 0.01, 5, 1, 0.4, None, 10000),
              "inference": (0.05, 0.005, 1, 2, 0, 2000, 10000)},
    "go": {"prediction": (0.15, 0.01, 5.5, 5, 0.5, None, 10000),
           "inference": (0.05, 0.005, 3, 3, 0.05, None, 10000)},
    "anatomy": {"prediction": (0.05, 0.01, 5.5, 3, 0.3, None, 7000),
                "inference": (0.05, 0.001, 2, 2, 0.05, None, 7000)},
}

_BASELINE_NOTE = {
    "elem": (" The Box2EL code trains the baselines with negatives drawn once before "
             "training and initialises centre and radius jointly (radii start near 0.06, "
             "mOWL's default is 1); without these the C ⊓ D ⊑ E and ∃r.C ⊑ D numbers "
             "stay far below the paper and the C ⊑ D numbers on GO far above it."),
    "boxel": (" The published numbers for ∃r.C ⊑ D, and the combined numbers that "
              "include them, come from a bug of the reference evaluation (see the "
              "regime 'overlap-volume-scoring') and cannot be reproduced."),
}
_BASELINE_NOTE["emelpp"] = _BASELINE_NOTE["elem"]
_BASELINE_NOTE["elbe"] = (" The Box2EL code trains ELBE with negatives drawn once before "
                          "training and a joint initialisation of centre and offset.")

# method -> ontology -> task -> (embed_dim, margin or None, lr)
_BASELINES = {
    "elem": {"galen": {"prediction": (200, 0.05, 0.0005), "inference": (200, 0, 0.0005)},
             "go": {"prediction": (200, 0.1, 0.0005), "inference": (200, 0.1, 0.001)},
             "anatomy": {"prediction": (200, 0.05, 0.0005),
                         "inference": (200, 0.05, 0.0005)}},
    "elbe": {"galen": {"prediction": (200, 0.05, 0.005), "inference": (200, 0.15, 0.01)},
             "go": {"prediction": (200, 0.1, 0.005), "inference": (200, 0.05, 0.005)},
             "anatomy": {"prediction": (200, 0.05, 0.005),
                         "inference": (200, 0.05, 0.005)}},
    "boxel": {"galen": {"prediction": (50, None, 0.001), "inference": (25, None, 0.005)},
              "go": {"prediction": (25, None, 0.01), "inference": (25, None, 0.001)},
              "anatomy": {"prediction": (25, None, 0.001),
                          "inference": (25, None, 0.001)}},
}
_BASELINES["emelpp"] = _BASELINES["elem"]

# ontology -> learning rate of TransBox (configs_our.json of HuiYang1997/TransBox, section
# "transbox": the section that run_many_seeds.py runs; margin 0, reg_factor 0, num_neg 1,
# neg_dist 1, use_bias 0, num_components 200 = one mask component per dimension, d = 200)
_TRANSBOX = {"galen": 0.0005, "go": 0.0005, "anatomy": 0.005}

_TRANSBOX_NOTE = (
    "The paper averages 10 runs of 5,000 epochs (Section 6.2). The model and its settings "
    "are those of run_many_seeds.py (model 'transbox' = model/TransBox_reg.py, config "
    "section 'transbox'); the section 'transrolebox' belongs to a model class that the "
    "published code does not contain. formulation='reference' gives the code's losses "
    "and initialisation; the regime squares them per axiom. Role inclusions and chains "
    "are trained, as in the code. Checked on ORIX with 3 seeds (2026-11-20): all "
    "normal forms and the combined ranks of GALEN, GO and Anatomy agree with Tables 7, "
    "8 and 11-13 within seed noise; the paper averages 10 seeds.")


def _build():
    out = []
    for onto, tasks in _BOX2EL.items():
        for task, (margin, lr, delta, num_neg, reg, milestone, epochs) in tasks.items():
            kwargs = {"num_negatives": num_neg, "margin": margin}
            if milestone:
                kwargs["lr_milestones"] = [milestone]
            out.append(Recipe(
                benchmark=f"box2el-{onto}-{task}", method="box2el", source=JACKERMEIER2024,
                model_kwargs={"embed_dim": 200, "learning_rate": lr, "delta": delta,
                              "reg_factor": reg},
                training_regime="box2el-code-training", regime_kwargs=kwargs,
                epochs=epochs, validate_every=100, locator=_B2E_LOC, notes=_B2E_NOTE))
    for method, ontos in _BASELINES.items():
        for onto, tasks in ontos.items():
            for task, (dim, margin, lr) in tasks.items():
                # The baselines of the code: negatives drawn once, joint initialisation,
                # the loss details of model/*.py (see Box2ELCodeTraining).
                kwargs = {"num_negatives": 1, "baseline_code": True}
                if margin is not None:
                    kwargs["margin"] = margin
                if method == "elbe":
                    kwargs["neg_target"] = 2.0
                model_kwargs = {"embed_dim": dim, "learning_rate": lr}
                if method == "emelpp":
                    # model/EmELpp.py trains on the role inclusions and chains.
                    model_kwargs["load_role_axioms"] = True
                out.append(Recipe(
                    benchmark=f"box2el-{onto}-{task}", method=method,
                    source=JACKERMEIER2024,
                    model_kwargs=model_kwargs,
                    training_regime="box2el-code-training", regime_kwargs=kwargs,
                    epochs=10000, validate_every=100,
                    locator="Table 10 of the paper; model/*.py of KRR-Oxford/BoxSquaredEL",
                    notes=_B2E_NOTE + _BASELINE_NOTE.get(method, "")))
    for onto, lr in _TRANSBOX.items():
        out.append(Recipe(
            benchmark=f"transbox-{onto}-prediction", method="transbox", source=YANG2025,
            model_kwargs={"embed_dim": 200, "learning_rate": lr, "reg_factor": 0.0,
                          "formulation": "reference", "load_role_axioms": True},
            training_regime="transbox-code-training",
            regime_kwargs={"num_negatives": 1, "margin": 0.0, "neg_dist": 1.0},
            epochs=5000, validate_every=100,
            locator="configs_our.json (section 'transbox'), run_many_seeds.py, "
                    "baseline_sweep.py and model/TransBox_reg.py of HuiYang1997/TransBox",
            notes=_TRANSBOX_NOTE))
    return tuple(out)


RECIPES = _build()
