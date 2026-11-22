"""Settings of BoxSEL (Zhu et al. 2024) on the three Statistical EL ontologies.

Source: the table "Best hyperparameters used for different ontologies" (appendix): embedding
dimension, epochs, batch size, learning rate and the bound ``beta`` of the regularisers; the
loss is ``L_TBox + L_loc + L_vol`` (Section "Implementation"), trained with Adam.

mOWL's BoxSEL takes one Adam step per epoch on the mean conditional loss over *all*
conditionals of a structure, not on mini-batches of ``batch size`` conditionals: the paper's
epochs (20 for Person and Hybrid, 30 for Country) are passes over the learning set in
batches, so the recipes use the number of steps they amount to (``epochs x |learning set| / batch
size``, with the learning-set sizes of the appendix table of the knowledge base / query
split). The cooling schedule of the softplus
temperature is mentioned but its values are not given (see the notes of each recipe).
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import ZHU2024

_NOTE = ("embed_dim, learning rate, beta and the three loss terms are the paper's. Epochs are "
         "the number of full-batch steps that the paper's epochs of mini-batches of {bs} "
         "conditionals amount to ({epochs} epochs x {n} learning conditionals / {bs}); "
         "the softplus temperature schedule (10 growing to 1000, the values mOWL's BoxSEL tests "
         "use) is NOT the paper's: it describes a cooling schedule from a high temperature "
         "(Appendix, Softplus Volume and Cooling Schedule) but gives no values. Without a "
         "sharp enough softplus the conjunction boxes of the conditionals collapse to empty "
         "boxes and the probabilities become undefined. The volume regulariser L_vol (Eq. 10) "
         "is on, as in the paper; as printed it is a constant of about beta^n plus a push "
         "towards larger volumes, so its printed value (1e8 for beta = 10, n = 8) is not "
         "informative.")

#: name -> (embed_dim, epochs, batch size, learning rate, beta, learning conditionals)
_TABLE = {"person": (32, 20, 512, 0.01, 10.0, 219282),
          "country": (16, 30, 256, 0.05, 10.0, 19749),
          "hybrid": (8, 20, 256, 0.01, 1.0, 5568)}

RECIPES = tuple(
    Recipe(
        benchmark=f"boxsel-sel-{name}", method="boxsel", source=ZHU2024,
        model_kwargs={"embed_dim": dim, "learning_rate": lr, "beta": beta,
                      "location_weight": 1.0, "volume_weight": 1.0,
                      "temperature": 10.0, "temperature_final": 1000.0,
                      "positive_scale": False},
        epochs=int(round(epochs * n / bs)),
        locator="Table 'Best hyperparameters' (ED, EP, BS, LR, BR); Section 'Implementation'",
        notes=_NOTE.format(bs=bs, epochs=epochs, n=n))
    for name, (dim, epochs, bs, lr, beta, n) in _TABLE.items())
