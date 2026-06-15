"""
FALCON: Fuzzy ALC Membership Heatmap
====================================

`FALCON <https://arxiv.org/abs/2208.07628>`_ embeds an :math:`\\mathcal{ALC}`
ontology by interpreting every class expression as a *fuzzy set* over a
collection of named and sampled anonymous entities. After training, the degree
to which each concept holds for each entity can be visualized as a heatmap, in
the style of Figure 1 of the FALCON paper.

This example trains :class:`FALCONModel <mowl.models.FALCONModel>` on the Family
ontology and renders the concept-by-entity membership heatmap with
:class:`FALCONVisualizer <mowl.visualization.FALCONVisualizer>`.
"""

# %%
# Imports and JVM initialisation
# ------------------------------

import mowl
mowl.init_jvm("10g")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mowl.datasets.builtin import FamilyDataset
from mowl.models import FALCONModel
from mowl.visualization import FALCONVisualizer

# %%
# Train the model
# ---------------
#
# We train FALCON with equal weighting of the TBox and ABox-EC loss terms
# (``alpha = beta = 0.5``), matching the original Family experiment.

dataset = FamilyDataset()
model = FALCONModel(dataset, embed_dim=50, anon_e=4, alpha=0.5, beta=0.5,
                    learning_rate=0.01)
model.train(epochs=200, validate_every=50)

# %%
# Visualize the learned fuzzy model
# ---------------------------------
#
# Dark cells denote a membership degree close to ``1`` and light cells close to
# ``0``. ``anon_i`` columns are randomly sampled anonymous entities.

visualizer = FALCONVisualizer(model)
ax = visualizer.plot(n_anon=4)
plt.show()
