"""
FALCON: Reproducing the Family Membership Heatmap
=================================================

`FALCON <https://arxiv.org/abs/2208.07628>`_ embeds an :math:`\\mathcal{ALC}`
ontology by interpreting every class expression as a *fuzzy set* over a
collection of named and sampled anonymous entities. This example reproduces the
Family experiment of the FALCON paper (Figure 1): we build a small family
ontology with one individual per concept, train :class:`FALCONModel
<mowl.models.FALCONModel>`, and render the concept-by-entity membership heatmap
with :class:`FALCONVisualizer <mowl.visualization.FALCONVisualizer>`.

In the resulting heatmap each individual is a member (dark) of its asserted
concept *and* of every super-concept entailed by the TBox, while remaining a
non-member (light) of disjoint concepts — for example ``boy_0`` belongs to
``Boy``, ``Child``, ``Male`` and ``Person``.
"""

# %%
# Imports and JVM initialisation
# ------------------------------

import mowl
mowl.init_jvm("10g")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.models import FALCONModel
from mowl.visualization import FALCONVisualizer
from mowl.owlapi.adapter import OWLAPIAdapter

# %%
# Build the family ontology
# -------------------------
#
# The TBox encodes the class hierarchy (``SubClassOf``), the defining
# intersections (e.g. ``Parent ⊓ Male ⊑ Father``) and disjointness axioms. The
# ABox asserts one individual per concept (``male_0 : Male``, ...), which anchors
# the learned fuzzy model.

adapter = OWLAPIAdapter()
ontology = adapter.create_ontology("http://mowl/family")


def cls(name):
    return adapter.create_class("http://" + name)


concepts = ["Male", "Female", "Parent", "Child", "Person",
            "Father", "Mother", "Boy", "Girl", "Grandma"]
c = {name: cls(name) for name in concepts}
axioms = HashSet()

subsumptions = [("Male", "Person"), ("Female", "Person"), ("Parent", "Person"),
                ("Child", "Person"), ("Father", "Male"), ("Father", "Parent"),
                ("Mother", "Female"), ("Mother", "Parent"), ("Boy", "Male"),
                ("Boy", "Child"), ("Girl", "Female"), ("Girl", "Child"),
                ("Grandma", "Mother")]
for sub, sup in subsumptions:
    axioms.add(adapter.create_subclass_of(c[sub], c[sup]))

intersections = [(("Parent", "Male"), "Father"), (("Parent", "Female"), "Mother"),
                 (("Child", "Male"), "Boy"), (("Child", "Female"), "Girl")]
for (left, right), target in intersections:
    axioms.add(adapter.create_subclass_of(
        adapter.create_object_intersection_of(c[left], c[right]), c[target]))

for a, b in [("Male", "Female"), ("Parent", "Child"),
             ("Boy", "Girl"), ("Father", "Mother")]:
    axioms.add(adapter.create_disjoint_classes(c[a], c[b]))

for name in concepts:
    individual = adapter.create_individual("http://" + name.lower() + "_0")
    axioms.add(adapter.create_class_assertion(c[name], individual))

adapter.owl_manager.addAxioms(ontology, axioms)
dataset = Dataset(ontology, validation=ontology, testing=ontology)

# %%
# Train FALCON
# ------------
#
# We weight the TBox and concept-assertion (ABox-EC) loss terms equally
# (``alpha = beta = 0.5``), as in the original Family experiment.

model = FALCONModel(dataset, embed_dim=50, anon_e=4, alpha=0.5, beta=0.5,
                    learning_rate=0.01, num_negs=8)
model.train(epochs=1000, validate_every=250)

# %%
# Visualize the learned fuzzy model
# ---------------------------------
#
# Dark cells denote a membership degree close to ``1`` and light cells close to
# ``0``. ``anon_i`` columns are randomly sampled anonymous entities.

visualizer = FALCONVisualizer(model)
ax = visualizer.plot(n_anon=2)
plt.show()
