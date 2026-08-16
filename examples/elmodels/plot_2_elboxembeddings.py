"""
ELBoxEmbeddings
===========================

This example is based on the paper `Description Logic EL++ Embeddings with Intersectional \
Closure <https://arxiv.org/abs/2202.14018v1>`_. This paper is based on the idea of \
:doc:`/examples/elmodels/plot_1_elembeddings`, but in this work the main point is to solve the \
*intersectional closure* problem.

In the case of :doc:`/examples/elmodels/plot_1_elembeddings`, the geometric objects representing \
ontology classes are :math:`n`-dimensional balls. One of the normal forms in EL is:

.. math::
   C_1 \sqcap C_2 \sqsubseteq D

As we can see, there is an intersection operation :math:`C_1 \sqcap C_2`. Computing this \
intersection using balls is not a closed operations because the region contained in the \
intersection of two balls is not a ball. To solve that issue, this paper proposes the idea of \
changing the geometric objects to boxes, for which the intersection operation has the closure \
property.
"""

# %%
# This example is quite similar to the one found in :doc:`/examples/elmodels/plot_1_elembeddings`.
# There might be slight changes in the training part but the most important changes are in the
# definition of loss functions definition of the loss functions for each normal form.


import mowl
mowl.init_jvm("10g")
import torch as th


# %%
#
# ELBoxEmbeddings (PyTorch) module
# ---------------------------------
#
# ELBoxEmbeddings defines a geometric modelling for all the GCIs in the EL language.
# The implementation of the module can be found at :class:`mowl.nn.ELBEModule`

# %%
#
# ELBoxEmbeddings model
# ----------------------
#
# The module :class:`mowl.nn.ELBEModule` is used in the :class:`mowl.models.ELBE`.
# In the use case of this example, we will test over a biological problem, which is
# protein-protein interactions. Given two proteins :math:`p_1,p_2`, the phenomenon
# ":math:`p_1` interacts with :math:`p_2`" is encoded using GCI 2 as:
#
# .. math::
#    p_1 \sqsubseteq \exists interacts\_with. p_2
#
# For that, we can use the class :class:`mowl.models.elbe.examples.model_ppi.ELBEPPI`, which uses the :class:`mowl.datasets.builtin.PPIYeastSlimDataset` dataset.



# %%
# Training the model
# -------------------


from mowl.datasets.builtin import PPIYeastSlimDataset
from mowl.models.elbe.examples.model_ppi import ELBEPPI

dataset = PPIYeastSlimDataset()

model = ELBEPPI(dataset,
                embed_dim=30,
                margin=-0.05,
                reg_norm=1,
                learning_rate=0.001,
                batch_size=20000,
                model_filepath=None,
                device='cuda' if th.cuda.is_available() else 'cpu')

model.train(epochs=20)



# %%
# Evaluating the model
# ----------------------
#
# Now, it is time to evaluate embeddings. For this, we use the
# :class:`PPIEvaluator <mowl.evaluation.PPIEvaluator>` class, which
# :class:`ELBEPPI <mowl.models.elbe.examples.model_ppi.ELBEPPI>` already sets up
# for us.


model.evaluate(dataset.testing)
