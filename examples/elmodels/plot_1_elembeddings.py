"""
EL Embeddings
===============

This example corresponds to the paper `EL Embeddings: Geometric Construction of Models for the \
Description Logic EL++ <https://www.ijcai.org/proceedings/2019/845>`_.

The idea of this paper is to embed EL by modeling ontology classes as \
:math:`n`-dimensional balls (:math:`n`-balls) and ontology object properties as \
transformations of those :math:`n`-balls. For each of the normal forms, there is a distance \
function defined that will work as loss functions in the optimization framework.
"""

# %%
# Let's just define the imports that will be needed along the example:

import mowl
mowl.init_jvm("10g")
import torch as th



# %%
# The EL-Embeddings model, maps ontology classes, object properties and operators into a
# geometric model. The :math:`\mathcal{EL}` description logic is expressed using the
# following General Concept Inclusions (GCIs):
#
# .. math::
#    \begin{align}
#    C &\sqsubseteq D & (\text{GCI 0}) \\
#    C_1 \sqcap C_2 &\sqsubseteq D & (\text{GCI 1}) \\
#    C &\sqsubseteq \exists R. D & (\text{GCI 2})\\
#    \exists R. C &\sqsubseteq D & (\text{GCI 3})\\
#    C &\sqsubseteq \bot & (\text{GCI BOT 0}) \\
#    C_1 \sqcap C_2 &\sqsubseteq \bot & (\text{GCI BOT 1}) \\
#    \exists R. C &\sqsubseteq \bot & (\text{GCI BOT 3})
#    \end{align}
#
# where :math:`C,C_1, C_2,D` are ontology classes and :math:`R` is an ontology object property

# %%
#
# EL-Embeddings (PyTorch) module.
# -------------------------------
#
# EL-Embeddings defines a geometric modelling for all the GCIs in the EL language.
# The implementation of ELEmbeddings module can be found at :class:`mowl.nn.el.elem.module.ELEmModule`.
#
# EL-Embeddings model
# -------------------
# 
# The module :class:`mowl.nn.el.elem.module.ELEmModule` is used in the :class:`mowl.models.elembeddings.model.ELEmbeddings`.
# In the use case of this example, we will test over a biological problem, which is
# protein-protein interactions. Given two proteins :math:`p_1,p_2`, the phenomenon
# ":math:`p_1` interacts with :math:`p_2`" is encoded using GCI 2 as:
#
# .. math::
#    p_1 \sqsubseteq \exists interacts\_with. p_2
#
# For that, we can use the class :class:`mowl.models.elembeddings.examples.model_ppi.ELEmPPI` mode, which uses the :class:`mowl.datasets.builtin.PPIYeastSlimDataset` dataset.

# %%
# Training the model
# -------------------


from mowl.datasets.builtin import PPIYeastSlimDataset
from mowl.models.elembeddings.examples.model_ppi import ELEmPPI

dataset = PPIYeastSlimDataset()

model = ELEmPPI(dataset,
                embed_dim=30,
                margin=0.1,
                reg_norm=1,
                learning_rate=0.001,
                epochs=20,
                batch_size=4096,
                model_filepath=None,
                device='cpu')

model.train()


# %%
# Evaluating the model
# ----------------------
#
# Now, it is time to evaluate embeddings. For this, we use the
# :class:`ModelRankBasedEvaluator <mowl.evaluation.ModelRankBasedEvaluator>` class.


from mowl.evaluation.rank_based import ModelRankBasedEvaluator

with th.no_grad():                                                                        
    model.load_best_model()                                                               
    evaluator = ModelRankBasedEvaluator(                                                  
        model,                                                                            
        device = "cpu",
        eval_method = model.eval_method
    )                                                                                         
                                                                                                  
    evaluator.evaluate(show=True)
