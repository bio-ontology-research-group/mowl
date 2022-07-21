Embedding the :math:`\mathcal{EL}` language
============================================
.. |EL| replace:: :math:`\mathcal{EL}`

The :math:`\mathcal{EL}` language is part of the Description Logics family. Concept descriptions in :math:`\mathcal{EL}` can be expressed in the following normal forms:

.. math::
   \begin{align}
   C &\sqsubseteq D & (\text{GCI 0}) \\
   C_1 \sqcap C_2 &\sqsubseteq D & (\text{GCI 1}) \\
   C &\sqsubseteq \exists R. D & (\text{GCI 2})\\
   \exists R. C &\sqsubseteq D & (\text{GCI 3}) 
   \end{align}

   
.. note::

   GCI stands for General Concept Inclusion

The bottom concept can exist in the right side of GCIs 0,1,3 only, which can be considered as special cases and extend the normal forms to include the following:

.. math::
   \begin{align}
   C &\sqsubseteq \bot & (\text{GCI BOT 0}) \\
   C_1 \sqcap C_2 &\sqsubseteq \bot & (\text{GCI BOT 1}) \\
   \exists R. C &\sqsubseteq \bot & (\text{GCI BOT 3}) 
   \end{align}


mOWL provides different functionalities to generate models that aim to embed axioms in :math:`\mathcal{EL}`. Let's start!

The ELDataset class
------------------------
The :class:`mowl.datasets.ELDataset` class is the first thing you should know about. mOWL first entry point are ontologies. However, not all of them are in the normalized in the |EL| language.
