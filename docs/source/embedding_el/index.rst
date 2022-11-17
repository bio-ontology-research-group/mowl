Embedding the EL language
============================================
.. |eldataset| replace:: :class:`ELDataset <mowl.datasets.el.ELDataset>`
.. |elmodule| replace:: :class:`ELModule <mowl.nn.elmodule.ELModule>`
.. |el| replace:: :math:`\mathcal{EL}`
.. |tutorial_elembeddings| replace:: :doc:`../examples/elmodels/plot_1_elembeddings`
.. |tutorial_elboxembeddings| replace:: :doc:`../examples/elmodels/plot_2_elboxembeddings`

The :math:`\mathcal{EL}` language is part of the Description Logics family. Concept descriptions in :math:`\mathcal{EL}` can be expressed in the following normal forms:

.. math::
   \begin{align}
   C &\sqsubseteq D & (\text{GCI 0}) \\
   C_1 \sqcap C_2 &\sqsubseteq D & (\text{GCI 1}) \\
   C &\sqsubseteq \exists R. D & (\text{GCI 2})\\
   \exists R. C &\sqsubseteq D & (\text{GCI 3}) 
   \end{align}

   
.. hint::

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
The :class:`ELDataset <mowl.datasets.el.ELDataset>` class is the first thing you should know about. mOWL first entry point are ontologies. However, not all of them are normalized in the |el| language. For that reason, we have to normalize the ontology. To do so, we rely on the `Jcel <https://julianmendez.github.io/jcel/>`_ library.

To create a |el| dataset use the following code:

.. testcode:: [eldataset]
	      
   from mowl.datasets.builtin import FamilyDataset
   from mowl.datasets.el import ELDataset

   ontology = FamilyDataset().ontology
   el_dataset = ELDataset(
		ontology,
		class_index_dict = None,
		object_property_index_dict = None,
		extended = True)

As mentioned in the :class:`ELDataset <mowl.datasets.el.ELDataset>` API docs, the variable ``class_index_dict`` is a dictionary where keys are classes names and values are integer indices. The reason for this is that ``ELDataset`` is a collection of integer datasets and the ``class_index_dict`` dictionary keeps the mapping to the datasets. The same situation is true for ``object_property_index_dict``, but it applies for ontology object properties.
The class dictionary can be predefined and input to the dataset. Otherwise it will be created from the input ``ontology``.

The most important method of |eldataset| is:
   
.. testcode:: [eldataset]

   gci_datasets = el_dataset.get_gci_datasets()

That will return a collection of :class:`torch.utils.data.Dataset` objects. If ``extended = False``, then:

.. code-block:: bash

   >> gci_datasets
   {
   'gci0': <mowl.datasets.el.el_dataset.GCI0Dataset at 0x7f977c9d4250>,
   'gci1': <mowl.datasets.el.el_dataset.GCI1Dataset at 0x7f977c9d4220>,
   'gci2': <mowl.datasets.el.el_dataset.GCI2Dataset at 0x7f977c9d42e0>,
   'gci3': <mowl.datasets.el.el_dataset.GCI3Dataset at 0x7f977c9d4340>
   }

which means that only 4 normal forms were obtained after the normalization process. On the other hand, if ``extended = True``, then:

.. code-block:: bash

   >> gci_datasets
   {
   'gci0': <mowl.datasets.el.el_dataset.GCI0Dataset at 0x7f67f3f4ff10>,
   'gci1': <mowl.datasets.el.el_dataset.GCI1Dataset at 0x7f67f351c040>,
   'gci2': <mowl.datasets.el.el_dataset.GCI2Dataset at 0x7f67f351c160>,
   'gci3': <mowl.datasets.el.el_dataset.GCI3Dataset at 0x7f67f3f4feb0>,
   'gci0_bot': <mowl.datasets.el.el_dataset.GCI0Dataset at 0x7f67f3f4ff40>,
   'gci1_bot': <mowl.datasets.el.el_dataset.GCI1Dataset at 0x7f67f351c130>,
   'gci3_bot': <mowl.datasets.el.el_dataset.GCI3Dataset at 0x7f67fc3b99d0>
   }

in this case, normal forms 0, 1 and 3 have been split to consider apart the special cases where the :math:`\bot` concept appears in the right side of each GCI.

The datasets generated can be used directly or through a :class:`torch.utils.data.DataLoader` object. For example:

.. testcode:: [eldataset]

   from torch.utils.data import DataLoader
   dataloader_gci0 = DataLoader(gci_datasets["gci0"])

The ELModule class
----------------------
Previously, we introduced the data-related aspect of this tutorial. Now, let's see how to use the data to train a model.

In the :doc:`/api/nn/index` module, we define the :class:`ELModule <mowl.nn.elmodule.ELModule>` abstract class, which is a subclass of :class:`torch.nn.Module`. To use this class, it is required to define loss functions for the GCIs of interest. For example:

.. testcode:: [eldataset]

   from mowl.nn.elmodule import ELModule

   class MyELModule(ELModule):
       def __init__(self):
           super().__init__()

       def gci0_loss(self, gci, neg = False):
           """
	   your code here
	   """
	   if neg:
	       """
	       your code in case this loss function has a negative version
	       """
	       pass
	   loss = 0
	   return loss

       def gci1_loss(self, gci, neg = False):
	   loss = 1
	   return loss

       def gci2_loss(self, gci, neg = False):
           loss = 2
	   return loss

       def gci3_loss(self, gci, neg = False):
           loss = 3
	   return loss

	
We have created an ELModule that computes losses for axioms in the GCI0 normal form. Notice that if negative loss is required, it should be encoded inside the original loss function and accesed through the ``neg`` parameter.

Following these procedure is all what is needed. It is not necessary to define the forward function. However, let's see how this works by looking at the implementation in the parent class:

.. testcode:: [eldataset]
	      
   import torch.nn as nn

   class ELModule(nn.Module):

       def __init__(self):
           super().__init__()

       """
       .
       .
       .
       loss functions definitions here
       .
       .
       .
       """

       def get_loss_function(self, gci_name):
           if gci_name == "gci2_bot":
               raise ValueError("GCI2 does not allow bottom entity in the right side.")
           return {
	       "gci0_bot": self.gci0_bot_loss,
               "gci1_bot": self.gci1_bot_loss,
               "gci3_bot": self.gci3_bot_loss,
               "gci0"    : self.gci0_loss,
               "gci1"    : self.gci1_loss,
               "gci2"    : self.gci2_loss,
               "gci3"    : self.gci3_loss
           }[gci_name]

       def forward(self, gci, gci_name, neg = False):
           loss_fn = self.get_loss_function(gci_name)
        
           loss = loss_fn(gci, neg = neg)
           return loss

We can see that the already implemented forward function takes the data, the GCI name and the ``neg`` parameter. The idea here is that in the training loop we can get the losses for all the GCIs, and their potential negative versions and we can aggregate them appropriately. In the following section we will see an example of how to use use the :class:`ELModule <mowl.nn.elmodule.ELModule>` and how it matches with the |eldataset| class.

The ELEmbeddingModel class
---------------------------------

At this point, it would be possible to just use the |eldataset| and the |elmodule| together in a script to train a model. Something like this:

.. testcode:: [eldataset]

   from torch.utils.data import DataLoader
   from mowl.datasets.el import ELDataset
   from mowl.nn.elmodule import ELModule
   from mowl.datasets.builtin import PPIYeastSlimDataset

   dataset = PPIYeastSlimDataset()
   class_index_dict = {v:k for k,v in enumerate(dataset.classes.as_str)}
   object_property_index_dict = {v:k for k,v in enumerate(dataset.object_properties.as_str)}

   training_datasets = ELDataset(dataset.ontology, class_index_dict = class_index_dict, object_property_index_dict = object_property_index_dict, extended = False) 
   validation_datasets = ELDataset(dataset.validation, class_index_dict = class_index_dict, object_property_index_dict = object_property_index_dict, extended = False) 
   testing_datasets = ELDataset(dataset.testing, class_index_dict = class_index_dict, object_property_index_dict = object_property_index_dict, extended = False) 

   """
   Furthermore if we need DataLoaders (which might not be always the case)
   """

   training_dataloaders = {k: DataLoader(v, batch_size = 64) for k,v in training_datasets.get_gci_datasets().items()}
   #validation_dataloaders = ..
   #testing_dataloaders = ...

   
   model = MyELModule() #Let's reuse the module of the example before.

   for epoch in range(10):
       for gci_name, gci_dataloader in training_dataloaders.items():
           for i, batch in enumerate(gci_dataloader):
		loss = model(batch, gci_name)

		# .
		# .
		# .
		#More logic for training
		# .
		# .
		# .
		continue

In the previous script, there are some lines of code dedicated to preprocessing the data. That functionality is what is encoded in the :class:`ELEmbeddingModel <mowl.base_models.elmodel.ELEmbeddingModel>` such that if we use it, we can bypass all the data preprocessing and start directly in the training, validation and testing loops.

To see actual examples of EL models, let's go to |tutorial_elembeddings| and |tutorial_elboxembeddings|.
