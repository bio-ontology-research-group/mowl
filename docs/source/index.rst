==================================
 Welcome to mOWL's documentation!
==================================

**mOWL** is a Python library for Machine Learning with
Ontologies. Here you can find functionalities to manipulate
ontologies and use them as data for several methods that generate
embeddings of ontology entities.

Background
===========

Several methods to generate vector representations have been developed and identified in: `Semantic
similarity and machine learning with ontologies
<https://academic.oup.com/bib/article/22/4/bbaa199/5922325>`_. In this
work, several tutorials and slides have been created for users that
are interested on ontologies and machine learning methods. The tutorials can be found as Jupyter notebooks at this `link to the notebooks <https://github.com/bio-ontology-research-group/machine-learning-with-ontologies#notebooks>`_. The slides can be found at this `link to the slides <https://github.com/bio-ontology-research-group/machine-learning-with-ontologies#slides>`_.

`mOWL: Python library for machine learning with biomedical ontologies <https://doi.org/10.1093/bioinformatics/btac811>`_ was developed to provide a standardized framework to use the already existing methods and to ease the development of new ones.

Some of those tutorials can be found in mOWL in the :doc:`/examples/index/` section and, eventually, more examples will be available directly in mOWL.


Getting started
===============

**mOWL** can be installed from `source code <https://github.com/bio-ontology-research-group/mowl>`_ or from `PyPi <https://pypi.org/project/mowl-borg/>`_. For more details on installation check out the how to :doc:`install/index` section of the project.


mOWL, JPype and the JVM
=============================

mOWL is a Python library. Furtheremore, it binds the OWLAPI, which is written in Java. For that reason, mOWL uses JPype to enable JVM and access Java code from Python scripts.

.. image:: imgs/architecture.png
	   

In order to use mOWL with all its functionalities, the Java Virtual Machine must be started. We can do that in the following way:

.. code:: python

   import mowl
   mowl.init_jvm("2g")

In the above piece of code, we specify the amount of memory given to the JVM. The memory parameter (`2g` in the example) corresponds to the parameter "-Xmx" for the JVM initialization step. For more information about the JVM memory management please follow this `link <https://docs.oracle.com/cd/E13150_01/jrockit_jvm/jrockit/geninfo/diagnos/garbage_collect.html>`_.

.. note::

   The function ``init_jvm`` can only be called once during running time. This means that the JVM cannot be restarted and this is a limitation of JPype as stated in this `section <https://jpype.readthedocs.io/en/latest/api.html#jpype.shutdownJVM>`_ of their documentation.

Relevant papers:
==========================
- `mOWL: Python library for machine learning with biomedical ontologies <https://doi.org/10.1093/bioinformatics/btac811>`_
- `Ontology Embedding: A Survey of Methods, Applications and Resources <https://arxiv.org/abs/2406.10964>`_
- `Evaluating Different Methods for Semantic Reasoning Over Ontologies <https://ceur-ws.org/Vol-3592/paper9.pdf>`_
- `Prioritizing genomic variants through neuro-symbolic, knowledge-enhanced learning <https://doi.org/10.1093/bioinformatics/btae301>`_
   
Authors
=======

**mOWL** is a project initiated and developed by the `Bio-Ontology Research Group <https://cemse.kaust.edu.sa/borg>`_ from KAUST.
Furthermore, mOWL had other collaboration by being part of:

* `Biohackathon Japan 2024 <http://2024.biohackathon.org/>`_
* `Biohackathon MENA 2023 <https://biohackathon-europe.org/>`_ as project ``#20``.
* `Biohackathon Europe 2022 <https://2022.biohackathon-europe.org/>`_ as project ``#18``.
* `Biohackathon Europe 2021 <https://2021.biohackathon-europe.org/>`_ as project ``#27``.


License
=======

The package is released under the BSD 3-Clause License.

Citation 
==========

If you used mOWL in your work, please consider citing `this article <https://doi.org/10.1093/bioinformatics/btac811>`_.


.. code:: bibtex
	  
   @article{10.1093/bioinformatics/btac811,
	  author = {Zhapa-Camacho, Fernando and Kulmanov, Maxat and Hoehndorf, Robert},
	  title = "{mOWL: Python library for machine learning with biomedical ontologies}",
	  journal = {Bioinformatics},
	  year = {2022},
	  month = {12},
	  issn = {1367-4803},
	  doi = {10.1093/bioinformatics/btac811},
	  url = {https://doi.org/10.1093/bioinformatics/btac811},
	  note = {btac811},
	  eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac811/48438324/btac811.pdf},
	  }






.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/index
   datasets/index
   ontology/index
   owlapi/index
   corpus/index
   graphs/index
   embedding_el/index
   evaluation/index
   visualization/index
   examples/index
   
.. toctree::
   :maxdepth: 1
   :caption: FAQ
   :hidden:
   :glob:

   faq/errors/index
   
.. toctree::
   :maxdepth: 2
   :caption: API reference
   :hidden:
   :glob:

   api/base_models/index
   api/corpus/index
   api/datasets/index
   api/evaluation/index
   api/models/index
   api/nn/index
   api/ontology/index
   api/owlapi/index
   api/projection/index
   api/walking/index
   api/reasoning/index
   api/visualization/index

.. toctree::
   :maxdepth: 2
   :caption: Appendix
   :hidden:
   :glob:

   appendix/references


