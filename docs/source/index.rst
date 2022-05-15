Welcome to mOWL's documentation!
===================================

**mOWL** is a Python library for Machine Learning with Ontologies. Here you can find several methods to generate embeddings of ontology entities as described in the paper `Semantic similarity and machine learning with ontologies <https://academic.oup.com/bib/article/22/4/bbaa199/5922325>`_.


Getting started
----------------

.. note::

   This project is under development.



**mOWL** can be installed from the `source code <https://github.com/bio-ontology-research-group/mowl>`_ or from `Test PyPi <https://test.pypi.org/project/mowl-borg/>`_.

Source code installation can be done with the following commands:

.. code:: bash
	  
   git clone https://github.com/bio-ontology-research-group/mowl.git
   
   cd mowl

   conda env create -f environment.yml
   conda activate mowl

   cd mowl
   ./build_jars.sh



For more details on installation check out the how to :doc:`install/index` section of the project.

   
Authors
----------

**mOWL** is a project initiated and developed by the `Bio-Ontology Research Group <https://cemse.kaust.edu.sa/borg>`_ from KAUST.
Furthermore, mOWL had other collaboration by being part of the `Biohackathon Europe 2021 <https://2021.biohackathon-europe.org/>`_.

License
---------------

The package is released under the BSD 3-Clause License.


.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :hidden:
   :glob:

   install/index
   tutorials/index

.. toctree::
   :maxdepth: 3
   :caption: API
   :hidden:
   :glob:

   api/datasets/index
   api/graph/index
   api/walking/index
   api/embedding/index




