Welcome to mOWL's documentation!
===================================

**mOWL** is a Python library for Machine Learning with Ontologies. Here you can find several methods to generate embeddings of ontology entities as described in the paper `Semantic similarity and machine learning with ontologies <https://academic.oup.com/bib/article/22/4/bbaa199/5922325>`_.


Getting started
----------------

**mOWL** can be installed from the `source code <https://github.com/bio-ontology-research-group/mowl>`_ or from `Test PyPi <https://pypi.org/project/mowl-borg/>`_ or from `Conda <https://anaconda.org/ferzcam/mowl-borg>`_. For more details on installation check out the how to :doc:`install/index` section of the project.


Import mOWL and start the JVM
------------------------------

In order to use mOWL with all its functionalities, the Java Virtual Machine must be started. We can do that in the following way:

.. code:: python

   import mowl
   mowl.init_jvm("2g")

In the above piece of code, we specify the amount of memory given to the JVM. The memory parameter (`2g` in the example) corresponds to the parameter "-Xmx" for the JVM initialization step. For more information about the JVM memory management please follow this `link <https://docs.oracle.com/cd/E13150_01/jrockit_jvm/jrockit/geninfo/diagnos/garbage_collect.html>`_.

.. note::

   The function `init_jvm` can only be called once during running time. This means that the JVM cannot be restarted and this is a limitation of JPype as stated in this `section <https://jpype.readthedocs.io/en/latest/api.html#jpype.shutdownJVM>`_ of their documentation.


   
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
   embedding_el/index
.. toctree::
   :maxdepth: 3
   :caption: API
   :hidden:
   :glob:

   api/datasets/index
   api/ontology/index
   api/projection/index
   api/walking/index
   api/reasoning/index
   api/text/index
   api/embedding/index
   api/visualization/index



