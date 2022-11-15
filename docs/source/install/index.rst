Installation and Setup
===================================

mOWL runs on Linux and MAC OS X systems. The library has not been tested on Windows and it could not work properly due to Java compatibility.


System requirements
-------------------

- JDK version 8
- Python version 3.8

  
Install from source code
--------------------------

Before installing from source, make sure to meet the dependencies:

  
Installation can be done with the following commands:

.. code:: bash
	  
   git clone https://github.com/bio-ontology-research-group/mowl.git
   
   cd mowl

   conda env create -f environment.yml
   conda activate mowl

   cd mowl
   ./build_jars.sh
   
   python setup.py install

Since mOWL needs to bind the Java Virtual Machine, the last line will generate the necessary ``.jar`` files.



Install from PyPi
------------------------------

.. code:: bash
	  
   pip install mowl-borg


Install from Conda
------------------------------


.. code:: bash

   conda install -c ferzcam mowl-borg

