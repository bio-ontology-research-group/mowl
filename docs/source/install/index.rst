Installation and Setup
===================================

System requirements
-------------------

- Ubuntu >16.04
- Python version >3.8

Install from source code
--------------------------

Installation can be done with the following commands:

.. code:: bash
	  
   git clone https://github.com/bio-ontology-research-group/mowl.git
   
   cd mowl

   conda env create -f environment.yml
   conda activate mowl

   cd mowl
   ./build_jars.sh

Since mOWL needs to bind the Java Virtual Machine, the last line will generate the necessary `jar` files.



Install from PyPi
------------------------------

PyPi installation is on testing phase and can be done as follows:

.. code:: bash
	  
   pip install -i https://test.pypi.org/simple/ mowl-borg
