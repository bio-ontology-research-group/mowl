Installation and Setup
===================================

System requirements
-------------------

- Ubuntu >16.04
- Python version >3.8

Install
-------------------

Installation can be done with the following commands:

.. code:: bash
	  
   git clone https://github.com/bio-ontology-research-group/mowl.git
   
   cd mowl

   conda env create -f environment.yml
   conda activate mowl

   cd mowl
   ./build_jars.sh

The last line will generate the necessary `jar` files to bind Python with the code that runs in the JVM

