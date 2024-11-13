Installation and Setup
===================================

mOWL runs on Linux and MAC OS X systems. The library has not been tested on Windows and it could not work properly due to Java compatibility.


System requirements
-------------------

- JDK version 17
- Python version 3.9, 3.10, 3.11, 3.12
- Conda version >= 4.x.x


Python requirements
----------------------

- Gensim >= 4.x.x
- PyTorch >= 1.12.x
- PyKEEN >= 1.10.1


  
Install from source code
--------------------------

Before installing from source, make sure to meet the dependencies.

  
Installation can be done with the following commands:

.. code:: bash
	  
   git clone https://github.com/bio-ontology-research-group/mowl.git
   
   cd mowl

   conda env create -f envs/environment_3_8.yml
   conda activate mowl
   
   ./build_jars.sh

   python setup.py install

Since mOWL needs to bind the Java Virtual Machine, the last line will generate the necessary ``.jar`` files.

Install from PyPi
------------------------------

.. code:: bash
	  
   pip install mowl-borg

