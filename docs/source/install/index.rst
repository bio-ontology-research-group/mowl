Installation and Setup
===================================

System requirements
-------------------

- Ubuntu >16.04
- Python version >3.8

Install from Conda
-------------------

We recommend installation through ``conda``.

.. code:: bash

    git clone https://github.com/bio-ontology-research-group/mowl.git

	  
    conda env create -f environment.yml
    conda activate mowl

    mkdir -p ../data

    cd mowl
    ./rebuild.sh
