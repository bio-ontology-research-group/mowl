Installation and Setup
===================================

System requirements
-------------------

- Ubuntu 16.04
- > Python version 3.8

Install from Conda
-------------------

We recommend installation through ``conda``.

.. code:: bash

    conda env create -f environment.yml
    conda activate mowl

    mkdir -p ../data

    cd mowl
    ./rebuild.sh

Install from source
-------------------

.. code:: bash 

    git clone https://github.com/bio-ontology-research-group/mowl.git

Linux
``````````

Windows
``````````

macOS
``````````