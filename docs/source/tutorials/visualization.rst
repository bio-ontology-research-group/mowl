Visualizing ontology entities
===============================

After generating embeddings for ontology classes, it might be useful to have a visual representation of those embeddings. In mOWL, you can make a TSNE plot in 2D just with the following code

.. code:: python
   
   from mowl.visualization.base import TSNE
 
   tsne = TSNE(embeddings, labels)
   tsne.generate_points(5000, workers = 4)
   tsne.savefig("path-to-file.jpg")


.. note::

   Notice that the variable ``embeddings`` is a dictionary where the keys are the names of the entities and the values are the embedding vectors. The variable ``labels`` is also a dictionary and the keys must coincide with the keys of the ``embeddings``. For more information you can visit the :doc:`API documentation <../api/visualization/index>`.
