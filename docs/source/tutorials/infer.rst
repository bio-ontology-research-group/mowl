Infer new axioms from learned embeddings
=========================================

The embeddings generated using any of the methods in mOWL can be used to infer (or predict) new axioms.

The inference functionality of mOWL is based on axiom patterns that can be specified following a syntax based on regular expressions. For example, for predictions of the form :math:`C \sqsubseteq D`, we will use the following syntax:

.. code:: python

   "c?? SubClassOf c??"

The expression above means that we will query axioms that contains a class entity `c??` that is a subclass of another class entity `c??`. Any pattern expected for the entities in the query must be put between the `?` characters using a regular expression. For example, if we want to query all the subclass axioms where the first entity contains the digits `4932` and the second entity can be anything, we will use the expression:

.. code:: python

   "c?.*?4932.*? SubClassOf c?.*?"

The identifier `c??` is used of ontology classes. For ontology properties (or roles), we use `p??` as identifier.



