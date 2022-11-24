Common errors in mOWL
============================

ImportError: Attempt to create Java package 'X' without jvm
-----------------------------------------------------------------

This error arises when mOWL tries to access some Java object or method but the Java Virtual Machine (JVM) has not been started.

Usually, to solve this, it is enough to add these two lines in the beginning of the main script:

.. testcode::

   import mowl
   mowl.init_jvm("5g")


In the above piece of code, we specify the amount of memory given to the JVM. The memory parameter (`2g` in the example) corresponds to the parameter "-Xmx" for the JVM initialization step. For more information about the JVM memory management please follow this `link <https://docs.oracle.com/cd/E13150_01/jrockit_jvm/jrockit/geninfo/diagnos/garbage_collect.html>`_.

.. note::

   The function ``init_jvm`` can only be called once during running time. This means that the JVM cannot be restarted and this is a limitation of JPype as stated in this `section <https://jpype.readthedocs.io/en/latest/api.html#jpype.shutdownJVM>`_ of their documentation.
