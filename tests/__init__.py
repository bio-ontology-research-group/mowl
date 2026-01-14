import mowl

mowl.init_jvm("10g")

# Dataset downloads are handled in conftest.py via pytest_configure
# This file only initializes the JVM
