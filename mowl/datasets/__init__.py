import jpype
import jpype.imports
import os

dirname = os.path.dirname(__file__)
#jars_dir = os.path.join(dirname, "../../gateway/build/distributions/gateway/lib/")
jars_dir = os.path.join(dirname, "../lib/")
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Xmx10g",
        "-Djava.class.path=" + jars,
        convertStrings=False)


from .ppi_yeast import PPIYeastDataset
from .ppi_yeast import PPIYeastSlimDataset
