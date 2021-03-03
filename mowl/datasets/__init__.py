import jpype
import jpype.imports
import os

jars_dir = "gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jars,
        convertStrings=False)


from .ppi_yeast import PPIYeastDataset

