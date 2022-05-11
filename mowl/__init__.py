import jpype
import jpype.imports
import os


def init_jvm(memory):

    dirname = os.path.dirname(__file__)
    jars_dir = os.path.join(dirname, "lib/")
    jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'
    
    if not jpype.isJVMStarted():
        
        jpype.startJVM(
            jpype.getDefaultJVMPath(), "-ea",
            f"-Xmx{memory}",
            "-Djava.class.path=" + jars,
            convertStrings=False)

