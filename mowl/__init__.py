import jpype
import jpype.imports
import os
import platform


def init_jvm(memory):
    dirname = os.path.dirname(__file__)

    jars_dir = os.path.join(dirname, "lib/")
    if not os.path.exists(jars_dir):
        raise FileNotFoundError(f"JAR files not found. Make sure that the lib directory exists \
and contains the JAR dependencies.")

    if (platform.system() == 'Windows'):
        jars = f'{str.join(";", [jars_dir + name for name in os.listdir(jars_dir)])}'
    else:
        jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

    if not jpype.isJVMStarted():

        jpype.startJVM(
            jpype.getDefaultJVMPath(), "-ea",
            f"-Xmx{memory}",
            "-Djava.class.path=" + jars,
            convertStrings=False)
