import jpype
import jpype.imports
import os


def init_jvm(memory):
    dirname = os.path.dirname(__file__)
    with open("/tmp/pathmowl.txt", "w") as f:
        f.write(dirname)
    jars_dir = os.path.join(dirname, "lib/")
    if not os.path.exists(jars_dir):
        raise Exception(f"Directory {jars_dir} does not exist")
    jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

    if not jpype.isJVMStarted():

        jpype.startJVM(
            jpype.getDefaultJVMPath(), "-ea",
            f"-Xmx{memory}",
            "-Djava.class.path=" + jars,
            convertStrings=False)


