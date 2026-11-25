import jpype
import jpype.imports
import os
import platform

from mowl import _owlapi_compat  # noqa: F401  (registers the OWLAPI 5 method customizers)


def init_jvm(memory, extra_classpath=None):
    """Start the JVM with the JAR files that mOWL packages on the class path.

    :param memory: Maximum heap size, e.g. ``"4g"``.
    :param extra_classpath: Further JAR files (a path or a list of paths) to put on the
        class path, for example an additional reasoner (see :mod:`mowl.reasoning`). The
        environment variable ``MOWL_EXTRA_CLASSPATH`` (paths separated by the system path
        separator) is added as well. The class path can only be set before the JVM starts.
    """
    dirname = os.path.dirname(__file__)
    extra = []
    if extra_classpath:
        extra += [extra_classpath] if isinstance(extra_classpath, str) else list(
            extra_classpath)
    extra += [p for p in os.environ.get("MOWL_EXTRA_CLASSPATH", "").split(os.pathsep) if p]

    jars_dir = os.path.join(dirname, "lib/")
    if not os.path.exists(jars_dir):
        raise FileNotFoundError(f"JAR files not found. Make sure that the lib directory exists \
and contains the JAR dependencies.")

    if (platform.system() == 'Windows'):
        jars = f'{str.join(";", [jars_dir + name for name in os.listdir(jars_dir)])}'
    else:
        jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

    if extra:
        jars = os.pathsep.join([jars] + extra)

    if not jpype.isJVMStarted():

        jpype.startJVM(
            jpype.getDefaultJVMPath(), "-ea",
            f"-Xmx{memory}",
            "-Djava.class.path=" + jars,
            # ELK logs every reasoning stage at INFO through slf4j-simple
            "-Dorg.slf4j.simpleLogger.log.org.semanticweb.elk=error",
            convertStrings=False)
