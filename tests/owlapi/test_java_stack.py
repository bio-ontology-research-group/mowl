"""The set of Java libraries that mOWL ships is coherent.

These tests state what must hold for ``mowl/lib`` after any change to it; the first one
states which OWL API generation mOWL is on.
"""

import ast
import collections
import os
import re
import subprocess
import sys

import pytest

import mowl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "mowl", "lib")
JAVA_ROOTS = ("org", "uk", "de", "java", "javax", "com", "scala")

#: The OWL API generation mOWL is built on.
OWLAPI_MAJOR = 5


def _owlapi_version():
    from org.semanticweb.owlapi.util import VersionInfo
    return str(VersionInfo.getVersionInfo().getVersion())


def test_owlapi_generation():
    assert int(_owlapi_version().split(".")[0]) == OWLAPI_MAJOR, _owlapi_version()


def test_owlapi_jars_have_one_version():
    versions = {m.group(1) for name in os.listdir(LIB)
                for m in [re.match(r"owlapi-[a-z]+-(\d[\w.]*)\.jar$", name)] if m}
    assert len(versions) == 1, versions
    assert _owlapi_version().startswith(next(iter(versions)).split("-")[0])


def test_gateway_build_declares_the_shipped_owlapi():
    """gateway.jar is compiled against the OWL API version that is shipped."""
    with open(os.path.join(ROOT, "gateway", "build.gradle")) as f:
        declared = set(re.findall(r"net\.sourceforge\.owlapi:owlapi-[a-z]+:([\w.]+)", f.read()))
    assert declared, "gateway/build.gradle declares no OWL API dependency"
    shipped = {m.group(1) for name in os.listdir(LIB)
               for m in [re.match(r"owlapi-[a-z]+-(\d[\w.]*)\.jar$", name)] if m}
    assert declared == shipped, (declared, shipped)


def test_no_artifact_in_two_versions():
    artifacts = collections.defaultdict(set)
    for name in os.listdir(LIB):
        m = re.match(r"(.+?)-(\d[\w.]*?)(-(?:bin|lib|src))?\.jar$", name)
        if m:
            artifacts[m.group(1) + (m.group(3) or "")].add(m.group(2))
    doubled = {a: sorted(v) for a, v in artifacts.items() if len(v) > 1}
    assert not doubled, doubled


@pytest.mark.parametrize("resource", [
    "org/semanticweb/owlapi/model/OWLOntology.class",
    "org/semanticweb/owlapi/apibinding/OWLManager.class",
    "uk/ac/manchester/cs/owl/owlapi/OWLDataFactoryImpl.class",
    "org/semanticweb/elk/owlapi/ElkReasonerFactory.class",
    "org/semanticweb/HermiT/Reasoner.class",
    "org/mowl/Projectors/DL2VecProjector.class",
])
def test_one_copy_of_core_classes_on_the_class_path(resource):
    """A class that is on the class path twice is loaded from whichever JAR comes first
    in the directory listing, which differs between machines."""
    import java.lang
    urls = java.lang.ClassLoader.getSystemClassLoader().getResources(resource)
    found = []
    while urls.hasMoreElements():
        found.append(str(urls.nextElement().toString()))
    assert len(found) == 1, found


def _java_imports():
    found = set()
    for base, _, files in os.walk(os.path.join(ROOT, "mowl")):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            with open(path, encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                        and node.module.split(".")[0] in JAVA_ROOTS:
                    for alias in node.names:
                        found.add((node.module, alias.name))
    return sorted(found)


def test_mowl_has_java_imports():
    assert len(_java_imports()) > 50


@pytest.mark.parametrize("module,name", _java_imports(),
                         ids=[f"{m}.{n}" for m, n in _java_imports()])
def test_every_java_import_of_mowl_resolves(module, name):
    """Every ``from <java package> import <name>`` in mOWL's source resolves on the
    shipped class path, including those inside functions that no test happens to call."""
    import importlib
    if name == "*":
        importlib.import_module(module)
        return
    package = importlib.import_module(module)
    assert hasattr(package, name), f"{module}.{name}"


def test_all_reasoner_backends_are_available():
    from mowl.reasoning import list_reasoners
    unavailable = [b.name for b in list_reasoners() if not b.is_available()]
    assert not unavailable, unavailable


_FRESH = r"""
import sys, warnings
warnings.simplefilter("ignore")
import mowl
mowl.init_jvm("2g")
from tests.owlapi.compat import probes
from mowl.reasoning import create_reasoner
for name in ("el", "rich"):
    for backend in ("structural", "elk", "hermit", "km"):
        if (name, "reasoner_" + backend) in probes.SKIP:
            continue
        reasoner, _ = create_reasoner(probes.load(probes.ONTOLOGIES[name]), backend, warn=False)
        reasoner.getUnsatisfiableClasses()
        reasoner.dispose()
from mowl.ontology.normalize import ELNormalizer
ELNormalizer().normalize(probes.load(probes.ONTOLOGIES["el"]), use_cache=False)
print("FRESH-OK")
"""


def test_fresh_jvm_has_no_linkage_errors():
    """In a new process (nothing loaded by other tests), every reasoner and the
    normaliser run without a Java linkage error, which is how a JAR built for another
    OWL API generation shows up."""
    env = dict(os.environ, PYTHONPATH=ROOT)
    done = subprocess.run([sys.executable, "-c", _FRESH], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=900)
    output = done.stdout + done.stderr
    assert "FRESH-OK" in done.stdout, output[-3000:]
    for marker in ("NoSuchMethodError", "NoClassDefFoundError", "ClassNotFoundException",
                   "IncompatibleClassChangeError", "AbstractMethodError", "LinkageError"):
        assert marker not in output, output[-3000:]
