"""Record the golden values of the behaviour probes.

Run this ONLY with the library version whose behaviour is the reference, i.e. before an
upgrade of the Java libraries::

    python -m tests.owlapi.compat.generate_golden            # refuses to overwrite
    python -m tests.owlapi.compat.generate_golden --force el.projections

Regenerating a golden file after an upgrade to make a failing test pass defeats the
purpose. If the new behaviour is the correct one, say so in the commit that changes the
file and list the change in ``README.md``.
"""

import json
import os
import sys

import mowl
mowl.init_jvm("4g")

from tests.owlapi.compat import probes  # noqa: E402


def owlapi_version():
    from org.semanticweb.owlapi.util import VersionInfo
    return str(VersionInfo.getVersionInfo().getVersion())


def main(argv):
    force = "--force" in argv
    only = [a for a in argv if not a.startswith("--")]
    os.makedirs(probes.GOLDEN, exist_ok=True)
    version = owlapi_version()
    failures = 0
    for ontology, probe in probes.cases():
        if only and f"{ontology}.{probe}" not in only:
            continue
        target = probes.golden_path(ontology, probe)
        if os.path.exists(target) and not force:
            print("exists, kept:", os.path.basename(target))
            continue
        try:
            value = probes.PROBES[probe](probes.ONTOLOGIES[ontology])
        except Exception as error:                              # noqa: BLE001
            failures += 1
            print(f"FAILED {ontology}.{probe}: {type(error).__name__}: {error}")
            continue
        with open(target, "w") as f:
            json.dump({"owlapi": version, "value": value}, f, indent=1, sort_keys=True,
                      ensure_ascii=False)
            f.write("\n")
        print("written:", os.path.basename(target))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
