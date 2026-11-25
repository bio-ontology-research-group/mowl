"""Run the RELEASED v2.1.0 code (tree at ROOT, from `git archive v2.1.0`) and save, for the EL
modules ELEm, ELBE, BoxEL and Box2EL: their state dict, and every per-axiom loss (positive and,
where the module has one, negative) on fixed rows, with its shape.

usage: python -I gen_v210_fixtures.py ROOT OUT.pt
"""
import os
import sys
import warnings

root = os.path.abspath(sys.argv[1])
sys.path.insert(0, root)
warnings.filterwarnings("ignore")
import torch as th  # noqa: E402

import mowl  # noqa: E402
assert os.path.abspath(mowl.__file__).startswith(root), mowl.__file__
mowl.init_jvm("1g")
import mowl.nn as N  # noqa: E402

C, R, I, D = 6, 2, 3, 4
SPECS = {"ELEmModule": dict(args=(C, R, I)), "ELBEModule": dict(args=(C, R, I)),
         "BoxELModule": dict(args=(C, R, I)), "BoxSquaredELModule": dict(args=(C, R, I))}
# column kinds of every form: c = class, r = role, i = individual
FORMS = {"gci0": "cc", "gci0_bot": "cc", "gci1": "ccc", "gci1_bot": "ccc", "gci2": "crc",
         "gci3": "rcc", "gci3_bot": "rcc", "class_assertion": "ic",
         "object_property_assertion": "iri"}


def rows(kinds, n=5, seed=1):
    g = th.Generator().manual_seed(seed)
    size = {"c": C, "r": R, "i": I}
    return th.stack([th.randint(size[k], (n,), generator=g) for k in kinds], dim=1)


out = {"version": "2.1.0", "sizes": (C, R, I, D)}
for name, spec in SPECS.items():
    th.manual_seed(0)
    module = getattr(N, name)(*spec["args"], embed_dim=D)
    entry = {"state_dict": {k: v.clone() for k, v in module.state_dict().items()},
             "losses": {}, "attrs": sorted(n for n, _ in module.named_children())}
    for form, kinds in FORMS.items():
        data = rows(kinds)
        for neg in (False, True):
            key = form + ("/neg" if neg else "")
            try:
                with th.no_grad():
                    value = module.get_loss_function(form)(data, neg=neg)
            except Exception as e:  # noqa: BLE001
                entry["losses"][key] = {"error": f"{type(e).__name__}"}
                continue
            if isinstance(value, Exception):
                entry["losses"][key] = {"error": f"returned {type(value).__name__}"}
                continue
            entry["losses"][key] = {"rows": data, "value": value.clone(),
                                    "shape": tuple(value.shape)}
    with th.no_grad():
        reg = getattr(module, "regularization_loss", None)
        entry["regularization"] = None if reg is None else th.as_tensor(reg()).clone()
    out[name] = entry
    print(name, {k: v.get("shape", v.get("error")) for k, v in entry["losses"].items()})
th.save(out, sys.argv[2])
