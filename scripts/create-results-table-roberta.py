import json
from collections import defaultdict
from pathlib import Path
import jinja2

with open("scripts/results-table.jinja2") as f:
    template = jinja2.Template(f.read())

posdata = {}
maskdata = {}


def get_prf(numbers):
    return f"{numbers['precision_macro']*100:.0f} & {numbers['recall_macro']*100:.0f} & {numbers['f1_macro']*100:.0f}"


for path in Path("results").glob("*"):
    # eval-unified_eca_only_cause-1603801217.txt
    # eval-unified_es_all-1603820737.txt
    data = {
        key: float(value)
        for (key, value) in (line.split(" = ") for line in path.open().read().strip().split("\n"))
    }
    _, __, rest = path.name.partition("_")
    dataset, _, rest = rest.partition("_")
    rest, _, __ = rest.rpartition("-")
    if rest == "all":
        mask_type = "all"
        role = None
    else:
        mask_type, role = rest.split("_")

    if mask_type == "inband":  # posdata
        posdata.setdefault(dataset, {})[role] = get_prf(data)
    else:  # maskdata
        setting = "all" if mask_type == "all" else f"{mask_type}_{role}"
        maskdata.setdefault(dataset, {})[setting] = get_prf(data)


# posdata shape:
# dataset -> role -> str for P R F
# maskdata shape:
# dataset -> setting -> str for P R F
# setting: "without/only_" + role

maskresults, posresults = maskdata, posdata
label = {
    "et": "ET",
    "reman": "REMAN",
    "es": "ES",
    "gne": "GNE",
    "eca": "ECA",
}

print(template.render(maskdata=maskresults, posdata=posresults, label=label))
