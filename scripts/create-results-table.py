import json
from collections import defaultdict
from pathlib import Path
import jinja2

with open("scripts/results-table.jinja2") as f:
    template = jinja2.Template(f.read())

with open("workdata/masking-experiment/results.json") as f:
    maskdata = json.load(f)

with open("workdata/position-experiment/results.json") as f:
    posdata = json.load(f)

maskresults, posresults = defaultdict(lambda: defaultdict(dict)), defaultdict(dict)

def get_prf(numbers):
    return f"{numbers['precision']*100:.0f} & {numbers['recall']*100:.0f} & {numbers['f1']*100:.0f}"

for dataset in maskdata:
    dataset_name = dataset
    for setting in maskdata[dataset]:
        masknumbers = maskdata[dataset][setting]["all_macro"]
        maskprf = get_prf(masknumbers)
        maskresults[dataset_name][setting] = maskprf
        if "_" in setting:
            _adverb, _, role = setting.partition("_")
            posnumbers = posdata[dataset][f"iob-{role}"]["all_macro"]
            posprf = get_prf(posnumbers)
            posresults[dataset_name][role] = posprf

label = {
    "electoral_tweets": "ET",
    "reman": "REMAN",
    "emotion-stimulus": "ES",
    "gne": "GNE",
    "eca": "ECA",
}

print(template.render(maskdata=maskresults, posdata=posresults, label=label))
