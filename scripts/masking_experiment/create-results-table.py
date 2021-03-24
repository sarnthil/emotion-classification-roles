import json
from collections import defaultdict
from pathlib import Path
import jinja2

with open("scripts/masking_experiment/table.jinja2") as f:
    template = jinja2.Template(f.read())

with open("workdata/masking-experiment/results.json") as f:
    data = json.load(f)

results = defaultdict(dict)

def get_prf(numbers):
    return f"{numbers['precision']*100:.0f} & {numbers['recall']*100:.0f} & {numbers['f1']*100:.0f}"

for dataset in data:
    dataset_name = dataset
    for setting in data[dataset]:
        numbers = data[dataset][setting]["all_macro"]
        prf = get_prf(numbers)
        results[dataset_name][setting] = prf

print(template.render(r=results))
