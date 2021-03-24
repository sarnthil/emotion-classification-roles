from collections import defaultdict
from pathlib import Path
import jinja2

with open("scripts/table-allennlp.jinja2") as f:
    template = jinja2.Template(f.read())

results = defaultdict(dict)

def get_value(line):
    stuff, _, relevant = line.rpartition(":")
    return float(relevant.strip())

def get_prf(path):
    with path.open() as f:
        for line in f:
            if "Finished evaluating." not in line:
                continue
            next(f)  # Metrics:
            next(f)  # Accuracy:
            # p = get_value(next(f))
            # r = get_value(next(f))
            p = 0
            r = 0
            f = get_value(next(f))
            return f"{p:.2f} & {r:.2f} & {f:.2f}"

for dataset in Path("workdata/results/").glob("*"):
    dataset_name = dataset.name
    for setting in dataset.glob("*"):
        prf = get_prf(setting)
        setting = setting.name.partition("text_emotion_")[2].rpartition(".")[0]
        results[dataset_name][setting] = prf

print(template.render(r=results))
