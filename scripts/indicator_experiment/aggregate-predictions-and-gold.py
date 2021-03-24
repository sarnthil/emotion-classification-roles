import json
from pathlib import Path

for dataset in Path("workdata/indicator-experiment/predictions").glob("*"):
    for file in dataset.glob("*.predictions"):
        part = file.stem
        gold_path = Path(f"workdata/position-experiment/datasets/{dataset.name}/{part}.jsonl.test")
        out_path = Path(f"workdata/indicator-experiment/predictions/{dataset.name}/{part}.aggregated")
        with file.open() as predfile, gold_path.open() as goldfile, out_path.open(
            "w"
        ) as outfile:
            for gold_line, pred_line in zip(goldfile, predfile):
                gold = json.loads(gold_line)["gold-label"]
                prediction = json.loads(pred_line)["predicted"]
                tokens = json.loads(pred_line)["tokens"]
                # sentence, gold = gold_line.strip().split("\t")
                json.dump(
                    {"tokens": tokens, "prediction": prediction, "gold": gold},
                    outfile,
                )
                outfile.write("\n")
