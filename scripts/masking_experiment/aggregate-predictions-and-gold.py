import json
from pathlib import Path

for dataset in Path("workdata/masking-experiment/predictions").glob("*"):
    for file in dataset.glob("*.predictions"):
        part = file.stem
        gold_path = Path(f"workdata/masking-experiment/datasets/{dataset.name}/{part}.tsv.test")
        out_path = Path(f"workdata/masking-experiment/predictions/{dataset.name}/{part}.aggregated")
        with file.open() as predfile, gold_path.open() as goldfile, out_path.open(
            "w"
        ) as outfile:
            for gold_line, pred_line in zip(goldfile, predfile):
                prediction = json.loads(pred_line)["predicted"]
                sentence, gold = gold_line.strip().split("\t")
                json.dump(
                    {"sentence": sentence, "prediction": prediction, "gold": gold},
                    outfile,
                )
                outfile.write("\n")
