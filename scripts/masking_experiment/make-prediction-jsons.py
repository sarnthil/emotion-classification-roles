import json
from pathlib import Path

for dataset in Path("workdata/masking-experiment/datasets").glob("*"):
    for file_in in dataset.glob("*.tsv.test"):
        file_part = file_in.stem[:-4]
        file_out = dataset / f"predict_{file_part}.json.test"
        if not file_in.exists():
            continue
        with file_out.open("w") as out:
            with file_in.open() as f:
                for line in f:
                    sentence, _ = line.strip().split("\t")
                    json.dump({"sentence": sentence}, out)
                    out.write("\n")
