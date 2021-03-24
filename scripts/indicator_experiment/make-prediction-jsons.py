import json
from pathlib import Path

for dataset in Path("workdata/position-experiment/datasets").glob("*"):
    for file_in in dataset.glob("*.jsonl.test"):
        file_part = file_in.stem[:-6]
        file_out = dataset / f"predict_{file_part}.json.test"
        if not file_in.exists():
            continue
        with file_out.open("w") as out:
            with file_in.open() as f:
                for line in f:
                    # FIX ME: I need here to read json and create a new json. not tsv.
                    # sentence, _ = line.strip().split("\t")
                    data = json.loads(line)
                    sentence = data["tokens"]
                    tags = data["tags"]
                    json.dump({"tokens": sentence, "tags": tags}, out)
                    out.write("\n")
