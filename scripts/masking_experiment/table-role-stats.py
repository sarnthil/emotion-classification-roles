from pathlib import Path
from statistics import stdev, mean
from collections import defaultdict

datasets = ["emotion-stimulus", "electoral_tweets", "eca", "gne", "reman"]
roles = ["cause", "cue", "target", "experiencer"]
file_parts = ["all", *(f"only_{role}" for role in roles)]


for dataset in datasets:
    results = defaultdict(lambda: {"lines": 0, "words": 0})
    lengths = defaultdict(list)
    for file_part in file_parts:
        for split in ["dev", "train", "test"]:
            filename = Path(f"workdata/masking-experiment/datasets/{dataset}/{file_part}.tsv.{split}")
            if not filename.exists():
                continue
            with filename.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    text, _ = line.rstrip("\n").split("\t")
                    words = [word for word in text.split(" ") if word != "X"]
                    if not words:
                        continue
                    # results[file_part]["lines"] += 1
                    # results[file_part]["words"] += len(words)
                    lengths[file_part].append(len(words))
    print(dataset, end="")
    for file_part in file_parts:
        if file_part in lengths:
            # lines, words = results[file_part]["lines"], results[file_part]["words"]
            print(
                "",
                len(lengths[file_part]),
                f"{mean(lengths[file_part]):.2f}",
                # f"{stdev(lengths[file_part]):.2f}",
                sep=" & ",
                end="",
            )
        else:
            print(" & -- & --", end="")
    print(r"\\")
