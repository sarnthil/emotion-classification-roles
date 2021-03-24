import json
import click
from pathlib import Path

FILES = {}


def get_file(dataset, part, split):
    if (dataset, part, split) in FILES:
        return FILES[dataset, part, split]
    path = (
        Path("workdata")
        / "position-experiment"
        / "datasets"
        / dataset
        / f"{part}.jsonl.{split}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    FILES[dataset, part, split] = path.open("w")
    return FILES[dataset, part, split]


def get_cue_annotations(annotations):
    # FIX ME
    cue_keys = [key for key in annotations if key.startswith("cue-")]
    state = "O"
    for parts in zip(*(annotations[cue_key] for cue_key in cue_keys)):
        if "B" in set(parts) or "I" in set(parts):
            state = "B" if state == "O" else "I"
        else:
            state = "O"
        yield state


@click.command()
@click.argument("file", type=click.File("r"))
def cli(file):
    eca_skipped = 0
    try:
        for line in file:
            data = json.loads(line)
            if data["dataset"] == "reman" and any(
                key.startswith("cue-") for key in data["annotations"]
            ):
                data["annotations"]["cue"] = list(
                    get_cue_annotations(data["annotations"])
                )
            if len(data["emotions"]) != 1:
                if data["dataset"] == "eca":
                    eca_skipped += 1
                    continue
            emotion = data["emotions"][0]  # is okay because of assertion above
            for role in data["annotations"]:
                if role == "coreference" or role.startswith("cue-"):
                    continue
                instance = {
                    "tokens": data["tokens"],
                    "tags": data["annotations"][role],
                    "gold-label": emotion,
                }
                file = get_file(data["dataset"], f"iob-{role}", data["split"])
                file.write(json.dumps(instance))
                file.write("\n")
    finally:
        for fp in FILES:
            FILES[fp].close()
    print("ECA skipped", eca_skipped)


if __name__ == "__main__":
    cli()
