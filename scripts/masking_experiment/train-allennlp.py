import subprocess
import click
from pathlib import Path


@click.command()
@click.argument("dataset")
def cli(dataset):
    configbase = Path("workdata/masking-experiment/configs") / dataset
    modelbase = configbase.parent.parent / "models" / dataset
    for configpath in configbase.glob("*.json"):
        modelpath = modelbase / configpath.stem
        modelpath.parent.mkdir(exist_ok=True, parents=True)
        print("Training", modelpath)
        subprocess.call(
            [
                "allennlp",
                "train",
                str(configpath),
                "-s",
                str(modelpath),
                "--include-package",
                "scripts.masking_experiment.reader",
                "--include-package",
                "scripts.masking_experiment.lstm",
            ]
        )


if __name__ == "__main__":
    cli()
