import subprocess
import click
from pathlib import Path


@click.command()
@click.argument("dataset")
def cli(dataset):
    configbase = Path("workdata/indicator-experiment/configs") / dataset
    modelbase = Path("workdata/indicator-experiment/models") / dataset
    outputbase = Path("workdata/indicator-experiment/predictions") / dataset
    outputbase.mkdir(parents=True, exist_ok=True)
    for testpath in Path(f"workdata/position-experiment/datasets/{dataset}").glob("predict_*.json.test"):
        _predict, _, tail = testpath.name.partition("_")
        file_part, _json, _rest = tail.partition(".json")
        modelpath = modelbase / file_part
        print("Making predictions", modelpath)
        subprocess.call(
            [
                "allennlp",
                "predict",
                "--include-package",
                "scripts.indicator_experiment.reader",
                "--include-package",
                "scripts.indicator_experiment.lstm",
                "--predictor",
                "predictor_iob",
                "--include-package",
                "scripts.indicator_experiment.predictor",
                "--output-file",
                str(outputbase / file_part) + ".predictions",
                str(modelpath),
                str(testpath),
            ]
        )


if __name__ == "__main__":
    cli()
