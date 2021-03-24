import subprocess
import click
from pathlib import Path

# workdata/results/eca/text_emotion_without_cause.txt: workdata/allennlp-models/eca/text_emotion_without_cause/ workdata/eca/text_emotion_with
# out_cause.tsv.test_clean scripts/text_classification_tab.py scripts/lstm_classifier.py workdata/results

# allennlp evaluate --include-package scripts.text_classification_tab --include-package scripts.lstm_classifier workdata/allennlp-models/eca/text_emotion_without_cause workdata/eca/text_emotion_without_cause.tsv.test_clean 2>workdata/results/eca/text_emotion_without_cause.txt

@click.command()
@click.argument("dataset")
def cli(dataset):
    configbase = Path("workdata/masking-experiment/configs") / dataset
    modelbase = configbase.parent.parent / "models" / dataset
    for configpath in configbase.glob("*.json"):
        modelpath = modelbase / configpath.stem
        modelpath.parent.mkdir(exist_ok=True, parents=True)
        print("Evaluating", modelpath)
        subprocess.call(
            [
                "allennlp",
                "evaluate",
                str(configpath),
                "-s",
                str(modelpath),
                "--include-package",
                "scripts.reader",
                "--include-package",
                "scripts.lstm",
            ]
        )


if __name__ == "__main__":
    cli()
