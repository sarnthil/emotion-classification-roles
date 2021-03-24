import subprocess
import click
from pathlib import Path

# workdata/predictions/eca_all.predictions: workdata/eca/predict_emotion_all.json.test workdata/allennlp-models/eca/text_emotion_all/ scripts/
# text_classification_tab.py scripts/lstm_classifier.py scripts/predictor.py

# allennlp predict --include-package scripts.text_classification_tab --include-package scripts.lstm_classifier --predictor emo_predictor --include-package scripts.predictor --output-file workdata/predictions/eca_all.predictions workdata/allennlp-models/eca/text_emotion_all workdata/eca/predict_emotion_all.json.test

@click.command()
@click.argument("dataset")
def cli(dataset):
    configbase = Path("workdata/masking-experiment/configs") / dataset
    modelbase = Path("workdata/masking-experiment/models") / dataset
    outputbase = Path("workdata/masking-experiment/predictions") / dataset
    outputbase.mkdir(parents=True, exist_ok=True)
    for testpath in Path(f"workdata/masking-experiment/datasets/{dataset}").glob("predict_*.json.test"):
        _predict, _, tail = testpath.name.partition("_")
        file_part, _json, _rest = tail.partition(".json")
        # workdata/masking-experiment/models/eca/all/
        modelpath = modelbase / file_part
        print("Making predictions", modelpath)
        subprocess.call(
            [
                "allennlp",
                "predict",
                "--include-package",
                "scripts.masking_experiment.reader",
                "--include-package",
                "scripts.masking_experiment.lstm",
                "--predictor",
                "emo_predictor",
                "--include-package",
                "scripts.masking_experiment.predictor",
                "--output-file",
                str(outputbase / file_part) + ".predictions",
                str(modelpath),
                str(testpath),
            ]
        )


if __name__ == "__main__":
    cli()
