import json
from pathlib import Path

import click


@click.command()
def cli():
    # We reuse the position experiments datasets
    for dataset in Path("workdata/position-experiment/datasets").glob("*"):
        for devpath in dataset.glob("*.dev"):
            trainpath = devpath.parent / (devpath.stem + ".train")
            testpath = devpath.parent / (devpath.stem + ".test")
            data = {
                "dataset_reader": {"type": "jsonl_reader"},
                "train_data_path": str(trainpath),
                "test_data_path": str(testpath),
                "validation_data_path": str(devpath),
                "model": {
                    "type": "lstm_gold",
                    "text_field_embedder": {
                        "token_embedders": {
                            "tokens": {
                                "type": "embedding",
                                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
                                "embedding_dim": 300,
                                "trainable": True,
                            },
                        },
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": 300,
                        "hidden_size": 64,
                        "bidirectional": True,
                        "dropout": 0.3,
                        "num_layers": 2,
                    },
                    "regularizer": [[".*", {"type": "l2", "alpha": 0.00012}]],
                    "initializer": [
                        [
                            "encoder._module.weight_ih.*",
                            {"type": "kaiming_uniform"},
                        ],
                        ["encoder._module.weight_hh.*", {"type": "orthogonal"}],
                        [".*bias_ih.*", {"type": "zero"}],
                        [".*bias_hh.*", {"type": "lstm_hidden_bias"}],
                    ],
                },
                "iterator": {"type": "basic", "batch_size": 32,},
                "trainer": {
                    "optimizer": {"type": "adam", "lr": 0.0003},
                    "num_epochs": 100,
                    "patience": 50,
                    "cuda_device": 2,
                    "learning_rate_scheduler": {
                        "type": "reduce_on_plateau",
                        "mode": "min",
                        "factor": 0.2,
                        "verbose": False,
                    },
                    "should_log_parameter_statistics": True,
                    "should_log_learning_rate": True,
                    "grad_clipping": 5.0,
                    "grad_norm": 10.0,
                    "num_serialized_models_to_keep": 1,
                },
                "evaluate_on_test": True,
                "random_seed": 2021,
                "numpy_seed": 2021,
                "pytorch_seed": 2021,
            }
            outfile = Path(
                f"workdata/indicator-experiment/configs/{dataset.name}/{devpath.stem[:-6]}.json"
            )
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with outfile.open("w") as f:
                json.dump(data, f)


if __name__ == "__main__":
    cli()
