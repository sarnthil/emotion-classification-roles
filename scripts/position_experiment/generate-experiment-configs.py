import json
from pathlib import Path

import click


@click.command()
@click.option("--use-one-hot", is_flag=True, default=False)
def cli(use_one_hot):
    for dataset in Path("workdata/position-experiment/datasets").glob("*"):
        for devpath in dataset.glob("*.dev"):
            trainpath = devpath.parent / (devpath.stem + ".train")
            testpath = devpath.parent / (devpath.stem + ".test")
            data = {
                "dataset_reader": {"type": "jsonl_reader"},
                "train_data_path": str(trainpath),
                "validation_data_path": str(devpath),
                "test_data_path": str(testpath),
                "model": {
                    "type": "lstm_gold",
                    "text_field_embedder": {
                        "token_embedders": {
                            "tokens": {
                                "type": "embedding",
                                "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
                                "embedding_dim": 300,
                                "trainable": True,
                            },
                        },
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": 303 if use_one_hot else 301,
                        "hidden_size": 64,
                        "bidirectional": True,
                        "dropout": 0.35,
                        "num_layers": 1,
                    },
                    "regularizer": [
                        [
                            ".*",
                            {
                                "type": "l2",
                                "alpha": 0.0012,
                                }
                            ]],
                    "initializer": [
                         ["encoder._module.weight_ih.*", {"type": "kaiming_uniform"}],
                         ["encoder._module.weight_hh.*", {"type": "orthogonal"}],
                         [".*bias_ih.*", {"type": "zero"}],
                         [".*bias_hh.*", {"type": "lstm_hidden_bias"}],],
                    "mode": "one-hot" if use_one_hot else "simple",
                },
                "iterator": {
                    "type": "basic",
                    "batch_size": 32,
                },
                "trainer": {
                    "optimizer": {"type": "adam", "lr": 0.0004,},
                    "num_epochs": 100,
                    "patience": 25,
                    "cuda_device": 3,
                    "learning_rate_scheduler": {
                         "type": "reduce_on_plateau",
                         "factor": 0.2,
                         "mode": "min",
                    },
                    "grad_clipping": 5.0,
                    "grad_norm": 1.0,
                    "should_log_parameter_statistics": True,
                    "should_log_learning_rate": True,
                    "num_serialized_models_to_keep": 1,
                },
                "evaluate_on_test": True,
                "random_seed": 2021,
                "numpy_seed": 2021,
                "pytorch_seed": 2021,

            }
            outfile = Path(
                f"workdata/position-experiment/configs/{dataset.name}/{devpath.stem[:-6]}.json"
            )
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with outfile.open("w") as f:
                json.dump(data, f)


if __name__ == "__main__":
    cli()
