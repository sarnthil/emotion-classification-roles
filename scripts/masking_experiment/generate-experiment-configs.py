import sys
import json
import click
from pathlib import Path

@click.command()
def cli():
    for dataset in Path("workdata/masking-experiment/datasets").glob("*"):
        for devpath in dataset.glob("*.dev"):
            trainpath = devpath.parent / (devpath.stem + ".train")
            testpath = devpath.parent / (devpath.stem + ".test")
            data = {
                "dataset_reader": {
                    "type": "emo_reader"
                },
                "train_data_path": str(trainpath),
                "test_data_path": str(testpath),
                "validation_data_path": str(devpath),

                "model": {
                    "type": "lstm_classifier",

                    "word_embeddings": {
                        "tokens": {
                            "type": "embedding",
                            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
                            "embedding_dim": 300,
                            "trainable" : True

                        }
                    },

                    "encoder": {
                        "type": "lstm",
                        "input_size": 300,
                        "hidden_size": 64,
                        "bidirectional": True,
                        "dropout": 0.3,
                        "num_layers" : 1,
                    },
                    "regularizer": [
                        [
                            ".*",
                            {
                                "type": "l2",
                                "alpha": 0.0012,
                                }
                            ]
                        ],
                    "initializer": [
                         ["encoder._module.weight_ih.*", {"type": "kaiming_uniform"}],
                         ["encoder._module.weight_hh.*", {"type": "orthogonal"}],
                         [".*bias_ih.*", {"type": "zero"}],
                         [".*bias_hh.*", {"type": "lstm_hidden_bias"}],],
                },
                "iterator": {
                    "type": "bucket",
                    "batch_size": 32,
                    "sorting_keys": [["tokens", "num_tokens"]]
                },
                "trainer": {
                    "optimizer": {
                        "type": "adam",
                        "lr": 0.0004
                        },
                    "num_epochs": 100,
                    "patience": 50,
                    "cuda_device": 2,
                    "learning_rate_scheduler": {
                        "type": "reduce_on_plateau",
                        "mode": "min",
                        "factor": 0.2,
                        "patience": 3,
                        "verbose": False
                        },
                    "should_log_parameter_statistics": True,
                    "should_log_learning_rate": True,
                    "grad_clipping": 5.0,
                    "grad_norm": 1.0,
                    "num_serialized_models_to_keep": 1,
                },
                "evaluate_on_test": True,
                "random_seed": 2021,
                "numpy_seed": 2021,
                "pytorch_seed": 2021,

            }
            outfile = Path(f"workdata/masking-experiment/configs/{dataset.name}/{devpath.stem[:-4]}.json")
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with outfile.open("w") as f:
                json.dump(data, f)

if __name__ == '__main__':
    cli()
