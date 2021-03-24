import logging
import random
from pathlib import Path
from datetime import datetime, timezone

import click
import datasets as nlp
import torch
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import DistilBertTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DistilBertForTokenClassification
from transformers import RobertaTokenizer


def new_call(self, *args, **kwargs):
    return super(type(self), self).__call__(
        *args, **kwargs, is_split_into_words=True
    )


RobertaTokenizer.__call__ = new_call


def f1_micro(labels, preds):
    return f1_score(labels, preds, average="micro")

def f1_macro(labels, preds):
    return f1_score(labels, preds, average="macro")

def recall_macro(labels, preds):
    return recall_score(labels, preds, average="macro")

def recall_micro(labels, preds):
    return recall_score(labels, preds, average="micro")

def precision_macro(labels, preds):
    return precision_score(labels, preds, average="macro")

def precision_micro(labels, preds):
    return precision_score(labels, preds, average="micro")



def read_data(dataset, split):
    texts = []
    labels = []
    for doc in dataset[split]:
        texts.append(" ".join(token for token in doc["sentence"]))
        labels.append(doc["emotion"])
    return texts, labels


@click.command()
@click.option("--dataset", "-d", required=True)
@click.option("--mask-type", "-m", required=True)
@click.option("--role", "-r")
def cli(dataset, mask_type, role):
    if mask_type in ("all", "inbandall"):
        name = f"unified_{dataset}_{mask_type}"
    else:
        if not role:
            raise click.BadParameter("Role is missing")
        name = f"unified_{dataset}_{mask_type}_{role}"
    dataset = nlp.load_dataset("scripts/unified-loader.py", name=name)

    train_texts, train_labels = read_data(dataset, "train")
    test_texts, test_labels = read_data(dataset, "test")
    val_texts, val_labels = read_data(dataset, "validation")

    unique_labels = set(train_labels)
    label2id = {label: id for id, label in enumerate(unique_labels)}
    id2label = {id: label for label, id in label2id.items()}

    train_data = []
    for train_text, train_label in zip(train_texts, train_labels):
        train_data.append([train_text, label2id[train_label]])

    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "label"]

    eval_data = []
    for test_text, test_label in zip(test_texts, test_labels):
        eval_data.append([test_text, label2id[test_label]])

    eval_df = pd.DataFrame(eval_data)
    eval_df.columns = ["text", "label"]

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Create a ClassificationModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=len(unique_labels),
        args={
            "reprocess_input_data": True,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "overwrite_output_dir": True,
            "num_train_epochs": 5,
            "n_gpu": 3,
            "learning_rate": 5e-5,
            "use_early_stopping": True,
            "early_stopping_patience": 3,
            "manual_seed": 4,
            "no_cache": True,
        },
    )

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    
    result, model_outputs, wrong_predictions = model.eval_model(
        eval_df, acc=accuracy_score, f1_micro=f1_micro, f1_macro=f1_macro, recall_macro=recall_macro, recall_micro=recall_micro, precision_macro=precision_macro, precision_micro=precision_micro)
    print(id2label[predictions[0]])
    now = int(datetime.now().astimezone(timezone.utc).timestamp())
    source = Path("outputs/eval_results.txt")
    target = Path(f"results/eval-{name}-{now}.txt")
    source.rename(target)


if __name__ == "__main__":
    cli()
