import json
from pathlib import Path
from collections import defaultdict, Counter

from sklearn.metrics import precision_recall_fscore_support

PRECISION_FALLBACK = RECALL_FALLBACK = 1

# dataset -> setting -> emotion -> measure -> score
results = {}

# for file in Path("workdata/masking-experiment/predictions").glob("*.aggregated"):
for dataset_path in Path("workdata/masking-experiment/predictions").glob("*"):
    dataset = dataset_path.name
    for file in dataset_path.glob("*.aggregated"):
        part = file.stem
        results.setdefault(dataset, {}).setdefault(part, {})
        confusion_matrix = defaultdict(Counter)
        y_true, y_pred = [], []
        instances = 0
        with file.open() as f:
            for line in f:
                data = json.loads(line)
                confusion_matrix[data["gold"]][data["prediction"]] += 1
                y_true.append(data["gold"])
                y_pred.append(data["prediction"])
                instances += 1
        for emotion in confusion_matrix:
            tp = confusion_matrix[emotion][emotion]
            fn = sum(
                confusion_matrix[emotion][other]
                for other in confusion_matrix[emotion]
                if other != emotion
            )
            fp = sum(
                confusion_matrix[other][emotion]
                for other in confusion_matrix
                if other != emotion
            )
            tn = instances - tp - fn - fp
            precision = tp / (tp + fp) if tp + fp else PRECISION_FALLBACK
            recall = tp / (tp + fn) if tp + fn else RECALL_FALLBACK
            f1 = (
                2 * ((precision * recall) / (precision + recall))
                if precision and recall
                else 0
            )
            results[dataset][part][emotion] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        emos = list(results[dataset][part].keys())

        for average in ["macro", "micro"]:
            p, r, f, s = precision_recall_fscore_support(
                y_true, y_pred, zero_division=1, average=average
            )
            results[dataset][part][f"all_{average}"] = {
                "precision": p,
                "recall": r,
                "f1": f,
            }
        results[dataset][part]["all_unweighted_mean"] = {
            "precision": sum(
                results[dataset][part][emo]["precision"] for emo in emos
            )
            / len(emos),
            "recall": sum(results[dataset][part][emo]["recall"] for emo in emos)
            / len(emos),
            "f1": sum(results[dataset][part][emo]["f1"] for emo in emos)
            / len(emos),
        }
        results[dataset][part]["all_weighted_mean"] = {
            "precision": sum(
                results[dataset][part][emo]["precision"]
                * sum(
                    confusion_matrix[emo][other]
                    for other in confusion_matrix[emo]
                )
                for emo in emos
            )
            / instances,
            "recall": sum(
                results[dataset][part][emo]["recall"]
                * sum(
                    confusion_matrix[emo][other]
                    for other in confusion_matrix[emo]
                )
                for emo in emos
            )
            / instances,
            "f1": sum(
                results[dataset][part][emo]["f1"]
                * sum(
                    confusion_matrix[emo][other]
                    for other in confusion_matrix[emo]
                )
                for emo in emos
            )
            / instances,
        }

    with Path("workdata/masking-experiment/results.json").open("w") as f:
        json.dump(results, f)
