import json
from itertools import chain, combinations
from pathlib import Path

import datasets

# TODO: add inbandall positional indicators jointly
MARK_TYPES = [
    "all",
    "only",
    "without",
    "inband",
    "inbandall",
]

DATASET_MAP = {
    "electoral_tweets": "et",
    "emotion-stimulus": "es",
}

ROLES = {
    "cause",
    "experiencer",
    "target",
    "cue",
}  # FIXME: cue is weird (per emotion) in Reman
ROLES_AND_EMOTION = (
    ROLES  # | {"emotion"}  # We predict emotion now, so it's never the role
)
DATASET_AVAILABLE_ROLES = {
    "electoral_tweets": ROLES_AND_EMOTION,
    "emotion-stimulus": {"emotion", "cause"},
    "reman": ROLES_AND_EMOTION,  # FIXME
    "gne": ROLES_AND_EMOTION,
    "eca": {"emotion", "cause"},
}


def iob_unifier(iob_lists):
    last_role = None
    keys = list(iob_lists.keys())
    if not keys:
        while True:
            yield "O", None, None
    for i, _ in enumerate(iob_lists[keys[0]]):
        for role in keys:
            if iob_lists[role][i] == "B":
                yield "B", last_role, role
                last_role = role
                break
        else:
            for role in keys:
                if iob_lists[role][i] == "I":
                    yield "I", last_role, role
                    last_role = role
                    break
            else:
                yield "O", last_role, last_role


def as_mark_type(mark_type, tokens, role_annotations):
    if mark_type == "all":
        raise RuntimeError("We should never reach this")
        return tokens, role_annotations
    elif mark_type == "only":
        return (
            [
                "<blank/>" if role_annotation in "O" else token
                for (token, role_annotation) in zip(tokens, [*role_annotations.values()][0])
            ],
            role_annotations[0],
        )
    elif mark_type == "without":
        return (
            [
                "<blank/>" if role_annotation in "BI" else token
                for (token, role_annotation) in zip(tokens, [*role_annotations.values()][0])
            ],
            role_annotations[0],
        )
    elif mark_type in ("inband", "inbandall"):
        r_tokens, r_annos = [], []
        state = "O"
        for token, (annotation, last_role, role) in zip(tokens, iob_unifier(role_annotations)):
            if annotation == "O":
                if state == "O":
                    r_tokens.append(token)
                    r_annos.append(annotation)
                else:
                    r_tokens.append(f"</{last_role}>")
                    r_annos.append("I")
                    r_tokens.append(token)
                    r_annos.append(annotation)
            elif annotation == "B":
                if state == "O":
                    r_tokens.append(f"<{role}>")
                    r_annos.append("B")
                    r_tokens.append(token)
                    r_annos.append("I")
                else:
                    r_tokens.append(f"</{last_role}>")
                    r_annos.append("I")
                    r_tokens.append(f"<{role}>")
                    r_annos.append("B")
                    r_tokens.append(token)
                    r_annos.append("I")
            elif annotation == "I":
                r_tokens.append(token)
                r_annos.append(annotation)
            state = annotation
        if state in "BI":
            r_tokens.append(f"</{role}>")
            r_annos.append("I")
        return r_tokens, r_annos
    raise RuntimeError("Unknown mark_type")


class UnifiedConfig(datasets.BuilderConfig):
    def __init__(self, *args, task, dataset, mark_type, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.dataset = dataset
        self.mark_type = mark_type


class UnifiedDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = UnifiedConfig
    BUILDER_CONFIGS = []
    for dataset in (
        "electoral_tweets",
        "emotion-stimulus",
        "reman",
        "gne",
        "eca",
    ):
        for mark_type in MARK_TYPES:
            if mark_type in ("all", "inbandall"):
                BUILDER_CONFIGS.append(
                    UnifiedConfig(
                        name=f"unified_{DATASET_MAP.get(dataset, dataset)}_{mark_type}",
                        description="",
                        task=mark_type,
                        dataset=dataset,
                        mark_type=mark_type,
                    )
                )
            else:
                for task in DATASET_AVAILABLE_ROLES[dataset]:
                    BUILDER_CONFIGS.append(
                        UnifiedConfig(
                            name=f"unified_{DATASET_MAP.get(dataset, dataset)}_{mark_type}_{task}",
                            description="",
                            task=task,
                            dataset=dataset,
                            mark_type=mark_type,
                        )
                    )

    def _info(self):
        features = {
            "sentence": datasets.Sequence(datasets.Value("string")),
            "emotion": datasets.Value("string"),
        }
        features[self.config.task] = datasets.Sequence(datasets.Value("string"))
        return datasets.DatasetInfo(
            description="Emotion stimulus part of unified dataset",
            # This defines the different columns of the dataset and their types
            features=datasets.Features(features),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://example.com",
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        file_path = Path("sources/unified.jsonl")
        gen_kwargs = {
            "filepath": file_path,
            "task": self.config.task,
            "dataset": self.config.dataset,
            "mark_type": self.config.mark_type,
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"split": "train", **gen_kwargs},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test", **gen_kwargs},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "dev", **gen_kwargs},
            ),
        ]

    def _generate_examples(self, filepath, split, task, dataset, mark_type):
        """ Yields examples. """
        role = self.config.task
        with filepath.open() as f:
            for row in f:
                data = json.loads(row)
                if data["dataset"] != dataset:
                    continue
                if split != data["split"]:
                    continue
                emotion = data["emotions"][0]
                try:
                    if data["dataset"] == "reman" and role == "cue":
                        data["annotations"][role] = data["annotations"][
                            f"{role}-{emotion}"
                        ]
                    tokens = data["tokens"]
                    if role == "all":
                        role_value = []
                    else:
                        roles = (
                            [role]
                            if role != "inbandall"
                            else list(data["annotations"])
                        )
                        role_values = {
                            role: (
                                [emotion]
                                if role == "emotion"
                                else data["annotations"][role]
                            )
                            for role in roles
                        }
                        tokens, role_value = as_mark_type(
                            mark_type, tokens, role_values
                        )
                    entry = (
                        data["id"],
                        {
                            "sentence": tokens,
                            "emotion": emotion,
                            role: role_value,
                        },
                    )
                    # if split == "test":
                    #     entry[1]["emotion"] = ""
                except KeyError:
                    continue
                yield entry
