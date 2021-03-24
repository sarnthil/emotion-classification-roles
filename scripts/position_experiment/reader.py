import json
import math
from random import random
from typing import Dict, List, Iterable, Any, Tuple

import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Token
from allennlp.data.fields import SequenceLabelField, MetadataField, TextField, \
    ListField, LabelField, IndexField, AdjacencyField, ArrayField, SequenceField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides


@DatasetReader.register('jsonl_reader')
class JsonlReader(DatasetReader):
    """
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__()

        default_indexer = {'tokens': SingleIdTokenIndexer()}
        self._token_indexers = token_indexers or default_indexer

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as data_file:
            for idx, line in enumerate(data_file):
                datum = json.loads(line.strip())
                tokens = datum['tokens']
                tags = datum['tags'] if 'tags' in datum else None
                label = datum['gold-label'] if 'gold-label' in datum else None
                yield self.text_to_instance(tokens, tags, label)


    @overrides
    def text_to_instance(self, tokens: List[str], tags: List[str], label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_tokens = [Token(tok) for tok in tokens]
        tokens_field = TextField(tokenized_tokens, self._token_indexers)
        tags_field = MetadataField(tags)
        # tags_field = TextField(tokenized_tags, self._token_indexers)
        fields = {'tokens': tokens_field, 'tags': tags_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

