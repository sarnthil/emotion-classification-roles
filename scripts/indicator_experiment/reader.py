import math
import json
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

def indicator_tokens(tokens, tags):
    state = "O"
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if state in "BI" and tag in "OB":
            yield "</role>"
        if tag == "B":
            yield "<role>"
        yield tok
        state = tag
    if state != "O":
        yield "</role>"


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
                tokens = [*indicator_tokens(tokens, tags)] if tags else tokens
                label = datum['gold-label'] if 'gold-label' in datum else None
                yield self.text_to_instance(tokens, label)


    @overrides
    def text_to_instance(self, tokens: List[str], label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_tokens = [Token(tok) for tok in tokens]
        tokens_field = TextField(tokenized_tokens, self._token_indexers)
        fields = {'tokens': tokens_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
