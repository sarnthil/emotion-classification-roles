from typing import Dict, Optional, List, Any
from itertools import zip_longest

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.common import Params
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn.util import get_text_field_mask
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import (
    Seq2VecEncoder,
    PytorchSeq2VecWrapper,
)
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure
from allennlp.training.trainer import Trainer
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import Initializer
from overrides import overrides




def one_hot_word(word, labels):
    word_value = labels.get(word, -1)
    return [1.0 if i == word_value else 0.0 for i in range(len(labels))]


def one_hot_instance(instance, labels, pad_size):
    return [
        one_hot_word(word, labels) for word, _ in zip_longest(instance, range(pad_size))
    ]


def one_hot_batch(batch, labels, pad_size):
    return [one_hot_instance(instance, labels, pad_size) for instance in batch]


def map_from_labels(labels):
    return {label: i for i, label in enumerate(labels)}


def simple_mapped_instance(instance, the_map, pad_size):
    return [
        [the_map.get(word, 0)] for word, _ in zip_longest(instance, range(pad_size))
    ]



def simple_mapped_batch(batch, the_map, pad_size):
    return [simple_mapped_instance(instance, the_map, pad_size) for instance in batch]



@Model.register('lstm_gold')
class LSTMGold(Model):
    """
    Classifies emotion class for using gold iob tags for roles.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        An encoder that will learn the major logic of the task.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    init = Initializer.from_params(Params({"type": "kaiming_uniform"}))
    def __init__(
            self,
             vocab: Vocabulary,
             text_field_embedder: TextFieldEmbedder,
             encoder: Seq2VecEncoder,
             initializer: InitializerApplicator = InitializerApplicator(),
             regularizer: Optional[RegularizerApplicator] = None,
             mode: str = "one-hot",
             var_dropout: float = 0.35,
        ) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        self.text_field_embedder = text_field_embedder
        self._variational_dropout = InputVariationalDropout(var_dropout)
        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size("labels"),
        )

        self._accuracy = CategoricalAccuracy()
        self._f1_measure = FBetaMeasure(average="macro")
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.one_hot_map = map_from_labels("BIO")
        self.do_one_hot = mode == "one-hot"
        # uniform = Initializer.from_params(Params({"type": "uniform"}))
        # initializer: InitializerApplicator = InitializerApplicator([uniform]),
        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                tags: List[str],
                label: torch.Tensor = None)-> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ,no-member

        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        embedded_text_input = self._variational_dropout(embedded_text_input)

        pad_size = tokens["tokens"].shape[1]
        if embedded_text_input.is_cuda:
            tensor_constructor = torch.cuda.FloatTensor
        else:
            tensor_constructor = torch.Tensor

        if self.do_one_hot:
            tag_tensor = tensor_constructor(one_hot_batch(tags, self.one_hot_map, pad_size))
        else:
            tag_tensor = tensor_constructor(simple_mapped_batch(tags, {"B": 3, "I": 3, "O": -3}, pad_size))

        embedded_combination = torch.cat((embedded_text_input, tag_tensor), dim=2)

        # Shape: batch x seq_len x emb_dim
        encoded_sequence = self.encoder(embedded_combination, mask)

        logits = self.linear(encoded_sequence)
        probs = F.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            self._accuracy(logits, label)
            self._f1_measure(logits, label)
            output_dict["loss"] = self.loss_function(logits, label)

        # Store the iob tags in the output for later decoding
        output_dict["tags"] = tags


        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}

        if not self.training:
            all_metrics.update(
                {"accuracy": self._accuracy.get_metric(reset=reset)}
            )
            all_metrics.update(
                {"f1": self._f1_measure.get_metric(reset=reset)["fscore"]}
            )
        return all_metrics
