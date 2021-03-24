from typing import Dict, Optional, List, Any
from itertools import zip_longest

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, TokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import RegularizerApplicator, InitializerApplicator
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
from overrides import overrides


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
    def __init__(
            self,
             vocab: Vocabulary,
             text_field_embedder: TextFieldEmbedder,
             encoder: Seq2VecEncoder,
             initializer: InitializerApplicator = InitializerApplicator(),
             regularizer: Optional[RegularizerApplicator] = None,
        ) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        initializer(self)
        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size("labels"),
        )

        # Monitor the metrics - we use accuracy, as well as prec, rec, f1
        self._accuracy = CategoricalAccuracy()
        self._f1_measure = FBetaMeasure(average="macro")
        self.loss_function = torch.nn.CrossEntropyLoss()



    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.Tensor = None)-> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ,no-member

        embedded_text_input = self.text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        # Shape: batch x seq_len x emb_dim
        encoded_sequence = self.encoder(embedded_text_input, mask)

        logits = self.linear(encoded_sequence)
        probs = F.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            self._accuracy(logits, label)
            self._f1_measure(logits, label)
            output_dict["loss"] = self.loss_function(logits, label)

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
