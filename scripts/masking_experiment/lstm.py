from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
    StanfordSentimentTreeBankDatasetReader,
)
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
import torch
import torch.nn.functional as F
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules.input_variational_dropout import InputVariationalDropout


# EMBEDDING_DIM = 128
# HIDDEN_DIM = 128


@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        vocab: Vocabulary,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: RegularizerApplicator = RegularizerApplicator(),
        var_dropout: float = 0.35,
    ) -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self._variational_dropout = InputVariationalDropout(var_dropout)

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size("labels"),
        )

        self._accuracy = CategoricalAccuracy()
        self._f1_measure = FBetaMeasure(average="macro")
        self.loss_function = torch.nn.CrossEntropyLoss()
       
    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(
        self, tokens: Dict[str, torch.Tensor], label: torch.Tensor = None
    ) -> torch.Tensor:
        
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        embeddings = self._variational_dropout(embeddings)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        probs = F.softmax(logits, dim=-1)
        output = {"logits": logits, "probs": probs}

        if label is not None:
            self._accuracy(logits, label)
            self._f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

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
