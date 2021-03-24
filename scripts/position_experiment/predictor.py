from overrides import overrides
from allennlp.predictors.text_classifier import TextClassifierPredictor
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance


@Predictor.register('predictor_iob')
class PredictorIOB(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')
        outputs['tokens'] = [str(token) for token in instance.fields['tokens'].tokens]
        # outputs['tags'] = [str(tag) for tag in instance.fields['tags'].tags]
        outputs['predicted'] = label_vocab[outputs['logits'].argmax()]
        outputs.pop('logits')
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            tokens=json_dict['tokens'],
            tags = json_dict['tags'],
        )

