from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance

@Predictor.register('toxic')
class ToxicPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)
