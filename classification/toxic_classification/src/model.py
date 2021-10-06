from typing import Optional, Dict
import torch

from allennlp.common import Params
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import BooleanAccuracy
from overrides import overrides


@Model.register('toxic')
class ToxicModel(Model):
    def __init__(self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        initializer: InitializerApplicator= InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = RegularizerApplicator())->None:

        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.labels = ['severe_toxicity','obscene','identity_attack', 'insult','threat']
        initializer(self)
    

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Does a simple argmax over the probabilities, converts index to string label,
            and add label key to the dictionary with the result """

        prediction_list = output_dict['probabilities']
        classes = []

        for prediction in prediction_list:
            # Its a multilabelclassification so , need to iterate through all of the labels.
            final_labels = [self.labels[i] for i,p in enumerate(prediction) if p.item() > 0.5]
            classes.append(final_labels)

        output_dict['label'] = classes
        return output_dict


    def forward(self,
        text: Dict[str, torch.Tensor],
        extras: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]=None,
        threat_labels: Dict[str, torch.Tensor]=None )-> Dict[str, torch.Tensor]:

        import pdb
        pdb.set_trace()
        embedded_text = self.text_field_embedder(text)
        mask = util.get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)

        logits = self.classifier_feedforward(encoded_text)
        probabilities = torch.sigmoid(logits)

        output_dict= {'logits': logits, 'probabilities': probabilities}

        return output_dict