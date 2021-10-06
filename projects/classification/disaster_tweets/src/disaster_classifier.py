from typing import Dict, Optional
from allennlp.data.fields.tensor_field import TensorField
from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator

from src.loss import LabelSmoothSoftmaxCEV1
from src.modules import PosWiseFeedForward

@Model.register('disaster')
class BasicClassifier(Model):
    """ This model implements a basic text classifier. After embedding the text into a text field, we will 
    optionally encode the embeddings `Seq2SeqEncoder`. The resulting sequence is pooled using a `Seq2VecEncoder` and then 
    it passed to a linear classification layer"""
    
    
    def __init__(self, 
                vocab: Vocabulary,
                text_field_embedder: TextFieldEmbedder,
                seq2vec_encoder: Seq2VecEncoder=None,
                seq2seq_encoder: Seq2SeqEncoder=None,
                extra_token_encoder: Seq2VecEncoder=None,
                feedforward: Optional[FeedForward] = None,
                dropout:float=None,
                num_labels:int = None,
                label_namespace:str="labels",
                namespace:str='tokens',
                poswise_feedforward: Optional[PosWiseFeedForward]=None,
                initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = RegularizerApplicator(),
                 smoothing:float = 0.1,
            )->None:
        
        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        self._poswiseFeedforward = poswise_feedforward # PoswiseFeedForwardNet(d_model=100, d_ff = 400)
        self._smoothing = smoothing
        self._extra_token_encoder = extra_token_encoder
        
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()
            
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
            
        else:
            self._dropout = None
            
        self._label_namespace = label_namespace
        self._namespace = namespace
        
        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = LabelSmoothSoftmaxCEV1()
        
        initializer(self)
        
        
    def forward(
        self,
        tokens: TextFieldTensors,
        meta: torch.FloatTensor,
        label:  torch.IntTensor = None,
    )->Dict[str, torch.Tensor]:
        
        embedded_text = self._text_field_embedder(tokens)
        
        mask = get_text_field_mask(tokens)
        
        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)
        # import pdb
        # pdb.set_trace()
        # adding position wise feed forward
        if self._poswiseFeedforward:
            embedded_text = self._poswiseFeedforward(embedded_text)
        
        if self._seq2vec_encoder:
            embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)
        
        embedded_text = torch.cat([embedded_text, meta],dim=-1)

        if self._feedforward:
            embedded_text = self._feedforward(embedded_text.float())
        
        
        logits = self._classification_layer(embedded_text)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {'logits': logits, 'probs': probs}
        #output_dict['token_ids'] = tokens['tags']['tokens']
        output_dict['token_ids'] = util.get_token_ids_from_text_field_tensors(tokens)

        if label is not None:
            # smooth labeling
            loss = self._loss(logits, label)
            output_dict['loss'] = loss
            self._accuracy(logits, label)
            
        return output_dict
    
    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Does a simple check if prob is above threshold and asign label"""
        
        predictions = output_dict['probs']
        
        if predictions.dim() == 2:
            prediction_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            prediction_list = [predictions]
            
        classes = []
        for prediction in prediction_list:
            label_idx = prediction.argmax(dim=-1).item()
            #label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(label_idx, str(label_idx))
            
            classes.append(label_idx)
        
        output_dict['label'] = classes
        tokens = []
        #import pdb
        #pdb.set_trace()
        for instance_tokens  in output_dict['token_ids']:
                tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace="tokens")
                    for token_id in instance_tokens
                ]
            )
        
        output_dict['tokens'] = tokens
        return output_dict
    
    def get_metrics(self, reset: bool) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics