from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance
import csv
import string
from wordcloud import STOPWORDS
import numpy as np

import allennlp
from allennlp.models.archival import archive_model
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
import torch
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, text_field_embedder
from allennlp.training import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamaxOptimizer

from src.tweet_reader import TweetReader
from src.disaster_classifier import BasicClassifier

model_name = 'roberta-base'
max_length = 5000
metafeatures = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length', 'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']

def create_metafeatures(x):
    
    meta = {
        'word_count': len(x.split()),
        'unique_word_count': len(set(str(x).split())),
        'stop_word_count': len([w for w in str(x).lower().split() if w in STOPWORDS]),
        'url_count': len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]),
        'mean_word_length': np.mean([len(w) for w in str(x).split()]),
        'char_count': len(str(x)),
        'punctuation_count': len([c for c in str(x) if c in string.punctuation]),
        'hashtag_count': len([c for c in str(x) if c == '#']),
        'mention_count': len([c for c in str(x) if c == '@'])
    }
    return np.array([float(meta[f]) for f in metafeatures],dtype=np.float)
    

def build_model(vocab:Vocabulary) ->Model:
    print("Building the model")
    
    embedder = BasicTextFieldEmbedder(
        {'tokens': PretrainedTransformerEmbedder(model_name=model_name)}
    )
    seq2vec_encoder = BertPooler(pretrained_model=model_name, dropout=0.1)
    # adding extra 9 dim vector from metafeatures
    
    feedforward = FeedForward(num_layers=2, input_dim=768 + 9, hidden_dims=[256, 6], activations= [
        Activation.by_name('tanh')(), 
        Activation.by_name('linear')()],
      dropout= [0.3, 0.0])
    
    return BasicClassifier(vocab=vocab,
                           text_field_embedder=embedder,
                           seq2vec_encoder=seq2vec_encoder,
                           feedforward=feedforward,
                           namespace='tags',
                           num_labels=2,
                           smoothing=0.1)
    
def build_dataset_reader()->DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(model_name= model_name)
    token_indexer = {'tokens': PretrainedTransformerIndexer(model_name=model_name)}
    return TweetReader(tokenizer=tokenizer, 
                       token_indexers= token_indexer, 
                       max_length=max_length,
                       manual_distributed_sharding=True, 
                       manual_multiprocess_sharding=True)

@Predictor.register('disaster')
class DisasterPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["text"]
        meta = create_metafeatures(sentence)
        return self._dataset_reader.text_to_instance(text=sentence, metafeatures=meta)


    def predict(self, sentence: str)->JsonDict:
        return self.predict_json({'text': sentence})
    
    
def predict_from_archive(archive_path, sentence):
    predictor = Predictor.from_archive(archive_path,"disaster")
    return predictor.predict(sentence)


def predict_from_model(model_path, vocab_path, sentence):
    vocab = Vocabulary.from_files(vocab_path)
    reader = build_dataset_reader()
    model = build_model(vocab=vocab)
    
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    
    predictor = DisasterPredictor(model=model, dataset_reader=reader)
    output = predictor.predict(sentence)
    return output
    
if __name__ == '__main__':
    #predict_from_archive()
    sentence = "I love Berlin!"
    model_path = "tmp/best.th"
    vocab_path = "tmp/vocabulary"
    
    output = predict_from_model(model_path = model_path, vocab_path=vocab_path, sentence=sentence)
    print(output)