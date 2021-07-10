from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, ListField, LabelField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import ensure_list
import torch
from allennlp.data import Vocabulary

import csv
from typing import Dict
from overrides import overrides
import numpy as np
import pandas as pd

features = {'len': lambda x: len(x),
            'capitals': lambda comment: sum(1 for c in comment if c.isupper())
            
            }

def extract_features(comment):
    return {key: func(comment) for key, func in features.items()}

@DatasetReader.register("toxic")
class ToxicDataReader(DatasetReader):
    def __init__(self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            max_sequence_length: int = None,
            max_count: int = -1)->None:

        super().__init__()
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_sequence_length = max_sequence_length
        self._max_count = max_count

    @overrides
    def _read(self, file_path):
        with open(file_path) as data_file:
            reader = csv.DictReader(data_file)
            for row in reader:
                if self._max_count == 0:
                    break
                if 'target' in row.keys():
                    yield self.text_to_instance(
                        comment_text = row['comment_text'],
                        target = row['target'],
                        threat_labels =  [float(row.get(k)) for k in ['severe_toxicity','obscene','identity_attack', 'insult','threat']]
                    )
                else:
                    yield self.text_to_instance(
                        comment_text = row['comment_text'],
                        target = None,
                        threat_labels = None
                    )
                self._max_count -= 1
    def _truncate(self, tokens):
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]

        return tokens

    @overrides
    def text_to_instance(self, comment_text:str, target:float = None, 
                threat_labels: np.array =None)->Instance:
        tokenized_comment = self._tokenizer.tokenize(comment_text)
        if not tokenized_comment:
            tokenized_comment = [Token('__NULL__')]

        if self._max_sequence_length is not None:
            tokenized_comment= self._truncate(tokenized_comment)
        
        # create text field from tokens
        fields = {'text' :TextField(tokenized_comment, self._token_indexers)}
        
        # get extra features
        dict_features = extract_features(comment_text)
        # adding float features
        fields['extras'] = TensorField(tensor = np.array([value for key,value in dict_features.items()]))

        if target is not None:
            target = np.array(float(target)).astype('double')
            # adding float target
            fields['target'] = TensorField(target)
        
        if threat_labels is not None:
            threat_labels = TensorField(tensor=np.array(threat_labels))
            # adding threat labels
            fields['threat_labels'] = threat_labels
        
        return Instance(fields)

if __name__ == '__main__':
    reader = ToxicDataReader()
    instances = reader.read('./data/train.csv')
    vocab = Vocabulary.from_instances(instances)
