from typing import List, Dict, Iterable, Optional
import csv
import sys
import re
from allennlp.data.tokenizers import Token

import tqdm
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from nltk.corpus import stopwords
from allennlp.data import Vocabulary
from allennlp.data.batch import Batch
from allennlp.common.util import ensure_list
from overrides import overrides
import numpy as np
import torch
from allennlp.data.fields import TextField, ArrayField
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from collections import namedtuple
from src.utils import *
from nltk.stem import WordNetLemmatizer
import urllib

@DatasetReader.register('tweety')
class TweetReader(DatasetReader):
    """ Readign tweet dataset """
    def __init__(self, max_length:int = None, tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexer = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._clean_text = text_preprocessing
        self._lemmatizer = WordNetLemmatizer()
        
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields['tokens']._token_indexers = self._token_indexer
        
    @overrides
    def _read(self, file_path: str, skip_header: bool = True)->Iterable[Instance]:
        with open(file_path, 'r') as data_file:
            reader = csv.reader(data_file, quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)

            Data = namedtuple("Data", next(reader))
            for data in map(Data._make, self.shard_iterable(reader)):
                text = data.text
                if 'target' not in data._fields:
                    # for test data
                    yield self.text_to_instance(text, create_metafeatures(text))
                else:
                    yield self.text_to_instance(text, create_metafeatures(text), data.target)
    
    @overrides
    def text_to_instance(self,
                         text: str,
                         metafeatures:List[float],
                         target: Optional[int]=None)->Instance:
        
        if self._clean_text is not None:
            text = self._clean_text(text)
        
        if self.max_length is not None:
            text = text[:self.max_length]
        
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text)
        
        fields = {'tokens': text_field, 'meta': ArrayField(metafeatures)}
        
        if target is not None:
            fields['label'] = LabelField(int(target),skip_indexing=True)
            
        return Instance(fields)