
from itertools import chain
from typing import Iterable, Tuple

import allennlp
from allennlp.models.archival import archive_model
from allennlp.modules import seq2vec_encoders
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
def build_dataset_reader()->DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(model_name= model_name)
    token_indexer = {'tokens': PretrainedTransformerIndexer(model_name=model_name)}
    return TweetReader(tokenizer=tokenizer, 
                       token_indexers= token_indexer, 
                       max_length=max_length,
                       manual_distributed_sharding=True, 
                       manual_multiprocess_sharding=True)

def build_vocab(train_loader, dev_loader)->Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(
        chain(train_loader.iter_instances(), dev_loader.iter_instances())
    )

def build_model(vocab:Vocabulary) ->Model:
    print("Building the model")
    
    embedder = BasicTextFieldEmbedder(
        {'tokens': PretrainedTransformerEmbedder(model_name=model_name)}
    )
    seq2vec_encoder = BertPooler(pretrained_model=model_name, dropout=0.1)
    # adding extra 9 dim vector from metafeatures
    
    feedforward = FeedForward(num_layers=2, input_dim=seq2vec_encoders.get_output_dim() + 9, hidden_dims=[256, 6], activations= [
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
    
    
def build_data_loaders(
    reader,
    train_data_path: str,
    validation_data_path: str
)-> Tuple[DataLoader, DataLoader]:
    train_loader = MultiProcessDataLoader(
        reader, train_data_path, batch_size=8, shuffle=True
    )
    
    dev_loader = MultiProcessDataLoader(reader, validation_data_path, batch_size=8, shuffle=False)
    
    
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
)-> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamaxOptimizer(parameters)  # type: ignore
    # There are a *lot* of other things you could configure with the trainer.  See
    # http://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects for more
    # information.

    trainer = GradientDescentTrainer(
        model=model.cuda(),
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=1,
        optimizer=optimizer,
        validation_metric="+accuracy",
        cuda_device=0
    )
    return trainer

def run_training_loop(serialization_dir: str):
    reader = build_dataset_reader()
    
    train_loader, dev_loader = build_data_loaders(
        reader, "./data/train-v2.csv","./data/val-v2.csv"
    )
    
    vocab = build_vocab(train_loader, dev_loader)
    model = build_model(vocab)
    
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
    
    print("start training")
    trainer.train()
    
    print("finished training")
    print('saving archive')
    vocab.save_to_files('./tmp/vocabulary')
    #archive_model(serialization_dir)

if __name__ == '__main__':
    
    run_training_loop('tmp')