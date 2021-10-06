local batch_size = 128;
local dropout = 0.3;
local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
"dataset_reader":{
    "type": "tweety",
    "max_length": 5000,
    "token_indexers":{
        "elmo":{
            "type": "elmo_characters"
        },
        "tokens2":{
            "type": "single_id",
            "lowercase_tokens": true,
        },

    },
    "tokenizer": {
      "type": "whitespace",
    },
    "manual_distributed_sharding": true,
    "manual_multiprocess_sharding": true,
    
  },
"train_data_path": "data/train-v2.csv",
"validation_data_path": "data/val-v2.csv",
"model": {
    "type": "disaster",
    "text_field_embedder": {
      "token_embedders": {
        "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": true,
            "dropout": 0.3,
            "requires_grad": false
    },
        "tokens2":{
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
	        "trainable": false
        },
      }
    },
    "seq2vec_encoder":{
      "type": "stacked_bidirectional_lstm",
      "input_size": 1024 + 100,
      "hidden_size": 256,
      "num_layers": 4,
      "recurrent_dropout_probability": 0.2

    },
    "feedforward":{
      "input_dim": 9+ 2*256,
      "num_layers": 2,
      "hidden_dims": [200, 6],
      "activations": ["tanh", "linear"],
      "dropout": [0.3, 0.0],
    },
    "namespace": "tokens",
    "num_labels": 2,
    "smoothing": 0.1,
    "dropout": 0.2,
    // "poswise_feedforward": {
    //     "d_model": 100,
    //     "d_ff": 400
    // },
    "regularizer":{
      'regexes':[
        [ ".*",
          {'type': 'l2',
           'alpha': 0.001,
          },
        ],
      ],
    },
  },


"data_loader": {
    "batch_size": 32,
    "num_workers": 8
    },
"trainer": {
    "num_epochs": 20,
    "validation_metric": "+accuracy",
    #"use_amp": true,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 0.001,
      "weight_decay": 1e-2,
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        },
    "grad_norm":1.0,
    //"cuda_device":0,
    "patience": 5,
    "num_gradient_accumulation_steps": 1,
    'callbacks': [
        {
          type: 'tensorboard'
        },
    ],
    },

//   "distributed":{
//     "cuda_devices":[0,1],
// } 

}