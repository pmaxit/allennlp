local batch_size = 128;
local dropout = 0.3;
local transformer_model = "vinai/bertweet-base";
local transformer_dim = 768;

{
"dataset_reader":{
    "type": "tweety",
    "max_length": 5000,
    "token_indexers":{
        "tokens":{
            "type": "pretrained_transformer",
            "model_name": transformer_model,
        },
    },
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    },
    "manual_distributed_sharding": true,
    "manual_multiprocess_sharding": true,
    
  },
"train_data_path": "data/train_raw.csv",
"validation_data_path": "data/val_raw.csv",
"model": {
    "type": "disaster",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          //"gradient_checkpointing": true,
          "last_layer_only": false
        }
      }
    },
    "seq2vec_encoder":{
      "type": "stacked_bidirectional_lstm",
      "input_size": transformer_dim,
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
    "namespace": "tags",
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
    "num_epochs": 40,
    "validation_metric": "-loss",
    #"use_amp": true,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 0.05,
      "weight_decay": 1e-2,
    },
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        },
    "grad_norm":5.0,
    "cuda_device":0,
    "patience": 10,
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