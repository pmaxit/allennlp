local batch_size = 128;
local dropout = 0.3;
local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
"dataset_reader":{
    "type": "sharded",
    "base_reader":{
        "type": "tweety",
        "max_length": 5000,
        "token_indexers":{
            "tokens":{
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
        "keyword_indexers":{
            "tokens":{
                "type": "single_id",
                "lowercase_tokens": true
            },
        },
        "tokenizer": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
        },
        "manual_distributed_sharding": true,
        "manual_multiprocess_sharding": true
       }
    },
"train_data_path": "data/processed_train/*.part",
"validation_data_path": "data/processed_train/validation/*.part",
"test_data_path": "data/processed_train/validation/*.part",
"model": {
    "type": "disaster",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          //"gradient_checkpointing": true,
        }
      }
    },
   "keyword_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable":false
        }
      }
    },
    "seq2vec_encoder": {
       "type": "bert_pooler",
       "pretrained_model": transformer_model,
       "dropout": 0.4,
    },
    "keyword_encoder":{
      "type": "bag_of_embeddings",
      "embedding_dim": 100,
    },
    "feedforward":{
      "input_dim": 1024 + 100,
      "num_layers": 2,
      "hidden_dims": [200, 6],
      "activations": ["tanh", "linear"],
      "dropout": [0.4, 0.0],
    },
    "namespace": "tags",
    "num_labels": 2,
    "dropout": 0.3,

    "initializer": {
        "regexes": [
          [
              ".*tag_projection_layer.*weight",
              {
                  "type": "xavier_uniform"
              }
          ],
          [
              ".*tag_projection_layer.*bias",
              {
                  "type": "zero"
              }
          ],
          [
              ".*feedforward.*weight",
              {
                  "type": "xavier_uniform"
              }
          ],
          [
              ".*feedforward.*bias",
              {
                  "type": "zero"
              }
          ],
          [
              ".*weight_ih.*",
              {
                  "type": "xavier_uniform"
              }
          ],
          [
              ".*weight_hh.*",
              {
                  "type": "orthogonal"
              }
          ],
          [
              ".*bias_ih.*",
              {
                  "type": "zero"
              }
          ],
          [
              ".*bias_hh.*",
              {
                  "type": "lstm_hidden_bias"
              }
          ]
        ]
      },
  },


"data_loader": {
    "batch_size": 16,
    "num_workers": 8
    },
"trainer": {
    "num_epochs": 40,
    "validation_metric": "+accuracy",
    "use_amp": true,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 0.01,
      "weight_decay": 0.01,
    },
    "learning_rate_scheduler": {
          "type": "linear_with_warmup",
          "warmup_steps": 100,
        },
    "grad_norm":5.0,
    "patience": 5,
    "num_gradient_accumulation_steps": 4,
    },

"distributed":{
    "cuda_devices":[0,1],
} 

}
