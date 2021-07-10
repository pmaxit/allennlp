local batch_size = 32;
local dropout=0.3;
local lstm_hidden_size=200;

{"dataset_reader":{
    "type": "toxic",
    "max_sequence_length": 5000,
    "max_count": 1000,
    "token_indexers":{
        "tokens":{
            "type": "single_id",
            "lowercase_tokens": true
        },
        "characters":{
            "type": "characters",
            "min_padding_length": 3
        },
    },
},
"train_data_path": "./data/train.csv",
//"valid_data_path": "./data/valid.csv",
//"test_data_path" : "./data/test.csv",
"vocabulary":{
    "type": "from_files",
    "directory": "/home/jovyan/project/classification/toxic_classification/tmp/vocab.tar.gz"
},
"model":{
    "type": "toxic",
    "text_field_embedder":{
        "token_embedders":{
            "tokens":{
                "type":"embedding",
                #"pretrained_file": "https://github.com/kmr0877/IMDB-Sentiment-Classification-CBOW-Model/blob/master/glove.6B.50d.txt.gz?raw=true",
                "embedding_dim": 50,
                "trainable": true
            },
            "characters":{
                "type": "character_encoding",
                "embedding":{
                    "embedding_dim": 16
                },
                "encoder":{
                    "type": "cnn-highway",
                    "activation": "relu",
                    "embedding_dim": 16,
                    "filters": [
                        [1, 16],
                        [2, 16],
                        [3, 32],
                        [4, 32],
                        [5, 64],
                        [6, 64],
                        [7, 128],
                    ],
                    "num_highway": 2,
                    "projection_dim": 512,
                    "projection_location": "after_highway",
                    "do_layer_norm": true
                },
            },
        },
    },
    "encoder":{
        "type": "bag_of_embeddings",
        "embedding_dim": 512 + 50
    },
    "classifier_feedforward":{
        "input_dim": 512+ 50,
        "num_layers": 2,
        "hidden_dims": [200, 6],
        "activations": ["tanh", "linear"],
        "dropout": [0.2, 0.0],
    },
},

"data_loader":{
    "batch_size": batch_size,
},

"trainer":{
    "num_epochs": 40,
    "patience": 3,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "num_gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "adadelta",
        "lr": 1.0,
        "rho": 0.95
    },
},

}