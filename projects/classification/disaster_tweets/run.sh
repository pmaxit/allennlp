#allennlp train dist_baseline.jsonnet --include-package src -s tmp -f 
allennlp predict tmp/model.tar.gz data/test --include-package src --batch-size 4 --predictor disaster --use-dataset-reader --output-file data/output.csv

#ulimit -n 200000

# will train the model and store only the weights
# python -m src.train  

# to predict single instance
# python -m src.predictor

# to train
#allennlp train dist_baseline.jsonnet --include-package src -s tmp -f 

# to predict whole file
# allennlp predict tmp/model.tar.gz data/test --include-package src --batch-size 4 --predictor disaster --use-dataset-reader --output-file data/output.csv

# to evaluate on test dataset and get the score
# allennlp evaluate