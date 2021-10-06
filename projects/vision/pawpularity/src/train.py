from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.model import Trainer
from src.data import PawpularityDModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger


import pandas as pd
import yaml
Config = yaml.load(open('./config.yaml'))

import logging
logging.basicConfig(level=logging.INFO)

neptune_logger = NeptuneLogger(
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwYWY0OTQ4MS03MGY4LTRhNjUtOTFlZC0zZjVjMjlmZGQxNjQifQ==",
    project_name="puneetgirdhar.in/pawpular",
    experiment_name="default",  # Optional,
    params={"max_epochs": 10},  # Optional,
    tags=["pytorch-lightning", "mlp"],  # Optional,
)
def training(train, test, fold=0):
    model = Trainer()

    data_module = PawpularityDModule(train, fold, test)
    
    # stop the training early
    early_stopping_callback = EarlyStopping(monitor='RMSE', mode='min', patience=Config['patience_earlystop'])
    
    # store model checkpoints
    checkpoint_callback = ModelCheckpoint(
      dirpath="checkpoints",
      filename=f"best-checkpoint-fold={fold}",
      verbose=True,
      monitor="val_rmse",
      mode="min"
    )
    # define trainer
    trainer = pl.Trainer(
      gpus = 2,
      accelerator='ddp',
      checkpoint_callback=True,
      callbacks=[early_stopping_callback,checkpoint_callback],
      max_epochs = Config['epochs'],
      precision = Config['precision'],
      progress_bar_refresh_rate=1, 
      num_sanity_val_steps=1 if Config['debug'] else 0,
      stochastic_weight_avg = True,
      logger=neptune_logger
    )
    
    # fit
    trainer.fit(model, data_module) 
    
    
if  __name__ == '__main__':
    
    train = pd.read_csv('./data/train_5folds.csv')
    test = pd.read_csv('./data/test.csv')
    
    for fold_ in range(5):
      logging.info(f"Training for Fold {fold_}")
      training(train, test, fold_)