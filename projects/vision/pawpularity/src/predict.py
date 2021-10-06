from pytorch_lightning.accelerators import accelerator
from src.model import Trainer
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch

from src.data import PawpularityDModule

def predict_dataloader(model, dataloader):
    predictions = []
    model.eval()
    for batch in dataloader:
        predictions.append(model(batch).detach().cpu().numpy())
    
    return np.vstack(predictions)

def predict_with_fold(fold, data_module):
    # loop through all the folds
    # predict test data from each fold
    # report accuracy from out of fold data
    # get mean accuracy from training data.
    # that's the best guess for test data
    checkpoint= "./checkpoints/best-checkpoint-fold={}.ckpt".format(fold)
    model = Trainer.load_from_checkpoint(checkpoint_path=checkpoint)
    trainer = pl.Trainer(gpus=0)
    
    trainer.test(ckpt_path=checkpoint, model=model, datamodule=data_module)
    
    results = trainer.model.predictions
    return results


def predict(num_folds = 5, data_module=None):
    assert data_module is not None
    
    results = []
    for fold_ in range(5):
        res = predict_with_fold(fold_, dm)
        res = res.squeeze(0).squeeze(-1)
        results.append(res)
    
    results = torch.stack(results, dim=-1)
    return results.mean(dim=-1)
    

if __name__ == '__main__':

    train = pd.read_csv('./data/train_5folds.csv')
    test = pd.read_csv('./data/test.csv')
    num_folds = 5

    #test = test.sample(frac=0.5)
    dm = PawpularityDModule(train, 0, test)
    # need datamodule just for test data loader
    
    results = predict(num_folds =num_folds, data_module=dm)
    test['score'] = results.numpy()
    
    test.to_csv('submission.csv',index=False)