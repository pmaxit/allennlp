# Pytorch Lightning Utils
import pytorch_lightning as pl
import timm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer, required 
import timm
from torch.optim import Adam, SGD
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from src.transforms import *
import ttach as tta


import yaml
Config = yaml.load(open('./config.yaml'))

class Trainer(pl.LightningModule):
    
    def __init__(self, model_name = Config['model'],out_features=Config['img_ftr_len'],
                 pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        
        # feature extractor
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes = out_features)
        
        
        
        # final fc layer
        self.finalfc = nn.Sequential(
            nn.Linear(out_features+12, 120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120, 1)
        )
        
        self.criterion = nn.MSELoss()
                
    def forward(self, img, ftrs):
        
        # feature extractor extracts features from the image
        imgouts = self.model(img)
 
        # we combine the meta features with the image features given
        ftrout = torch.cat([imgouts, ftrs], dim=-1)
        
        # we then pass the combined feature into final layer
        output = self.finalfc(ftrout)
    
        return output


    def training_step(self, batch, batch_idx):

        img, ftrs, score = batch
        
        if Config['train_type'] == 'mix_up':
            mixed_x, mixed_z, y_a, y_b, lam = mixup_data(img, ftrs, score, lam=Config['lam']  )
            output = self.forward(mixed_x, mixed_z)
            loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
        else:
            output = self.forward(img, ftrs)
            loss = self.criterion(output, score)
        
        try:
            rmse = mean_squared_error(score.detach().cpu(), output.detach().cpu(), squared=False) 

            self.log("RMSE", rmse, on_step= True, prog_bar=True, logger=True)
            self.log("Train Loss", loss, on_step= True,prog_bar=False, logger=True)
        
        except:
            pass

        return {"loss": loss, "predictions": output.detach(), "labels": score.detach()}

    def training_epoch_end(self, outputs):

        preds = []
        labels = []
        
        for output in outputs:
            
            preds += output['predictions']
            labels += output['labels']

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        train_rmse = mean_squared_error(labels.detach().cpu(), preds.detach().cpu(), squared=False)
        
        self.print(f'Epoch {self.current_epoch}: Training RMSE: {train_rmse:.4f}')
        
        self.log("mean_train_rmse", train_rmse, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        
        img, ftrs, score = batch
        
        with torch.no_grad():
            output = self.forward(img, ftrs)

            loss = self.criterion(output, score)
        
        self.log('val_loss', loss, on_step= True, prog_bar=False, logger=True)
        return {"predictions": output.detach(), "labels": score}
      

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []
        
        for output in outputs:
            preds += output['predictions']
            labels += output['labels']

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        val_rmse = mean_squared_error(labels.detach().cpu(), preds.detach().cpu(), squared=False)
        
        self.print(f'Epoch {self.current_epoch}: Validation RMSE: {val_rmse:.4f}')

        
        self.log("val_rmse", val_rmse, prog_bar=True, logger=True)
        

    def test_step(self, batch, batch_idx):
        img, ftrs = batch
        if Config['tta']:
            predictions = []
            for transform in tta.aliases.d4_transform():
                augmented_image = transform.augment_image(img)
                output = self.forward(augmented_image, ftrs).squeeze(-1).cpu().numpy()
                predictions.append(output)

            output = np.stack(predictions,axis=-1)
            output = torch.from_numpy(output)
        else:
            output = self.forward(img, ftrs).detach()
        
        return {'predictions': output}

    def test_epoch_end(self, outputs) -> None:
        predictions = torch.stack([x['predictions'] for x in outputs])

        self.predictions = predictions
        
        return predictions
        
    def configure_optimizers(self):

        param_optimizer = list(self.model.named_parameters())
        
        # configuring parameters
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": float(Config['weight_decay']),
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    
        # we use adam optimizer with Cosine annealing LR
        optimizer = Adam(optimizer_parameters, lr=float(Config['lr']))
        
        
        scheduler = CosineAnnealingLR(optimizer,
                              T_max=float(Config['T_max']),
                              eta_min=float(Config['min_lr']),
                              last_epoch=-1)

        return dict(
          optimizer=optimizer,
          lr_scheduler=scheduler
        )
