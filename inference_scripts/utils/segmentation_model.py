import tqdm
import rasterio
import itertools
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from glob import glob
import pandas as pd
import re
import ast

import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses

class SegmentationModel(pl.LightningModule):

    def __init__(self, encoder, decoder, loss_name, learning_rate, max_epochs, batch_size, **kwargs):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.val_step = 0

        # Configure the model architecture
        if decoder == "Unet":
            self.net = smp.Unet(
                encoder_name = encoder,
                in_channels = 7,
                encoder_weights = None,
                classes = 3
                )
        elif decoder == "UnetPlusPlus":
            self.net = smp.UnetPlusPlus(
                encoder_name = encoder,
                in_channels = 7,
                encoder_weights = None,
                classes = 3,
                decoder_attention_type = "scse"
                )
        elif decoder == "DeepLabV3Plus":
            self.net = smp.DeepLabV3Plus(
                encoder_name = encoder,
                in_channels = 7,
                encoder_weights = None,
                classes = 3
                )
        
        # Configure the loss function
        if self.loss_name == 'focal':
            self.loss_funct = losses.FocalLoss(
                mode = "multiclass",
                ignore_index = None
                )
        elif self.loss_name == 'jaccard':
            self.loss_funct = losses.JaccardLoss(
                mode = "multiclass",
                from_logits = True
                )
        
        return None
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_funct(y_hat, y)     
        return {'loss': loss}
    
    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("avg_loss", avg_loss, self.current_epoch)        
        return None

    def validation_step(self, batch, batch_idx):
        return None
    
    def validation_epoch_end(self, outputs):
        return None

    def configure_optimizers(self):
        opt =  torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [45], gamma=0.1)
        sched_dict = {'scheduler': sched, 'interval': 'epoch'}
        return [opt], [sched_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument('--encoder', type=str, default='ResNet18')
        parser.add_argument('--decoder', type=str, default='Unet')
        parser.add_argument('--loss_name', type=str, default='jaccard')
        return parent_parser