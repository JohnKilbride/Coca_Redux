import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from torch import nn
from math import ceil
# from glob import glob
from argparse import ArgumentParser

import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.strategies import DDPSpawnStrategy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torchmetrics.functional as metrics

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses

from utils.dataset import load_train_test_val
from utils.augmentations import load_augmentations
from utils.classic_unet import UnetClassic

class SegmentationModel(pl.LightningModule):

    def __init__(self, encoder, decoder, loss_name, learning_rate, max_epochs, batch_size, decay, augs, **kwargs):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.decay = decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.augs = augs
        
        # Log the hyper parameters to tensorboard
        self.save_hyperparameters()
        
        # Set up some counters
        self.val_epoch_logger = 0
        self.train_epoch_logger = 0
        self.train_step_counter = 0

        # Configure the model architecture
        if self.decoder == "Unet":
            self.net = smp.Unet(
                encoder_name = self.encoder,
                in_channels = 7,
                encoder_weights = None,
                classes = 3,
                encoder_depth=5
                )
        elif self.decoder == "UnetClassic":
            self.net = UnetClassic(7, 3, True)
        elif decoder == "DeepLabV3Plus":
            self.net = smp.DeepLabV3Plus(
                encoder_name = self.encoder,
                in_channels = 7,
                encoder_weights = None,
                classes = 3
                )
        elif self.decoder == 'PAN':
            self.net = smp.PAN(
                encoder_name = self.encoder, 
                encoder_weights = None,
                decoder_channels = 32, 
                in_channels = 7, 
                classes = 3, 
                activation = None, 
                upsampling = 1
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
        
        # UNpack the batch
        x, y = batch
        
        # Crop label if needed for PAN
        if self.decoder == "PAN":
            y = self.pan_interior_clip(y)
        
        # Compute the validation Loss
        y_hat = self.net(x)
        
        loss = self.loss_funct(y_hat, y)

        # Compute the loss metrics
        y_hat_max = torch.argmax(y_hat, dim=1)
        accuracy = metrics.accuracy(y_hat_max, y, num_classes=3, average='macro')
        f1 = metrics.f1_score(y_hat_max, y, num_classes=3, mdmc_average='samplewise', average='macro')
        jaccard = metrics.jaccard_index(y_hat_max, y, num_classes=3)
        
        # If this is the first step, log the actual data, otherwise, log a dummy variable
        # This is to allow the validation_epoch_end function to log the segmentation images
        # while avoiding excess memory usage from storing all results
        if self.train_epoch_logger == 0:
            outputs = {
                "x":x.detach().cpu(),
                "y": y.detach().cpu(),
                "y_hat": y_hat.detach().cpu(),
                "loss": loss,
                "accuracy": accuracy,
                "f1": f1, 
                "jaccard": jaccard
                }
        else:
            outputs = {
                "x": torch.Tensor(0).detach().cpu(),
                "y": torch.Tensor(0).detach().cpu(),
                "y_hat": torch.Tensor(0).detach().cpu(),
                "loss": loss,
                "accuracy": accuracy,
                "f1": f1, 
                "jaccard": jaccard
                }
                
        # Increment the validation step counter
        self.train_epoch_logger += 1
        self.train_step_counter += 1        
        self.logger.experiment.add_scalar("step_loss", loss, self.train_step_counter)              

        return outputs
    
    def training_epoch_end(self,outputs):
        
        # Get the averages of the loss metrics
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['f1'] for x in outputs]).mean()
        avg_jaccard = torch.stack([x['jaccard'] for x in outputs]).mean()
        
        # Log the metrics
        self.logger.experiment.add_scalar("loss", avg_loss, self.current_epoch)      
        self.logger.experiment.add_scalar("accuracy", avg_acc, self.current_epoch)      
        self.logger.experiment.add_scalar("f1", avg_f1, self.current_epoch)      
        self.logger.experiment.add_scalar("jaccard", avg_jaccard, self.current_epoch)    
        
        # Make the grids
        image = [x['x'] for x in outputs][0].clamp(-2.5, 2.5)
        label = [x['y'] for x in outputs][0].unsqueeze(1)
        logits = [x['y_hat'] for x in outputs][0]
        prediction = torch.argmax(logits, dim=1).unsqueeze(1)
        
        grid_1 = make_grid(image[:, [5,3,2], :, :].float(), nrow=self.batch_size, normalize=True)
        grid_2 = make_grid(label.float(), nrow=self.batch_size, normalize=True)
        grid_3 = make_grid(logits.float(), nrow=self.batch_size, normalize=True)
        grid_4 = make_grid(prediction.float(), nrow=self.batch_size, normalize=True)

        # Log the segmentation results
        self.logger.experiment.add_image('input', grid_1, self.current_epoch)
        self.logger.experiment.add_image('label', grid_2, self.current_epoch)
        self.logger.experiment.add_image('logits', grid_3, self.current_epoch)
        self.logger.experiment.add_image('segmentation', grid_4, self.current_epoch)
        
        # Reset the validation step counter
        self.train_epoch_logger = 0
                
        return None

    def validation_step(self, batch, batch_idx):
        
        # Unpack the batch
        x, y = batch
        
        # Crop label if needed for PAN
        if self.decoder == "PAN":
            y = self.pan_interior_clip(y)
        
        # Compute the forward pass
        y_hat = self.net(x)
        
        # Compute the validation Loss
        loss = self.loss_funct(y_hat, y)   
        
        # Get the predicted label
        y_hat_max = torch.argmax(y_hat, dim=1)

        # Get the interior of the label & predictions
        if self.decoder != 'PAN':
            y = self.interior_clip(y)
            y_hat_max = self.interior_clip(y_hat_max)
        
        # Compute the loss metrics
        val_accuracy = metrics.accuracy(y_hat_max, y, num_classes=3, average='macro')
        val_f1 = metrics.f1_score(y_hat_max, y, num_classes=3, mdmc_average='samplewise', average='macro')
        val_jaccard = metrics.jaccard_index(y_hat_max, y, num_classes=3)
        
        # If this is the first step, log the actual data, otherwise, log a dummy variable
        # This is to allow the validation_epoch_end function to log the segmentation images
        # while avoiding excess memory usage from storing all results
        if self.val_epoch_logger == 0:
            outputs = {
                "x":x.detach().cpu(),
                "y": y.detach().cpu(),
                "y_hat": y_hat.detach().cpu(),
                "val_loss": loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1, 
                "val_jaccard": val_jaccard
                }
        else:
            outputs = {
                "x": torch.Tensor(0).detach().cpu(),
                "y": torch.Tensor(0).detach().cpu(),
                "y_hat": torch.Tensor(0).detach().cpu(),
                "val_loss": loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1, 
                "val_jaccard": val_jaccard
                }
                
        # Increment the validation step counter
        self.val_epoch_logger += 1
        
        # Log the validation loss for early stopping
        self.log("val_logger_loss", loss)
        
        return outputs
    
    def validation_epoch_end(self, outputs):
        
        # Get the averages of the loss metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_jaccard = torch.stack([x['val_jaccard'] for x in outputs]).mean()
        
        # Log the metrics
        self.logger.experiment.add_scalar("avg_val_loss", avg_loss, self.current_epoch)      
        self.logger.experiment.add_scalar("avg_val_accuracy", avg_acc, self.current_epoch)      
        self.logger.experiment.add_scalar("avg_val_f1", avg_f1, self.current_epoch)      
        self.logger.experiment.add_scalar("avg_val_jaccard", avg_jaccard, self.current_epoch)    
        
        # Make the grids
        image = [x['x'] for x in outputs][0].clamp(-2.5, 2.5)
        label = [x['y'] for x in outputs][0].unsqueeze(1)
        logits = [x['y_hat'] for x in outputs][0]
        prediction = torch.argmax(logits, dim=1).unsqueeze(1)
        
        grid_1 = make_grid(image[:, [5,3,2], :, :].float(), nrow=self.batch_size, normalize=True)
        grid_2 = make_grid(label.float(), nrow=self.batch_size, normalize=True)
        grid_3 = make_grid(logits.float(), nrow=self.batch_size, normalize=True)
        grid_4 = make_grid(prediction.float(), nrow=self.batch_size, normalize=True)

        # Log the segmentation results
        self.logger.experiment.add_image('val_input', grid_1, self.current_epoch)
        self.logger.experiment.add_image('val_label', grid_2, self.current_epoch)
        self.logger.experiment.add_image('val_logits', grid_3, self.current_epoch)
        self.logger.experiment.add_image('val_segmentation', grid_4, self.current_epoch)
        
        # Reset the validation step counter
        self.val_epoch_logger = 0
        
        return None

    def configure_optimizers(self):
        opt =  torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [60,80], gamma=0.1)
        sched_dict = {'scheduler': sched, 'interval': 'epoch'}
        return [opt], [sched_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument('--encoder', type=str, default='ResNet18')
        parser.add_argument('--decoder', type=str, default='Unet')
        parser.add_argument('--loss_name', type=str, default='jaccard')
        return parent_parser
    
    def interior_clip(self, input_tensor):
        width = input_tensor.shape[-2]
        height = input_tensor.shape[-1]
        clip_size = width // 2
        return input_tensor[:,clip_size:clip_size*2, clip_size:clip_size*2]
    
    def interior_clip(self, input_tensor):
        width = input_tensor.shape[-2]
        height = input_tensor.shape[-1]
        clip_size = width // 2
        return input_tensor[:,clip_size:clip_size*2, clip_size:clip_size*2]
    
    def pan_interior_clip(self, input_tensor):
        # Get the dimensions of the tensor
        width = input_tensor.shape[-2] // 2
        height = input_tensor.shape[-1] // 2
        
        # Clip the tensor
        width_c = (width // 4) 
        height_c = (height // 4)
        input_tensor = input_tensor[:,width-width_c:width+width_c, height-height_c:height+height_c]
        
        return input_tensor.contiguous()

def cli_main (args):
    
    # ------------
    # data
    # ------------
    
    # Define the transforms
    args.augs = load_augmentations()
    
    # Load in the dataset
    train_dataset, test_dataset = load_train_test_val(args.seed, args.data_dir, args.norm_stats_csv, args.augs)
    
    # Define the DataLoader
    train_loader = DataLoader(train_dataset, 
                              batch_size = args.batch_size, 
                              shuffle = True,
                              num_workers = args.num_workers,
                              persistent_workers = False,
                              pin_memory = True,
                              )    
    test_loader = DataLoader(test_dataset, 
                             batch_size = args.batch_size, 
                             shuffle = False,
                             num_workers = args.num_workers,
                             persistent_workers = False,
                             pin_memory = True,
                             )      
    
    # ------------
    # logging
    # ------------  
    # Add the weights and biases logger
    model_name = '{decoder}-{loss_name}-{encoder}-{lr}-{decay}-{batch}-{resolution}-{suffix}'
    args.model_name = model_name.format(
        decoder = args.decoder,
        encoder = args.encoder,
        loss_name = args.loss_name,
        batch = args.batch_size,
        lr = args.learning_rate,
        decay = args.decay,
        resolution = args.resolution,
        suffix = args.suffix
        )
    logger = TensorBoardLogger(
        name = args.model_name,
        save_dir = args.log_dir,
        )       

    # ------------
    # Load Model
    # ------------
    # Seed everything
    pl.seed_everything(args.seed, workers=True)
    
    # Construct the model
    dict_args = vars(args)
    net = SegmentationModel(**dict_args)
    
    # ------------- 
    # Running the model
    # -------------
    callback = EarlyStopping(
        monitor = "val_logger_loss", 
        min_delta=0.001, 
        patience=30, 
        verbose=False, 
        mode="min"
        )
    
    # Model logging 
    ModelCheckpoint(
        monitor='val_logger_loss',
        filename=args.model_name+'-{epoch:02d}-{val_logger_loss:.2f}',
        save_top_k=1
        )
    
    # Set the device to GPU zero if 0 is specified for devices
    if args.gpus == 0:
        args.gpus = [0]
    elif args.gpus == 1:
        args.gpus = [1]
   
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger = logger,
        callbacks = [callback],
        strategy=DDPSpawnStrategy(find_unused_parameters=False),
        log_every_n_steps = 1,
        # gradient_clip_val = 0.5
        # deterministic=True,
        # auto_select_gpus=True
        )
    trainer.fit(net, train_loader, test_loader)
  
    print('\n-------------------------------------------')
    print('            Run Completed                   ')
    print('-------------------------------------------\n')
    
    return None
            
if __name__ == '__main__':

    # ------------
    # args
    # ------------
    
    # Instantiate the parser
    parser = ArgumentParser(add_help=False)
    
    ### Add the model/PTL specific args
    parser = SegmentationModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--num_workers', type=int, default=1)
    
    ### Dataset stuff
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--norm_stats_csv', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    
    ### Training parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--decay', type=float, default=None)

    # Parse the args
    args = parser.parse_args()
   
    # Run the processing
    cli_main(args)
    