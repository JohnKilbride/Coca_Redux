import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import pytorch_lightning as pl

import torchmetrics.functional as metrics

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as losses

class SegmentationModel(pl.LightningModule):

    def __init__(self, encoder, decoder, loss_name, learning_rate, max_epochs, batch_size, decay, augs, test_partitions=None, **kwargs):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.decay = decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.augs = augs
        self.test_partitions = test_partitions
        
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
                classes = 3
                )
        elif self.decoder == "UnetPlusPlus":
            self.net = smp.UnetPlusPlus(
                encoder_name = self.encoder,
                in_channels = 7,
                encoder_weights = None,
                classes = 3
                )
        else:
            raise ValueError("Invalid decoder name specified.")
            
        # Configure the loss function
        if self.loss_name == "jaccard":
            self.loss_funct = losses.JaccardLoss(
                mode = "multiclass",
                from_logits = True
                )
        elif self.loss_name == "lovasz":
            self.loss_funct = losses.LovaszLoss(
                mode = "multiclass",
                from_logits = True
                )
        else:
            raise ValueError("Invalid loss function name specified.")
            
        return None
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        
        # Unpack the batch
        x, y = batch
        
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
        self.logger.experiment.add_scalar("train_accuracy", avg_acc, self.current_epoch)      
        self.logger.experiment.add_scalar("train_f1", avg_f1, self.current_epoch)      
        self.logger.experiment.add_scalar("train_jaccard", avg_jaccard, self.current_epoch)    
        
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
        
        # Log the partitions if it is the first epoch
        if self.current_epoch == 0 and self.test_partitions is not None:
            self.logger.experiment.add_text("test_folds", str(self.test_partitions))
        else:
            self.logger.experiment.add_text("test_folds", "N/A")
        
        # Reset the validation step counter
        self.train_epoch_logger = 0
                
        return None

    def validation_step(self, batch, batch_idx):
        
        # Unpack the batch
        x, y = batch

        # Compute the forward pass
        y_hat = self.net(x)
        
        # Compute the validation Loss
        loss = self.loss_funct(y_hat, y)   
        
        # Get the predicted label
        y_hat_max = torch.argmax(y_hat, dim=1)

        # Get the interior of the label & predictions
        # Note, this reflects our ultimate inference stratedgy
        y = self.interior_clip(y)
        y_hat_max = self.interior_clip(y_hat_max)
        
        # Compute the loss metrics
        val_accuracy = metrics.accuracy(y_hat_max, y, num_classes=3, average='macro')
        val_accuracy_background = metrics.accuracy(y_hat_max.eq(0).long(), y.eq(0).long(), num_classes=2)
        val_accuracy_pasture = metrics.accuracy(y_hat_max.eq(1).long(), y.eq(1).long(), num_classes=2)
        val_accuracy_coca = metrics.accuracy(y_hat_max.eq(2).long(), y.eq(2).long(), num_classes=2)
        
        val_f1 = metrics.f1_score(y_hat_max, y, num_classes=3, mdmc_average='samplewise', average='macro')
        val_f1_background = metrics.f1_score(y_hat_max.eq(0).long(), y.eq(0).long(), num_classes=2, mdmc_average='samplewise', average='macro')
        val_f1_pasture = metrics.f1_score(y_hat_max.eq(1).long(), y.eq(1).long(), num_classes=2, mdmc_average='samplewise', average='macro')
        val_f1_coca = metrics.f1_score(y_hat_max.eq(2).long(), y.eq(2).long(), num_classes=2, mdmc_average='samplewise', average='macro')
        
        val_jaccard = metrics.jaccard_index(y_hat_max, y, num_classes=3, absent_score=1)
        val_jaccard_background = metrics.jaccard_index(y_hat_max.eq(0).long(), y.eq(0).long(), num_classes=2, absent_score=1)
        val_jaccard_pasture = metrics.jaccard_index(y_hat_max.eq(1).long(), y.eq(1).long(), num_classes=2, absent_score=1)
        val_jaccard_coca = metrics.jaccard_index(y_hat_max.eq(2).long(), y.eq(2).long(), num_classes=2, absent_score=1)
        
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
                "val_accuracy_background": val_accuracy_background,
                "val_accuracy_pasture": val_accuracy_pasture,
                "val_accuracy_coca": val_accuracy_coca,
                "val_f1": val_f1, 
                "val_f1_background": val_f1_background, 
                "val_f1_pasture": val_f1_pasture, 
                "val_f1_coca": val_f1_coca, 
                "val_jaccard": val_jaccard,
                "val_jaccard_background": val_jaccard_background, 
                "val_jaccard_pasture": val_jaccard_pasture, 
                "val_jaccard_coca": val_jaccard_coca, 
                }
        else:
            outputs = {
                "x": torch.Tensor(0).detach().cpu(),
                "y": torch.Tensor(0).detach().cpu(),
                "y_hat": torch.Tensor(0).detach().cpu(),
                "val_loss": loss,
                "val_accuracy": val_accuracy,
                "val_accuracy_background": val_accuracy_background,
                "val_accuracy_pasture": val_accuracy_pasture,
                "val_accuracy_coca": val_accuracy_coca,
                "val_f1": val_f1, 
                "val_f1_background": val_f1_background, 
                "val_f1_pasture": val_f1_pasture, 
                "val_f1_coca": val_f1_coca, 
                "val_jaccard": val_jaccard,
                "val_jaccard_background": val_jaccard_background, 
                "val_jaccard_pasture": val_jaccard_pasture, 
                "val_jaccard_coca": val_jaccard_coca, 
                }
                
        # Increment the validation step counter
        self.val_epoch_logger += 1
        
        # Log the validation loss for early stopping
        self.log("val_logger_loss", loss)
        
        return outputs
    
    def validation_epoch_end(self, outputs):
        
        # Get the averages of the loss metrics
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("avg_val_loss", avg_loss, self.current_epoch) 
        
        # Get the Average classification accuracy
        avg_acc = torch.stack([x['val_jaccard'] for x in outputs]).mean()
        avg_acc_background = torch.stack([x['val_accuracy_background'] for x in outputs]).mean()
        avg_acc_pasture = torch.stack([x['val_accuracy_pasture'] for x in outputs]).mean()
        avg_acc_coca = torch.stack([x['val_accuracy_coca'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("val_accuracy", avg_acc, self.current_epoch)  
        self.logger.experiment.add_scalar("val_accuracy_background", avg_acc_background, self.current_epoch)  
        self.logger.experiment.add_scalar("val_accuracy_pasture", avg_acc_pasture, self.current_epoch)  
        self.logger.experiment.add_scalar("val_accuracy_coca", avg_acc_coca, self.current_epoch)  
        
        # F1 metrics
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_f1_background = torch.stack([x['val_f1_background'] for x in outputs]).mean()
        avg_f1_pasture = torch.stack([x['val_f1_pasture'] for x in outputs]).mean()
        avg_f1_coca = torch.stack([x['val_f1_coca'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("val_f1", avg_f1, self.current_epoch) 
        self.logger.experiment.add_scalar("val_f1_background", avg_f1_background, self.current_epoch) 
        self.logger.experiment.add_scalar("val_f1_pasture", avg_f1_pasture, self.current_epoch) 
        self.logger.experiment.add_scalar("val_f1_coca", avg_f1_coca, self.current_epoch) 
        
        # Jaccard metrics
        avg_jaccard = torch.stack([x['val_jaccard'] for x in outputs]).mean()
        avg_jaccard_background = torch.stack([x['val_jaccard_background'] for x in outputs]).mean()
        avg_jaccard_pasture = torch.stack([x['val_jaccard_pasture'] for x in outputs]).mean()
        avg_jaccard_coca = torch.stack([x['val_jaccard_coca'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("val_jaccard", avg_jaccard, self.current_epoch) 
        self.logger.experiment.add_scalar("val_jaccard_background", avg_jaccard_background, self.current_epoch) 
        self.logger.experiment.add_scalar("val_jaccard_pasture", avg_jaccard_pasture, self.current_epoch) 
        self.logger.experiment.add_scalar("val_jaccard_coca", avg_jaccard_coca, self.current_epoch) 
        
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
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [10,15], gamma=0.5)
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
        clip_size_width = width // 2
        clip_size_height = height // 2
        return input_tensor[:,clip_size_width:clip_size_width*2, clip_size_height:clip_size_height*2]
    