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


def clean_string_list(input_string):
    
    # Remove new line characters
    input_string = re.sub('\n', '', input_string)
        
    # Rmove any repeated white spaces
    input_string = re.sub("\s\s+", " ", input_string)

    # Replace the remaining white spaces with commas
    input_string = re.sub(' ', ',', input_string)
    
    # Clean any weird issues
    out_string = ""
    for i, current_character in enumerate(input_string):
        
        # Get the previous character
        if i == 0:
            previous_char = None
        else:
            previous_char = input_string[i-1]
        
        # Get the current character
        if previous_char == '[' and current_character == ',':
            pass
        else:
            out_string += current_character
    
    return torch.Tensor(ast.literal_eval(out_string))

# GLOBAL VARIABLES
MEANS = clean_string_list(pd.read_csv("/home/john/datasets/coca_data_2022/csvs/norm_stats.csv").means[0])
STDS = clean_string_list(pd.read_csv("/home/john/datasets/coca_data_2022/csvs/norm_stats.csv").stds[0])

class SimpleAutoEncoder(pl.LightningModule):

    def __init__(self, encoder, learning_rate, max_epochs, batch_size, **kwargs):
        super().__init__()
        
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.last_logging = 0

        # Load in the ResNeSt encoder
        self.net = smp.Unet(
            encoder_name = encoder,
            in_channels = 7,
            encoder_weights = None,
            classes = 3
            )
        
        self.loss_funct = losses.FocalLoss(
            mode = "multiclass",
            ignore_index = None
            )
        # self.loss_funct = losses.JaccardLoss(
        #     mode = "multiclass",
        #     from_logits = True
        #     )
        
        
        return None
        
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        
        # Compute the training loss
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_funct(y_hat, y)     
        
        # Do the logging
        self.log("loss", loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        # Compute the validation Loss
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_funct(y_hat, y)    

        # then compute metrics with required reduction (see metric docs)
        y_hat_max = torch.argmax(y_hat, dim=1)
        
        self.log("val_loss", loss, on_epoch=True)
        self.log("accuracy", metrics.accuracy(y_hat_max, y, num_classes=3), on_epoch=True)
        self.log("recall", metrics.recall(y_hat_max, y, num_classes=3, mdmc_average='samplewise'), on_epoch=True)
        self.log("precision", metrics.precision(y_hat_max, y, num_classes=3, mdmc_average='samplewise'), on_epoch=True)       
        self.log("auroc", metrics.auroc(y_hat, y, num_classes=3), on_epoch=True)
        self.log("f1_score", metrics.f1_score(y_hat, y, num_classes=3, mdmc_average='samplewise'), on_epoch=True)
        self.log("jaccard_index", metrics.jaccard_index(y_hat, y, num_classes=3), on_epoch=True)
        
        # Periodically do the plotting
        if self.last_logging == self.current_epoch:

            # Make the grids
            grid_1 = make_grid(x[:, [5,3,2], :, :].float(), nrow=self.batch_size, normalize=True)
            grid_2 = make_grid(y.unsqueeze(1).float(), nrow=self.batch_size, normalize=True)
            grid_3 = make_grid(torch.argmax(y_hat, dim=1).unsqueeze(1).float(), nrow=self.batch_size, normalize=True)
    
            # Join the different images
            joined = torch.cat([grid_1, grid_2, grid_3], dim=1)
            
            # Log the images as wandb Image
            self.logger.log_image('output', [joined])
            self.last_logging += 1
        
        return None
    
    def training_epoch_end(self, batch):
        self.log('epoch', self.current_epoch, on_step=False, on_epoch=True)
        return None
    
    # def validation_epoch_end(self, batch):
    #     return None

    def configure_optimizers(self):
        opt =  torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, [10, 15], gamma=0.1)
        sched_dict = {'scheduler': sched, 'interval': 'epoch'}
        return [opt], [sched_dict]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AutoEncoder")
        parser.add_argument('--encoder', type=str, default='ResNet18')
        return parent_parser

class ModelApplicator():
    
    def __init__(self, 
                 ptl_model, 
                 raster_path: str, 
                 output_dir: str, 
                 output_name: str,
                 tile_size: int,
                 inference_bands: list, 
                 norm_means: list, 
                 norm_stds: list,
                 nodata_value = 255,
                 output_dtype = rasterio.uint8,
                 cuda_device: int = 0
                 ):
        
        self.ptl_model = ptl_model
        self.raster_path = raster_path
        self.output_dir = output_dir
        self.output_name = output_name
        self.prediction_path = None
        self.tile_size = tile_size
        self.inference_bands = inference_bands
        self.norm_means = norm_means
        self.norm_stds = norm_stds
        self.nodata_value = nodata_value
        self.output_dtype = output_dtype
        self.cuda_device = cuda_device

        # Get the the size of the raster
        with rasterio.open(raster_path) as src:
            self.source_rows = src.height
            self.source_cols = src.width
            self.source_rect = Rectangle(0, 0, src.height, src.width)
            self.source_array = src.read()
            
            # # Open the raster
            # data_array = []
            # for i in range(1,11):
            #     band = src.read(i)
            #     # band = np.where(band == -9999, 0, band)
            #     # band = np.where(band == -10000, 0, band)
            #     # band = np.where(band == -32768, 0, band)
            #     data_array.append(band)
    
            # # Glue the corrected bands together
            # self.source_array = np.stack(data_array, axis=0)

        # Number 'tile_size' subsets to perform inference over
        self.inference_positions_rows = int(ceil(self.source_rows / (self.tile_size / 2)))
        self.inference_positions_cols = int(ceil(self.source_cols / (self.tile_size / 2)))
        
        # Create the tile manager objects
        self.tile_managers = self.__create_tile_dataset()
    
        return None
    
    def predict(self):
        
        # Generate predictions over each of the inference targets
        # These are the 
        tiled_predictions = self.__get_predictions_over_tiles()
        
        # Assemble the prediction tiles into a single raster
        image = np.bmat(tiled_predictions)
        
        # Trim the prediction raster to the original extent
        trimmed = image[0:self.source_rows, 0:self.source_cols]
        
        # Write out the prediction
        self.__write_prediction_raster(trimmed)

        return None
    
    def __create_tile_dataset(self):
        
        # List which will hold the tile manager objects
        tile_manager_list = []
        
        # Loop over the rows of inference positions
        for cur_row in range(0, self.inference_positions_rows):
        
            # Loop over the columns of inference positions
            for cur_col in range(0, self.inference_positions_cols):
                
                # Get the origin of the current inference target (interior of tile)
                subset_origin_row = cur_row * int(self.tile_size/2)
                subset_origin_col = cur_col * int(self.tile_size/2)
                                
                # Create a new tile manager object
                new_manager = TileManager(
                    origin_row = subset_origin_row,
                    origin_col = subset_origin_col,
                    source_rect = self.source_rect,
                    tile_size = self.tile_size
                    )
                tile_manager_list.append(new_manager)
                
        return tile_manager_list
            
    def __get_predictions_over_tiles(self):
        
        # Store the outputs from each row
        outputs_rows = []
        
        # Loop over the rows of inference positions
        count = 0
        t = tqdm.tqdm(range(self.inference_positions_rows), position=0, leave=True)
        t.set_description("Prediction progress")
        for cur_row in t:
            
            # Store the outputs from each column
            output_columns = []
        
            # Loop over the columns of inference positions
            for cur_col in range(0, self.inference_positions_cols): 
                
                # Get the current tile
                current_tile = self.tile_managers[count]
                
                # Get the prediction
                prediction = current_tile.get_prediction(
                    ptl_model = self.ptl_model,
                    raster_array = self.source_array, 
                    inference_bands = self.inference_bands,
                    norm_means = self.norm_means, 
                    norm_stds = self.norm_stds,
                    device = self.cuda_device
                    )
                
                # Append the column predictions
                output_columns.append(prediction)
                
                # Increment the counter and progress bar
                count += 1
                
            # Append the column predictions to the row predictions
            outputs_rows.append(output_columns)
            
        return outputs_rows    
        
    def __write_prediction_raster(self, output_raster):
        
        # Register GDAL format drivers and configuration options with a
        # context manager.
        with rasterio.Env():
            
            # Open the source raster
            with rasterio.open(self.raster_path) as src:
                
                # Write an array as a raster band to a new 8-bit file. For
                # the new file's profile, we start with the profile of the source
                profile = src.profile
            
                # And then change the band count to 1, set the
                # dtype to uint8, and specify LZW compression.
                profile.update(
                    dtype = self.output_dtype,
                    count = 1,
                    compress = 'lzw',
                    nodata = self.nodata_value
                    )
            
                # Set the prediction path
                self.prediction_path = self.output_dir + "/" + self.output_name + ".tif"
            
            
            
                with rasterio.open(self.prediction_path, 'w', **profile) as dst:
                    dst.write(output_raster.astype(self.output_dtype), 1)
                    # dst.write_mask(src_mask)
                    
        return None
    
class TileManager():
    '''
    The Tile Manager deals with with creating a prediction over a region. 
    '''
    
    def __init__(self, origin_row, origin_col, source_rect, tile_size):
        
        self.origin_row = origin_row
        self.origin_col = origin_col
        self.source_rect = source_rect
        self.tile_size = tile_size
        
        # Construct the inference tile
        self.model_rect = Rectangle(origin_row - (tile_size/4), origin_col - (tile_size/4), 
                                     origin_row + ((tile_size/4) * 3), origin_col  + ((tile_size/4) * 3))
        
        # Get read window by intersecting the source and the model_input
        self.window_rect = source_rect.intersection(self.model_rect)
        
        # Get the padding amounts
        self.pad_amounts = self.__get_padding_amounts()
        
        return None
    
    def get_prediction(self, ptl_model, raster_array, inference_bands, 
                       norm_means, norm_stds, device):
                
        # Read in the windowed subset of the raster as a tensor
        array = self.__read_raster_data(raster_array, inference_bands)

        # Normalize the input raster and convert to a tensor
        normalized = self.__normalize_tensor(array, norm_means, norm_stds)
        
        # Pad the tensor
        padding_layer = torch.nn.ZeroPad2d(self.pad_amounts)
        model_input = padding_layer(normalized.unsqueeze(0))
        
        # Perform inference
        logits = ptl_model(model_input.cuda(device))
        
        # Take the max logit value for each class
        predictions = torch.argmax(logits.squeeze(0), dim=0)
        
        # Combine the two labels
        combined = predictions.cpu().numpy()
        
        # Clip out the prediction target (center of the scene)
        step = int(self.tile_size/4)
        output = combined[step:int(step*3), step:int(step*3)]    
        
        return output
    
    def __get_padding_amounts(self):
        
        # Get the corners
        model_ulh = self.model_rect.ulh()
        model_lrh = self.model_rect.lrh()
        source_ulh = self.source_rect.ulh()
        source_lrh = self.source_rect.lrh()
        
        # Get the amount of left padding
        if model_ulh[1] < source_ulh[1]:
            left_pad = int(source_ulh[1] - model_ulh[1] )
        else:
            left_pad = 0
        
        # Get the amount of right padding
        if model_lrh[1] > source_lrh[1]:
            right_pad = int(model_lrh[1] - source_lrh[1])
        else:
            right_pad = 0
        
        # Get the amount of upper padding
        if model_ulh[0] < source_ulh[0]:
            upper_pad = int(source_ulh[0] - model_ulh[0])
        else:
            upper_pad = 0
        
        # Get the amount of lower padding
        if model_lrh[0] > source_lrh[0]:
             lower_pad = int(model_lrh[0] - source_lrh[0])
        else:
            lower_pad = 0
            
        return (left_pad, right_pad, upper_pad, lower_pad)
    
    def __read_raster_data(self, raster_array, inference_bands):
        
        # Get the window readed parameters
        ulh_coords = self.window_rect.ulh()
        lrh_coords = self.window_rect.lrh()
            
        # Select the bands for inference and convert to a torch tensor
        array = torch.Tensor(raster_array[inference_bands, ulh_coords[0]:lrh_coords[0], ulh_coords[1]:lrh_coords[1]]).cpu()
    
        return array

    def __normalize_tensor(self, input_tensor, means, stds):
        
        # Create a tensor with the means and standard deviations
        norm_transform = transforms.Normalize(means.squeeze(0), stds.squeeze(0))
        
        # Apply the normalization
        norm = norm_transform(input_tensor)
                
        return norm

class Rectangle():
    """
    Based on:
        
    Citation: https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles
    Author: Oleh Prypin
    
    Thanks amigo!
    """
    def __init__(self, row1, col1, row2, col2):
        if row1>row2 or col1>col2:
            raise ValueError("Coordinates are invalid")
        self.row1, self.col1, self.row2, self.col2 = row1, col1, row2, col2
    
    def intersection(self, other):
        a, b = self, other
        row1 = max(min(a.row1, a.row2), min(b.row1, b.row2))
        col1 = max(min(a.col1, a.col2), min(b.col1, b.col2))
        row2 = min(max(a.row1, a.row2), max(b.row1, b.row2))
        col2 = min(max(a.col1, a.col2), max(b.col1, b.col2))
        if row1<row2 and col1<col2:
            return type(self)(row1, col1, row2, col2)
    
    def ulh (self):
        return (int(self.row1), int(self.col1))
    
    def lrh (self):
        return (int(self.row2), int(self.col2))

    def __iter__(self):
        yield self.row1
        yield self.col1
        yield self.row2
        yield self.col2

    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self)==tuple(other)
    
    def __ne__(self, other):
        return not (self==other)

    def __repr__(self):
        return type(self).__name__+repr(tuple(self))

    def __pairwise(self, iterable):
        # https://docs.python.org/dev/library/itertools.html#recipes
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
    
if __name__ == "__main__":
    
    # Define the parameters
    time_series_dir = '/home/john/datasets/coca_data_2022/inference_composites'
    output_dir = '/home/john/datasets/coca_data_2022/outputs'
    output_path_base = '/home/john/datasets/coca_data_2022/outputs'
    chunk_size = 512  
    
    # Define the bands to be used during inference
    bands = list(range(0,7))

    # Load the pytorch model
    print('Loading model...')
    chp_path = '/home/john/datasets/coca_data_2022/model_logs/Unet-FocalLoss-resnet18-0.001_32/version_None/checkpoints/epoch=10-step=27070.ckpt'
    configuration = {   
        "encoder": 'resnet18', 
        "learning_rate": 0.001,
        "max_epochs": 50,
        'batch_size': 16
        }
    net = SimpleAutoEncoder.load_from_checkpoint(chp_path, **configuration)
    net.eval()
    net.freeze()
    net.cuda(0)
    
    for year in range(1984, 2019):
        
        if year in [1984, 1989, 1994, 1999, 2004, 2009, 2014, 2019]:
        
            # Format the input name
            input_name = "/home/john/datasets/coca_data_2022/inference_composites/composite_" + str(year) + ".tif"
            
            # Create the output name
            out_name = "outputs_" + str(year)
            
            # Instantiate the geotiff applicator
            applicator = ModelApplicator(ptl_model = net, 
                                         raster_path = input_name, 
                                         output_dir = output_dir, 
                                         output_name = out_name,
                                         tile_size = chunk_size,
                                         inference_bands = bands,
                                         norm_means = MEANS, 
                                         norm_stds = STDS,
                                         cuda_device = 0
                                         )
            
            # Run the processing
            applicator.predict()
            
