import tqdm
import json
import rasterio
from rasterio.windows import Window
import itertools
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from glob import glob
import pandas as pd
import re
import ast
from multiprocessing import Pool
import matplotlib.pyplot as plt
from datetime import datetime

from argparse import ArgumentParser

import torch

from utils.inference_helpers import *
from utils.segmentation_model import *

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

class ModelApplicator():
    
    def __init__(self, 
                 ptl_model, 
                 raster_path: str, 
                 output_dir: str, 
                 output_name: str,
                 tile_size: int,
                 scale_factor: int,
                 inference_bands: list, 
                 norm_means: list, 
                 norm_stds: list,
                 nodata_value = 255,
                 output_dtype = rasterio.uint8,
                 cuda_device: int = 0,
                 silent: bool = False,
                 in_memory: bool = False
                 ):
        
        self.ptl_model = ptl_model
        self.raster_path = raster_path
        self.output_dir = output_dir
        self.output_name = output_name
        self.prediction_path = None
        self.tile_size = tile_size
        self.scale_factor = scale_factor
        self.inference_bands = inference_bands
        self.norm_means = norm_means
        self.norm_stds = norm_stds
        self.nodata_value = nodata_value
        self.output_dtype = output_dtype
        self.cuda_device = cuda_device
        self.silent = silent
        self.in_memory = in_memory
        
        # Get raster properties
        with rasterio.open(self.raster_path) as src:
            self.source_rows = src.height
            self.source_cols = src.width
            self.source_rect = Rectangle(0, 0, src.height, src.width)
            
            # Read the raster if in memory
            if self.in_memory:
                self.source_array = src.read()
                self.source_array = self.source_array.astype('int16')
            else:
                self.source_array = None
            
        # Get the offset needed
        self.start_offset = self.__get_offset()

        # Number 'tile_size' subsets to perform inference over
        self.inference_positions_rows = ceil(self.source_rows / (self.tile_size / self.scale_factor))
        self.inference_positions_cols = ceil(self.source_cols / (self.tile_size / self.scale_factor))
        
        # Create the tile manager objects
        self.tile_managers = self.__create_tile_dataset()
    
        return None
    
    def predict(self):
        
        # Disable any gradient tracking
        with torch.no_grad():
        
            # Generate predictions over each of the inference targets
            tiled_predictions = self.__get_predictions_over_tiles()
            
            # Assemble the prediction tiles into a single raster
            image = np.bmat(tiled_predictions)
            tiled_predictions = None
            
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
                
                # Define the origin of the tile
                tile_origin_row = (cur_row * (self.tile_size // self.scale_factor)) - self.start_offset
                tile_origin_col = (cur_col * (self.tile_size // self.scale_factor)) - self.start_offset
                          
                # Create a new tile manager object
                new_manager = TileManager(
                    origin_row = tile_origin_row,
                    origin_col = tile_origin_col,
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
        if self.silent:
            t = range(self.inference_positions_rows)
        else:
            t = tqdm.tqdm(range(self.inference_positions_rows), position=0, leave=True)
            t.set_description("Prediction progress")
        for cur_row in t:
            
            # Store the outputs from each column
            output_columns = []
        
            # Loop over the columns of inference positions
            for cur_col in range(0, self.inference_positions_cols): 
                
                # Get the current tile                
                current_tile = self.tile_managers[count]
                                
                # Select the bands for inference and convert to a torch tensor
                if self.in_memory:
                    
                    # Get the window coodinates
                    ulh_coords = current_tile.window_rect.ulh()
                    lrh_coords = current_tile.window_rect.lrh()
                    
                    # Read in the raster array
                    input_array = torch.Tensor(self.source_array[:, ulh_coords[0]:lrh_coords[0], ulh_coords[1]:lrh_coords[1]]).cuda(self.cuda_device)

                # Do windowed reading if not using memory
                else:
                    with rasterio.open(self.raster_path) as src:
                        
                        # Get the window coodinates
                        ulh_coords = current_tile.model_rect.ulh()
                        lrh_coords = current_tile.model_rect.lrh()
                        
                        # print(current_tile.model_rect)
                        # print(current_tile.window_rect)
                        read_window = Window(ulh_coords[1], ulh_coords[0], self.tile_size, self.tile_size)
                        input_array = torch.Tensor(src.read(window = read_window, boundless=True)).cuda(self.cuda_device)
                        
                # Get the prediction
                # print("input_array", input_array.shape)
                prediction = current_tile.get_prediction(
                    ptl_model = self.ptl_model,
                    raster_array = input_array, 
                    inference_bands = self.inference_bands,
                    norm_means = self.norm_means, 
                    norm_stds = self.norm_stds,
                    device = self.cuda_device, 
                    needs_padding = self.in_memory
                    )
                
                # Crop the prediction as needed
                cropped = self.__clip_to_scale_factor(prediction)
                
                # Append the column predictions
                output_columns.append(cropped.numpy().astype('int8'))
                
                # Increment the counter and progress bar
                count += 1
                
                # Clean up memory
                input_array = None
                prediction = None
                cropped = None
                
                # break
                
            # Append the column predictions to the row predictions
            outputs_rows.append(output_columns)
            
            # break
            
        return outputs_rows
    
    def __normalize (self, v):
        """https://stackoverflow.com/questions/68791508/min-max-normalization-of-a-tensor-in-pytorch"""
        v_min = v.min()
        v_max = v.max()
        v_p = (v - v_min)/(v_max - v_min)*(1 - 0) + 0
        return v_p
        
    
    def __clip_to_scale_factor(self, input_tensor):
        
        # Check if a valid scaling factor was specified. 
        if self.scale_factor not in [1,2,4,8,16,32]:
            raise ValueError("Scaling factor in applicator is not valid -- must be 1,2,4,8")
        
        # Apply the scaling factor
        if self.scale_factor == 1:
            return input_tensor
        elif self.scale_factor == 2:
            return self.__interior_clip(input_tensor)
        elif self.scale_factor == 4:
            temp = self.__interior_clip(input_tensor)
            temp = self.__interior_clip(temp)
            return temp
        elif self.scale_factor == 8:
            temp = self.__interior_clip(input_tensor)
            temp = self.__interior_clip(temp)
            temp = self.__interior_clip(temp)
            return temp
        elif self.scale_factor == 16:
            temp = self.__interior_clip(input_tensor)
            temp = self.__interior_clip(temp)
            temp = self.__interior_clip(temp)
            temp = self.__interior_clip(temp)
            return temp
        elif self.scale_factor == 32:
            temp = self.__interior_clip(input_tensor)
            temp = self.__interior_clip(temp)
            temp = self.__interior_clip(temp)
            temp = self.__interior_clip(temp)
            temp = self.__interior_clip(temp)
            return temp
        
        return input_tensor
        
    def __interior_clip(self, input_tensor):
        width = input_tensor.shape[0]
        height = input_tensor.shape[1]
        clip_size = width // 2
        return input_tensor[clip_size:clip_size*2, clip_size:clip_size*2]
        
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
                    
        return None
    
    def __get_offset (self):
        '''
        Gets the offset needed to start the inference process
        '''
        # Check if the specified scale is valid
        if self.scale_factor not in [1,2,4,8,16,32]:
            raise ValueError('Valid scale factor not entered, must be "1","2","4","8","16","32"')
            
        # Get the offset for the tile size specified
        offsets = [self.tile_size//factor for factor in [2,4,8,16,32]]
        offset_values = [sum(offsets[0:x:1]) for x in range(0, len(offsets)+1)]
                
        # Return the correct scale factor
        if self.scale_factor == 1:
            return offset_values[0]
        elif self.scale_factor == 2:
            return offset_values[1]
        elif self.scale_factor == 4:
            return offset_values[2]
        elif self.scale_factor == 8:
            return offset_values[3]
        elif self.scale_factor == 16:
            return offset_values[4]
        elif self.scale_factor == 32:
            return offset_values[5]
        
def run_applicator(args):
    '''
    "encoder": ,
    "decoder": ,
    "model_path": model_path,
    "time_series_dir": '/media/john/Expansion/coca_composites',
    "output_dir": '/media/john/Expansion/coca_classifications',
    "chunk_size": chunk_size,
    "scale_factor": scale_factor,
    "year": year,
    "out_name": "outputs_" + str(year) + "_factor_" + str(scale_factor) + "_chunk_" + str(chunk_size),
    "device": device,
    "bands": bands,
    "silent": True,
    "in_memory": False
    '''
    
    configuration = {   
        "encoder": args.encoder,
        "decoder": args.decoder,
        "learning_rate": 0.001,
        "loss_name": 'jaccard',
        "max_epochs": 50,
        'batch_size': 2
        }
    net = SegmentationModel.load_from_checkpoint(args.model_path, **configuration)
    net.eval()
    net.freeze()
    net.cuda(args.device)
    
    # Format the input name
    input_name = args.time_series_dir + "/composite_" + str(args.year) + ".tif"
    
    # Instantiate the geotiff applicator
    applicator = ModelApplicator(
        ptl_model = net, 
        raster_path = input_name, 
        output_dir = args.output_dir, 
        output_name = args.out_name,
        tile_size = args.chunk_size,
        scale_factor = args.scale_factor,
        inference_bands = args.bands,
        norm_means = MEANS, 
        norm_stds = STDS,
        cuda_device = args.device,
        silent = args.silent,
        in_memory = args.in_memory,
        )
          
    # Run the processing
    print("Processing -- " + str(args.year))
    applicator.predict()
    applicator = None
    net = None
    torch.cuda.empty_cache()
    print("Completed -- " + str(args.year))
    
    return None

def str_to_bool(in_str):
    if str(in_str) == "None":
        return None
    elif str(in_str) == "True":
        return True
    elif str(in_str) == "False":
        return False
    else:
        raise ValueError("str_to_bool -- value for bool should be a string with value 'True' or 'False' (check ya' argparse).")

if __name__ == "__main__":
    
    # Instantiate the parser
    parser = ArgumentParser(add_help=False)
    
    ### Add the model/PTL specific args
    dflt_mdl_pth = None
    parser.add_argument('--encoder', type=str, default = "resnet")
    parser.add_argument('--decoder', type=str, default = "Unet")
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--time_series_dir', type=str, default = None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--chunk_size', type=int, default = None)
    parser.add_argument('--year', type=int, default = None)
    parser.add_argument('--log_dir', type=str, default = None)
    parser.add_argument('--scale_factor', type=int, default = None)
    parser.add_argument('--out_name', type=str, default=True)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--bands', type=json.loads, default='[0,1,2]')
    parser.add_argument('--silent', type=str_to_bool, default='True')
    parser.add_argument('--in_memory', type=str_to_bool, default='False')

    # Parse the args
    args = parser.parse_args()
    
    # # Manually set the args for debugging
    # tmp_chp_dir = "/home/john/datasets/coca_data_2022/resnest_cross_val/timm-mobilenetv3_large_100-Unet-jaccard-0.001-128-128-False-Experiment2-fold=0/version_0/checkpoints/"
    # tmp_model_path = tmp_chp_dir + '/' + 'timm-mobilenetv3_large_100-Unet-jaccard-0.001-128-128-False-Experiment2-fold=0-epoch=13-val_logger_loss=0.29.ckpt'
    # args.encoder = "timm-mobilenetv3_large_100"
    # args.decoder = "Unet"
    # args.model_path = tmp_model_path
    # args.time_series_dir = "/media/john/Expansion/coca_composites_small"
    # args.output_dir =  '/media/john/Expansion/coca_classifications'
    # args.chunk_size = 128
    # args.scale_factor = 2
    # args.year = 2019
    # args.out_name =  "test_" + str(args.year ) + "_factor_" + str(args.scale_factor) + "_chunk_" + str(args.chunk_size)
    # args.device = 0
    # # args.bands = str([0,1,2,3,4,5,6])
    # args.silent = 'False'
    # args.in_memory = 'False'
    
    # startTime = datetime.now()
    run_applicator(args)
    # print("\nInference Time", datetime.now() - startTime)


