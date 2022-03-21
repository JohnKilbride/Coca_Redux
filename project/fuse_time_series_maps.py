import tqdm
import os
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
from scipy import stats
from osgeo import gdal

from argparse import ArgumentParser

import torch 

from utils.inference_helpers import Rectangle
from utils.inference_helpers import TileManager

class TimeSeriesMerger():
    
    def __init__(self, 
                 raster_dir: str, 
                 output_dir: str, 
                 output_name: str,
                 start_year: int,
                 end_year: int,
                 nodata_value = 255,
                 output_dtype = rasterio.uint8,
                 silent: bool = False,
                 ):
        
        self.raster_dir = raster_dir
        self.output_dir = output_dir
        self.output_name = output_name
        self.start_year = start_year
        self.end_year = end_year
        self.nodata_value = nodata_value
        self.output_dtype = output_dtype
        self.silent = silent
        
        # If the output dir does not exist, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Get the names of the raster paths that will be used during inference
        self.raster_paths = glob(self.raster_dir + "/*.tif")
        if len(self.raster_paths) == 0:
            raise ValueError("Error: raster dir specified in TimeSeriesMerger has no tif files in it.")
    
        return None
    
    def get_mode_of_layers (self):
        
        # Loop over the various years
        for year in range(self.start_year, self.end_year+1):
            
            # Get the relevant raster paths
            cur_year_paths = []
            for path in self.raster_paths:
                if str(year) in path:
                    cur_year_paths.append(path)
     
            # Generate predictions over each of the inference targets
            list_of_label_arrays = []
            for raster_path in cur_year_paths:
                src = gdal.Open(raster_path)
                input_array = np.array(src.ReadAsArray())
                list_of_label_arrays.append(input_array.astype('int8'))
        
            # Here, stack the various inputs and take the mode
            stacked_labels = np.stack(list_of_label_arrays)
            stacked_mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_labels)
        
            # Write out the prediction
            self.__write_raster(stacked_mode, year)
            
            # break

        return None
        
    def __write_raster(self, output_raster, year):
        
        # Register GDAL format drivers and configuration options with a
        # context manager.
        with rasterio.Env():
            
            # Open the source raster
            with rasterio.open(self.raster_paths[0]) as src:
                
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
                output_path = self.output_dir + "/" + self.output_name + "_" + str(year) + ".tif"
            
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(output_raster.astype(self.output_dtype), 1)
                    
        return None

def str_to_bool(in_str):
    if str(in_str) == "True":
        return True
    elif str(in_str) == "False":
        return False
    else:
        raise ValueError("str_to_bool -- value for bool should be a string with value 'True' or 'False' (check ya' argparse).")
        
def main(args):
    
    # Instantiate the mode maker object
    dict_args = vars(args)
    merger = TimeSeriesMerger(**dict_args)
    
    # Run the prediction function
    merger.get_mode_of_layers()
    
    return None
    
if __name__ == "__main__":
    
    # Instantiate the parser
    parser = ArgumentParser(add_help=False)
    
    ### Add the model/PTL specific args
    parser.add_argument('--raster_dir', type=str, default = None)
    parser.add_argument('--output_dir', type=str, default = None)
    parser.add_argument('--output_name', type=str, default = None)
    parser.add_argument('--start_year', type=int, default = None)
    parser.add_argument('--end_year', type=int, default = None)
    parser.add_argument('--silent', type=str_to_bool, default='True')

    # Parse the args
    args = parser.parse_args()
    
    # Manually set the args for debugging
    args.raster_dir = "/media/john/Expansion/coca_classifications"
    args.output_dir =  '/media/john/Expansion/coca_cv_mode_maps'
    args.output_name = "coca_mode_test"
    args.start_year = 1984
    args.end_year = 2019
    args.silent = 'False'
    
    # Get the runtime of the script
    startTime = datetime.now()
    main(args)
    print("\nInference Time", datetime.now() - startTime)


