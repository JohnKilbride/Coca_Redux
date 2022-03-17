import glob
import os
import string
import random
import torch
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import re
import ast
import shutil
import torchvision.transforms as transforms
from multiprocessing import Pool
import os

def loop_over_example (pool_args):
    '''
    Function takes in an image, assumed to be produced by gdal_retile.py
    and formats it into a normalized example. The input is assumed to be a landsat
    satellite timeseries with 6 channels. The 6 channel time series is converted
    into a 3 channel time series of the Tasseled Cap indicies. A label is then 
    appended to the time series and it is exported to the targer directory
    
    This function is designed to be used with pool processes, as such the function
    parameters should be supplied in a list

    Parameters
    ----------
    pool_args : list
        List where index 0 contains a string with the raster file path and index
        1 contains a string with the output directory.

    Returns
    -------
    None.

    '''
    # Unpack the pool args
    file = pool_args[0]
    output_dir = pool_args[1]

    # Extract the file name
    file_name = file.split('/')[-1].split('.')[0]
    
    # Read in the subset
    raster = gdal.Open(file)
    image_data = np.array(raster.ReadAsArray())
    
    # Split the features and the label
    spectral = image_data[0:78,:,:].reshape((13,6,image_data.shape[1], image_data.shape[2]))
    label = image_data[78:89,:,:]
    dem = np.expand_dims(image_data[-1,:,:], axis=0)
     
    # Loop over each opf the years
    for year in range(2,13):
        
        # Get the spectral features and label
        x = spectral[[year],:,:,:].reshape(6, image_data.shape[1], image_data.shape[1])
        y = np.expand_dims(label[year-2,:,:], axis=0)
        output = np.concatenate((x, dem, y), axis=0)
        
        # Format the data for output
        output_example = torch.Tensor(output).short()
        
        # If 2008 log everything
        if year == 2: 
            torch.save(output_example, output_dir + '/' + file_name + "_" + str(year + 2006) + "_all.pt")
        
        # Otherwise, only log stuff with coca
        elif y.max() == 2:
            
            # Save the data as torch tensors
            torch.save(output_example, output_dir + '/' + file_name + "_" + str(year + 2006) + "_coca.pt")
            
        # elif random.randint(1,5) == 5:
            
        #     # Save the data as torch tensors
        #     torch.save(output_example, output_dir + '/' + file_name + "_" + str(year + 2006) + "_random.pt")
        
    return None

if __name__ == "__main__":
    
    # Define the directory with the different tiles
    tile_dir = "/media/john/linux_ssd/coca_tensors_128_train"

    # Define the tile size
    tile_size = 128

    # Get the subdirectories
    tile_sub_dirs = glob.glob(tile_dir + "/*.pt")
    
    
    
    
    
    
    
    