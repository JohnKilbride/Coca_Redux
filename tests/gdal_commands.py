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

def create_temporary_dir(base_dir):
    '''
    Create the temporary directory.
    '''
    new_dir = base_dir + "/" + randomword(25)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir

def delete_temporary_dir(rm_dir):
    '''
    delete the directory
    '''
    shutil.rmtree(rm_dir)    
    return None

def randomword(length):
    '''
    Generate a random length string
    '''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def compute_tassled_cap(input_array, means, stds):
    '''
    Take a 6xNxM matrixa and compute the tasseled cap trasnformation
    '''
    # Compute the tasseled cap components
    coefs = np.asarray([
        [0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863],
        [-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800],
        [0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572],
        ])
    
    # Get the shape of the input
    shp = input_array.shape
    
    # Collapse the input array into a 2D array
    flattened = np.reshape(input_array, (shp[0], shp[1]*shp[2]))
    
    # Compute the tassled cap values and then reshape to a 3D array
    tc_values = np.reshape(np.dot(coefs, flattened), (3, shp[1], shp[2]))
    
    # Center the tasseled cap values
    normed = normalize_tensor(tc_values, means, stds)
    
    return normed

def normalize_tensor(input_tensor, means, stds):
    '''
    Center the values of the tensor.
    '''
    input_tensor = torch.Tensor(input_tensor)
    means = torch.Tensor(means)
    stds = torch.Tensor(stds)
    norm_transform = transforms.Normalize(mean=means, std=stds, inplace=True)
    normed = norm_transform(input_tensor).clone().detach().cpu().numpy()
    return normed

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

if __name__ == "__main__":
    
    # Define the directory with the different tiles
    tile_dir = "/home/john/datasets/forecasting_disturbance/amazon_tiles"
    
    # Define a path to sub-directories
    for input_file in glob.glob(tile_dir+"/*.tif"):
    
        # Define the output dir
        output_dir = "/home/john/datasets/forecasting_disturbance/tensors"
        
        # Get the means and the standard deviations
        means = clean_string_list(pd.read_csv("/home/john/datasets/forecasting_disturbance/csvs/norm_stats.csv").means[0])
        stds = clean_string_list(pd.read_csv("/home/john/datasets/forecasting_disturbance/csvs/norm_stats.csv").stds[0])
        
        # Tempdir location
        temp_dir_loc = os.path.dirname(input_file)
        
        # Define the tile size
        tile_size = 256
        
        # Define the output dir
        temp_dir = create_temporary_dir(temp_dir_loc)
        
        # Write the outputs to the temporary directory--config GDAL_CACHEMAX
        com1 = "gdal_retile.py " 
        com1 += "-ps " + str(tile_size) + " " + str(tile_size) + " "
        com1 += "-targetDir " + temp_dir + " "
        com1 += input_file 
        os.system(com1)
        
        # Identify the tiles that are too small
        to_remove = []
        tif_files = glob.glob(temp_dir + "/*.tif")
        for file in tif_files:
            raster = gdal.Open(file)
            width = raster.RasterXSize
            height = raster.RasterYSize
            if width != tile_size or height != tile_size:
                to_remove.append(file)
                
        # Remove all of the small tiles
        for file in to_remove:  
            os.remove(file)
            
        # Process each GeoTiff shard into a model example
        files = glob.glob(temp_dir + "/*.tif")
        files.sort()
        for file in files:
            
            # Extract the file name
            file_name = file.split('/')[-1].split('.')[0]
            
            # Read in the subset
            raster = gdal.Open(file)
            image_data = np.array(raster.ReadAsArray())
            
            # Split the features and the label
            features = image_data[0:-1,:,:]
            label = image_data[-1,:,:]
            
            # Reshape the features
            features_reshaped = features.reshape((int(features.shape[0]/6), 6, tile_size, tile_size))
            
            # Loop over each year and compute the Tasseled Cap Components
            # and perform the feature normalization
            components = np.concatenate([compute_tassled_cap(features_reshaped[i,:,:,:], means, stds) for i in range(0, features_reshaped.shape[0])], axis=0)
            
            # Reshape the components into a 4D tensor
            normed_features = np.reshape(components, (int(components.shape[0]/3), 3, tile_size, tile_size))
            
            # Save teh data as torch tensors
            
            torch.save(normed_features, output_dir + "/features/" + file_name + ".pt")
            torch.save(label, output_dir + "/labels/" + file_name + ".pt")
            
        
        # Delete the temporary directory containing the shards
        delete_temporary_dir(temp_dir)
    
    
    
    
    