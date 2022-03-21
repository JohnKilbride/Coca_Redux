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
import elasticdeform
import elasticdeform.torch as etorch
from tqdm import tqdm

def apply_elastic_deform(x, y):
    ''' 
    Apply an elastic deformation to an image
    '''
    # Get the displacement vector
    displacement_val = np.random.randn(2, 3, 3) * 3
        
    # construct PyTorch input and top gradient
    displacement = torch.tensor(displacement_val, requires_grad=False)

    # the deform_grid function is similar to the plain Python equivalent,
    # but it accepts and returns PyTorch Tensors
    x_deformed = torch.stack([etorch.deform_grid(x[i,:,:], displacement, mode='mirror', order=0, prefilter=False) for i in list(range(0,7))],0)
    y_deformed = etorch.deform_grid(y, displacement, mode='mirror', order=0, prefilter=False)

    return x_deformed, y_deformed

if __name__ == "__main__":
    
    # Define the directory with the different tiles
    tile_dir = "/media/john/linux_ssd/coca_tensors_128_test"
    output_dir = "/media/john/linux_ssd/coca_tensors_128_test"

    # Define the tile size
    tile_size = 128

    # Get the subdirectories
    pt_files = glob.glob(tile_dir + "/*.pt")
    
    # Make the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over the file paths
    for i, file in enumerate(tqdm(pt_files)):
        
        # print(i)
        
        # Read from disk or memory
        data = torch.load(file)
        
        # Split the features off of the label
        features = data[:-1,:,:].float()
        label = data[-1,:,:].long()
        
        # Apply the elastic deformation with 20% chance
        features, label = apply_elastic_deform(features, label)
        
        # Recombine
        output = torch.cat((features, label.unsqueeze(0)), dim=0).long()
        
        # Create the output file name
        output_file_name = file.split('/')[-1].split('.')[0] + "_deformed.pt"
        output_file = output_dir + "/" + output_file_name
        
        # Create the outupt name for the deformated data
        torch.save(output, output_file)
        
    
    
    
    