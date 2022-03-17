
import os
import string
import random
import torch
from glob import glob
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import re
import ast
import shutil
import torchvision.transforms as transforms
from multiprocessing import Pool
import random


if __name__ == "__main__":
    
    # Define the directory with the different tiles
    tile_dir = "/media/john/linux_ssd/autoencoder_dataset"
    
    # Get the ID of the times that need to be kept
    tile_info_csv = "/home/john/datasets/forecasting_disturbance/csvs/autoencoder_tiles_38460m.csv"
    tile_data = pd.read_csv(tile_info_csv)
    tile_ids = tile_data['grid_id'].tolist()
    
    # Loop over each of the tiles in the folder and delete the ones that are not in the id list
    for i, path in enumerate(glob(tile_dir + "/*.pt")):
        
        # Check if it is in the list
        found = False
        for tile_id in tile_ids:
            if tile_id in path:
                found = True
                break
        
        # Remove files with no match
        if not found:
            print('Removing {}'.format(path))
            os.remove(path)
        
        
    print("Program completed.")
    
    
    
    
    
    
    