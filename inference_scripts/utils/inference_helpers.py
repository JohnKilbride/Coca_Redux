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

class TileManager():
    '''
    The Tile Manager deals with with creating a prediction over a region. 
    '''
    
    def __init__(self, origin_row, origin_col, source_rect, tile_size):
        
        self.origin_row = origin_row
        self.origin_col = origin_col
        self.source_rect = source_rect
        self.tile_size = tile_size
        
        # # Construct the inference tile
        # self.model_rect = Rectangle(origin_row - (tile_size/4), origin_col - (tile_size/4), 
        #                              origin_row + ((tile_size/4) * 3), origin_col  + ((tile_size/4) * 3))
        self.model_rect = Rectangle(origin_row, origin_col, 
                             origin_row + tile_size, origin_col + tile_size)
        
        # Get read window by intersecting the source and the model_input
        self.window_rect = source_rect.intersection(self.model_rect)
        
        # Get the padding amounts
        self.pad_amounts = self.__get_padding_amounts()
        
        return None
    
    def get_prediction(self, ptl_model, raster_array, inference_bands, 
                       norm_means, norm_stds, device, needs_padding):

        # Normalize the input raster and convert to a tensor
        model_input = self.__normalize_tensor(raster_array, norm_means, norm_stds)
        
        # Pad the tensor
        if needs_padding:
            padding_layer = torch.nn.ZeroPad2d(self.pad_amounts)
            model_input = padding_layer(model_input.unsqueeze(0))
        else:
            model_input = model_input.unsqueeze(0)
        
        # Perform inference
        logits = ptl_model(model_input.cuda(device))
        
        # Take the max logit value for each class
        predictions = torch.argmax(logits.squeeze(0), dim=0)
        
        # Combine the two labels
        output = predictions.cpu()

        # Set some stuff to None to prevent leaks
        logits = None
        combined = None
        
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
    
    origin_row = -64
    origin_col = -64
    source_rect = Rectangle(0,0,2000,2000)
    tile_size = 128
    
    z = TileManager(origin_row, origin_col, source_rect, tile_size)
    
    
