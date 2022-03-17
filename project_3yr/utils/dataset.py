import torch
import os
import random as non_np_random
import pandas as pd
import re
import ast
from math import floor
from math import ceil
from glob import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import multiprocessing as mp
import numpy 
import elasticdeform
import elasticdeform.torch as etorch
from numpy import random
# from augmentations import load_augmentations
# from .augmentations import NullTransform
# from augmentations import AddGaussianNoise
# from augmentations import RandomMasking
# from .augmentations import MaskPixels
# from augmentations import AddGaussianNoise
# from augmentations import MaskPixels

def load_train_test_val(seed, train_data_dir, test_data_dir, norm_stats_path, transforms=None):
    '''
    Use a directory containing a folder of features and labels,
    this data loader instantiates 3 datasets: a training set, a validation set,
    and a development. This is done by assumimg each examhple a as name ofllowing the 
    naming sheme prodcuced by the gdal_retile.py script -- namely, it assumed that
    each file has the following naming scheme: "partitionName_i_j"

    The partition name corresponds to the spatial partition from which the example
    originated. 

    Parameters
    ----------
    seed : int
        Used for deterministic shuffling of the training, testing, validation sets
    
    data_dir: str
        XXXX

    num_channels: int
        XXXX

    Returns
    -------
    Train, Test, Validate datasets.

    '''
    # Get the statistics for normalization
    means = clean_string_list(pd.read_csv(norm_stats_path).means[0])
    stds = clean_string_list(pd.read_csv(norm_stats_path).stds[0])
    
    # Glob all of the tensor files in the directory
    train_tensors = glob(train_data_dir + "/*.pt")
    test_tensors = glob(test_data_dir + "/*.pt")
    
    # Get a list of the unique file names in each directory
    partition_names = []
    for file in train_tensors:
        file_name = file.split('/')[-1].split('_')[0]
        partition_names.append(file_name)
    partition_names.sort()
    partition_names = list(set(partition_names))
    
    # Randomly shuffle the list
    non_np_random.Random(seed).shuffle(partition_names)
    
    # Split the list 
    train_index_stop = floor(len(partition_names) * 0.8) 
    
    # Get the list of partitions for the training, testing, and validation dataset
    train_partitions = partition_names[0:train_index_stop]
    test_partitions = partition_names[train_index_stop:]
    
    # Load in the dataset
    train_examples_paths = string_to_string_match(train_tensors, train_partitions)
    test_examples_paths = string_to_string_match(test_tensors, test_partitions)
    
    # Supply the list to the data loader object which needs to process them.
    train_dataset = CocaDataset(train_examples_paths, means, stds, transforms, False)
    test_dataset = CocaDataset(test_examples_paths, means, stds, None, False)
    
    return train_dataset, test_dataset

def load_train_test_val_3yr(seed, data_dir, norm_stats_path, transforms=None):
    '''
    Use a directory containing a folder of features and labels,
    this data loader instantiates 3 datasets: a training set, a validation set,
    and a development. This is done by assumimg each examhple a as name ofllowing the 
    naming sheme prodcuced by the gdal_retile.py script -- namely, it assumed that
    each file has the following naming scheme: "partitionName_i_j"

    The partition name corresponds to the spatial partition from which the example
    originated. 

    Parameters
    ----------
    seed : int
        Used for deterministic shuffling of the training, testing, validation sets
    
    data_dir: str
        XXXX

    num_channels: int
        XXXX

    Returns
    -------
    Train, Test, Validate datasets.

    '''
    # Get the statistics for normalization
    means = clean_string_list(pd.read_csv(norm_stats_path).means[0])
    stds = clean_string_list(pd.read_csv(norm_stats_path).stds[0])
    
    # Glob all of the tensor files in the directory
    all_files = glob(data_dir + "/*.pt")
    
    # Get a list of the unique file names in each directory
    partition_names = []
    for file in all_files:
        file_name = file.split('/')[-1].split('_')[0]
        partition_names.append(file_name)
    partition_names.sort()
    partition_names = list(set(partition_names))
    
    # Randomly shuffle the list
    non_np_random.Random(seed).shuffle(partition_names)
    
    # Split the list 
    train_index_stop = floor(len(partition_names) * 0.8) 
    # test_index_stop = train_index_stop + floor(len(partition_names) * 0.1) 
    
    # Get the list of partitions for the training, testing, and validation dataset
    train_partitions = partition_names[0:train_index_stop]
    test_partitions = partition_names[train_index_stop:]
    
    # print('')
    # print(train_partitions)
    # print('')
    # print(test_partitions)
    # print('')
    # print(val_partitions)

    # Load in the dataset
    train_examples_paths = string_to_string_match(all_files, train_partitions)
    test_examples_paths = string_to_string_match(all_files, test_partitions)
    
    # Supply the list to the data loader object which needs to process them.
    train_dataset = CocaDataset_3yr(train_examples_paths, means, stds, None, True)
    test_dataset = CocaDataset_3yr(test_examples_paths, means, stds, None)
    
    return train_dataset, test_dataset

def string_to_string_match (files, patterns):
    '''
    For each filename in the list of files, the filename is checked to see if it
    contains any of the patterns in the patterns list. If so, the filename is included
    amongst that are returned. 

    Parameters
    ----------
    files : files
        A list of file names (strings).
    patterns : patterns
        A list of patterns (strings).

    Returns
    -------
    results : list
        a list of file names which contained a substring that matched one
        of the input patterns.

    '''
    results = []
    for file in files:
        for pattern in patterns:
            if pattern in file:
                results.append(file)
    return results

class CocaDataset_3yr(torch.utils.data.Dataset):
    
    def __init__(self, paths, means, stds, augmentations=None, use_deform=False):
        '''
        Args:
            root_dir (string): Directory with all of the images
            transform (calllable, optional): Optional transforms that can be 
                applied to the images. 
        '''
        # Initalize the class attributes
        # Initalize the class attributes
        self.paths = paths #glob(directory + "/*.pt")
        self.means = means
        self.stds = stds
        self.paths = paths #glob(directory + "/*.pt")
        self.augmentations = augmentations
        self.use_deform = use_deform
        
        # Create the normalization layer
        mean_vals = torch.cat([means[0:6], means[0:6], means[0:6], torch.Tensor([means[-1]])])
        stds_vals = torch.cat([stds[0:6], stds[0:6], stds[0:6], torch.Tensor([stds[-1]])])
        self.normalization = transforms.Normalize(mean=mean_vals, std=stds_vals, inplace=True)
        
        return None

    def __getitem__(self, idx):
        '''
        Args:
            idx (int): Index of tensor to retrieve
        
        Returns:
            torch.tensor: DL Tensor [row, col, band]
        
        '''          
        # Read from disk or memory
        data = torch.load(self.paths[idx])
        
        # # Apply the random augmentation
        # if self.augmentations is not None: 
        #     data = self.augmentations(data)
        
        # Split the features off of the label
        features = data[:-1,:,:].float()
        label = data[-1,:,:].long()
        
        # Apply the normalization
        features = self.normalization(features)
        
        # Apply the elastic deformation with 20% chance
        if self.use_deform and random.randint(1,5) == 5:
            features, label = apply_elastic_deform(features, label)
        
        return features, label 
        
    def __len__(self):
        return len(self.paths)

class CocaDataset(torch.utils.data.Dataset):
    
    def __init__(self, paths, means, stds, augmentations=None, use_deform=False):
        '''
        Args:
            root_dir (string): Directory with all of the images
            transform (calllable, optional): Optional transforms that can be 
                applied to the images. 
        '''
        # Initalize the class attributes
        self.paths = paths #glob(directory + "/*.pt")
        self.means = means
        self.stds = stds
        self.normalization = transforms.Normalize(mean=self.means, std=self.stds, inplace=True)
        self.augmentations = augmentations
        self.use_deform = use_deform
        
        return None

    def __getitem__(self, idx):
        '''
        Args:
            idx (int): Index of tensor to retrieve
        
        Returns:
            torch.tensor: DL Tensor [row, col, band]
        
        '''          
        # Read from disk or memory
        data = torch.load(self.paths[idx])
        
        # Split the features off of the label
        features = data[:-1,:,:].float()
        label = data[-1,:,:].long()
        
                # Apply the random augmentation
        if self.augmentations is not None: 
            features, label = self.augmentations((features, label))
        
        # Apply the normalization
        features = self.normalization(features)
        
        # # Apply the elastic deformation with 20% chance
        if self.use_deform and random.randint(1,5) == 5:
            features, label = apply_elastic_deform(features, label)
        
        return features, label 
        
    def __len__(self):
        return len(self.paths)

def apply_elastic_deform(x, y):
    ''' 
    Apply an elastic deformation to an image
    '''
    # Get the displacement vector
    displacement_val = np.random.randn(2, 3, 3) * random.randint(5, 15)
        
    # construct PyTorch input and top gradient
    displacement = torch.tensor(displacement_val, requires_grad=False)
    
    # print(etorch.deform_grid(x[0,:,:], displacement, axis = (0,1), mode='mirror', order=0, prefilter=False))
    
    # the deform_grid function is similar to the plain Python equivalent,
    # but it accepts and returns PyTorch Tensors
    x_deformed = torch.stack([etorch.deform_grid(x[i,:,:], displacement, mode='mirror', order=0, prefilter=False) for i in list(range(0,7))],0)
    y_deformed = etorch.deform_grid(y, displacement, mode='mirror', order=0, prefilter=False)

    return x_deformed, y_deformed

def clean_string_list(input_string):
    '''
    Does formatting to convert a string representation of a list into a list.
    Returns the output as a tensor. 
    '''
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

def plot_tensor(in_tensor):
    '''
    Plot the a given tensor for a single 3 channel input
    '''
    permuted = np.moveaxis(in_tensor.cpu().numpy(), (0,1,2), (2,0,1))
    normed = color_norm(permuted)
    plt.imshow(normed)
    plt.axis('off')
    plt.show()
    return None

def color_norm(array):
    """
    Normalizes numpy arrays into scale 0.0 - 1.0.
    """
    array_min = array.min()
    array_max = array.max()
    return ((array - array_min) / (array_max - array_min))

if __name__ == "__main__":
    
    train_data_dir = "/media/john/linux_ssd/coca_tensors_128_train"
    test_data_dir = "/media/john/linux_ssd/coca_tensors_128_test"
    stats_dir = "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv"
    load_train_test_val(234123432, train_data_dir, test_data_dir, stats_dir)
    
    # # Define a path to the dataset
    # data_dir = glob("/media/john/linux_ssd/coca_tensors_128/*.pt")
    
    # # Get the means and the standard deviations
    # means = clean_string_list(pd.read_csv("/home/john/datasets/coca_data_2022/csvs/norm_stats.csv").means[0])
    # stds = clean_string_list(pd.read_csv("/home/john/datasets/coca_data_2022/csvs/norm_stats.csv").stds[0])
    
    # # Load in the dataset
    # train_dataset = CocaDataset(data_dir, means, stds, augmentations=None)
    
    # # Batch size
    # batch_size = 1
    
    # # Define the DataLoader
    # training = DataLoader(train_dataset, 
    #                       batch_size = batch_size, 
    #                       shuffle = True,
    #                       num_workers = 1,
    #                       drop_last = True, 
    #                       persistent_workers = False,
    #                       pin_memory = False
    #                       )    
    
    # # Load a test example
    # for x, y in training:
    #     x = x.squeeze(0)
    #     y = y.squeeze(0)
    #     break
    
    # # Apply teh deformation
    # # x_deformed, y_deformed = apply_elastic_deform(x, y)
    
    # # Apply noise
    # aug = load_augmentations()
    # x_noise, y_noise = aug((x, y))
    
    # # Format the image for plotting
    # image_vis = color_norm(torch.Tensor(x)[[5,3,2],:,:].clamp(-3.5, 3.5)).permute(1,2,0).numpy()
    # # image_vis_def = color_norm(x_noise[[5,3,2],:,:].clamp(-3.5, 3.5)).permute(1,2,0).numpy()
    # image_vis_noise = color_norm(x_noise[[5,3,2],:,:].clamp(-3.5, 3.5)).permute(1,2,0).numpy()
    
    # label = y.numpy()
    # label_noise = y_noise.numpy()
    
    # # label_def = y_deformed.numpy()
    
    # # Original
    # plt.imshow(image_vis, interpolation='none')
    # plt.show()    
    # plt.imshow(label, interpolation='none')
    # plt.show()
    
    # # With augmentations
    # plt.imshow(image_vis_noise, interpolation='none')
    # plt.show()
    # plt.imshow(label_noise, interpolation='none')
    # plt.show()
    
    # With deformations + augmentations


    
    
    
    
    
    
    
    
    