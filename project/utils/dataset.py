import torch
import random as non_np_random
import pandas as pd
import re
import ast
from math import floor
from glob import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import numpy 
from numpy import random

def load_simple_train_test (seed, train_data_dir, test_data_dir, norm_stats_path, deformed_data_dir = None, transforms=None):
    '''
    This function produces a spatially stratified training and testing dataset (80/20) split. 
    The splitting is based on the spatial partitions defined in the file names. 
    
    Training examples are gathered from the training_data_dir, and testing examples
    are derived from the test_data_dir. It is assumed that each dir has data for all partitions.
    This allows the degree of overlap in the train/test set to be controlled. 
    
    If deformations are precomputed, a directory can be passed and any examples in the
    training set spatial partitions will be added to the training dataset. 
    
    If transforms are supplied, they will be added to the training dataset.
    
    Returns the training dataset, the testing dataset, and a list with the names
    of the testing partitions. 

    Parameters
    ----------
    seed : intger
        The seed used to shuffle the partitions.
    train_data_dir : str
        A spatially partitioned dataset containing the data formatted for training the model.
    test_data_dir : str
        A spatially partitioned dataset containing the data formatted for model evaluation.
    norm_stats_path : str
        DESCRIPTION.
    deformed_data_dir : str, optional
        A spatially partitioned dataset containing examples with elastic deformations.
    transforms : 
        Transforms applied to the dataset

    Raises
    ------
    ValueError
        If no deformed tensors are detected.

    Returns
    -------
    train_dataset :
        A PyTorch dataset containing the training examples.
    test_dataset :
        A PyTorch dataset containing the test examples.
    test_partitions : list
        A list containing straings with the partition names.

    '''
    # Get the statistics for normalization
    means = clean_string_list(pd.read_csv(norm_stats_path).means[0])
    stds = clean_string_list(pd.read_csv(norm_stats_path).stds[0])
    
    # Glob all of the tensor files in the directory
    train_tensors = glob(train_data_dir + "/*.pt")
    test_tensors = glob(test_data_dir + "/*.pt")
    
    # Get the deformed tensors if being used
    if deformed_data_dir != None:
        print("\n-----\nUsing deformed data\n-----")
        deformed_tensors = glob(deformed_data_dir + "/*.pt")
        if len(deformed_tensors) == 0:
            raise ValueError("No deformed tensors found in specified directory.")
            
    # Get a list of the unique file names in each directory
    partition_names = get_partitions(train_tensors)

    # Randomly shuffle the list
    non_np_random.Random(seed).shuffle(partition_names)
    
    # Split the list 
    train_index_stop = floor(len(partition_names) * 0.8) 
    
    # Get the list of partitions for the training, testing, and validation dataset
    train_partitions = partition_names[0:train_index_stop]
    test_partitions = partition_names[train_index_stop:]
    
    # Load in the dataset
    if deformed_data_dir != None:
        train_examples_paths = string_to_string_match(train_tensors + deformed_tensors, train_partitions)
    else:
        train_examples_paths = string_to_string_match(train_tensors, train_partitions)
    test_examples_paths = string_to_string_match(test_tensors, test_partitions)
    
    # Supply the list to the data loader object which needs to process them.
    train_dataset = CocaDataset(train_examples_paths, means, stds, transforms, False)
    test_dataset = CocaDataset(test_examples_paths, means, stds, None, False)
    
    return train_dataset, test_dataset, test_partitions

def partitions_to_folds(seed, train_data_dir, num_folds):

    # Glob all of the tensor files in the directory
    train_tensors = glob(train_data_dir + "/*.pt")

    # Get a list of the unique file names in each directory
    partition_names = get_partitions(train_tensors)
    
    # Randomly shuffle the list
    non_np_random.Random(seed).shuffle(partition_names)
    
    # Split the partitions into 5 folds
    partition_splits = [list(x) for x in np.array_split(partition_names, num_folds)]
    
    return partition_splits

def load_fold_train_test (test_partitions, train_data_dir, test_data_dir, norm_stats_path, deformed_data_dir = None, transforms=None):

    # Get the statistics for normalization
    means = clean_string_list(pd.read_csv(norm_stats_path).means[0])
    stds = clean_string_list(pd.read_csv(norm_stats_path).stds[0])
    
    # Glob all of the tensor files in the directory
    train_tensors = glob(train_data_dir + "/*.pt")
    test_tensors = glob(test_data_dir + "/*.pt")
    
    # Get the deformed tensors if being used
    if deformed_data_dir != None:
        print("\n-----\nUsing deformed data\n-----")
        deformed_tensors = glob(deformed_data_dir + "/*.pt")
        if len(deformed_tensors) == 0:
            raise ValueError("No deformed tensors found in specified directory.")
            
    # Get a list of the unique file names in each directory
    partition_names = get_partitions(train_tensors)
    
    # Get the test partitions
    train_partitions = []
    for partition in partition_names:
        if partition not in test_partitions:
            train_partitions.append(partition)

    # Load in the dataset
    if deformed_data_dir != None:
        train_examples_paths = string_to_string_match(train_tensors + deformed_tensors, train_partitions)
    else:
        train_examples_paths = string_to_string_match(train_tensors, train_partitions)
    test_examples_paths = string_to_string_match(test_tensors, test_partitions)
    
    # Supply the list to the data loader object which needs to process them.
    train_dataset = CocaDataset(train_examples_paths, means, stds, transforms, False)
    test_dataset = CocaDataset(test_examples_paths, means, stds, None, False)
    
    return train_dataset, test_dataset

def get_partitions (file_list):
    '''
    Get the partition names from the file strings

    Parameters
    ----------
    file_list : List
        A list of files from which the partition names ares extracted.

    Returns
    -------
    partition_names : List
        A List of partition names.

    '''
    partition_names = []
    for file in file_list:
        file_name = file.split('/')[-1].split('_')[0]
        partition_names.append(file_name)
        
    partition_names.sort()
    partition_names = list(set(partition_names))
    
    return partition_names
    

# def get_train_test_from_partitions(train_partitions, test_partitions, train_daat_dir, 
#                                    norm_stats_path, transforms=None):
    
#     # Get the statistics for normalization
#     means = clean_string_list(pd.read_csv(norm_stats_path).means[0])
#     stds = clean_string_list(pd.read_csv(norm_stats_path).stds[0])
    
    # return None

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
 
        # Apply the normalization
        features = self.normalization(features)
        
        # Apply the random augmentation
        if self.augmentations is not None: 
            features, label = self.augmentations((features, label))
        
        return features, label 
        
    def __len__(self):
        return len(self.paths)
    
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

    
    # Define a path to the dataset
    data_dir = glob("/media/john/linux_ssd/coca_tensors_128_train/*.pt")
    
    # Get the means and the standard deviations
    means = clean_string_list(pd.read_csv("/home/john/datasets/coca_data_2022/csvs/norm_stats.csv").means[0])
    stds = clean_string_list(pd.read_csv("/home/john/datasets/coca_data_2022/csvs/norm_stats.csv").stds[0])
    
    # Load in the dataset
    train_dataset = CocaDataset(data_dir, means, stds, augmentations=None)
    
    # Batch size
    batch_size = 1
    
    # Define the DataLoader
    training = DataLoader(train_dataset, 
                          batch_size = batch_size, 
                          shuffle = True,
                          num_workers = 1,
                          drop_last = True, 
                          persistent_workers = False,
                          pin_memory = False
                          )    
    
    # Load a test example
    for x, y in training:
        x = x.squeeze(0)
        y = y.squeeze(0)
        break
    
    # Apply teh deformation
    # x_deformed, y_deformed = apply_elastic_deform(x, y)
    
    # Apply noise
    # aug = load_augmentations()
    # x_noise, y_noise = aug((x, y))
    
    # # Format the image for plotting
    # image_vis = color_norm(torch.Tensor(x)[[5,3,2],:,:].clamp(-3.5, 3.5)).permute(1,2,0).numpy()
    # # image_vis_def = color_norm(x_noise[[5,3,2],:,:].clamp(-3.5, 3.5)).permute(1,2,0).numpy()
    # image_vis_noise = color_norm(x_deformed[[5,3,2],:,:].clamp(-3.5, 3.5)).permute(1,2,0).numpy()
    
    # label = y.numpy()
    # label_noise = y_deformed.numpy()
    
    # # label_def = y_deformed.numpy()
    

    # plt.imshow(image_vis, interpolation='none')
    # plt.show()  
    # plt.imshow(image_vis_noise, interpolation='none')
    # plt.show()
    # plt.imshow(label, interpolation='none')
    # plt.show()
    # plt.imshow(label_noise, interpolation='none')
    # plt.show()
    
    # With deformations + augmentations


    
    
    
    
    
    
    
    
    