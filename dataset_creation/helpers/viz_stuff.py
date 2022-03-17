import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import random

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

def plot_tensor_label(in_tensor):
    '''
    Plot the a given tensor for a single 3 channel input
    '''
    permuted = np.moveaxis(in_tensor.cpu().numpy(), (0,1,2), (2,0,1))
    plt.imshow(permuted, vmin=0, vmax=2)
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
    
    # Read in the CSV
    folder = "/home/john/datasets/coca_data_2022/tensors"
    
    # Get all of the tensors paths and shuffle them
    files = glob.glob(folder + "/*.pt")
    random.shuffle(files)
    
    # Get a random path
    path = files[0]
    
    # Read in teh dataset
    x = torch.load(path)
    
    # Do some plotting
    rgb = x[[2,1,0],:,:]
    swir = x[[5,3,2],:,:]
    dem = x[-2,:,:].unsqueeze(0)
    label = x[-1,:,:].unsqueeze(0)
    
    plot_tensor(rgb)
    # plot_tensor(swir)
    # plot_tensor(dem)
    plot_tensor_label(label)
    
    print(label.max())