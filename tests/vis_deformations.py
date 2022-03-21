import matplotlib.pyplot as plt
import numpy as np
import glob

import torch

def color_norm(array):
    """
    Normalizes numpy arrays into scale 0.0 - 1.0.
    """
    array_min = array.min()
    array_max = array.max()
    return ((array - array_min) / (array_max - array_min))

if __name__ == "__main__":
    
    # Define the directory with the different tiles
    tile_dir = "/media/john/linux_ssd/coca_tensors_128_test"
    
    # Specify the name of a tensor
    original_file = glob.glob(tile_dir + "/*.pt")[0]
    
    # Get the name of the deformed tensor
    tensor_file_name = original_file.split('/')[-1].split('.')[0] + "_deformed.pt"
    tensor_file = "/".join(original_file.split('/')[0:-1]) + "/" + tensor_file_name
    
    # Load in the original tensor that was create and the deformed tensor
    original = torch.load(original_file)
    tensor = torch.load(tensor_file)
    
    plt.imshow(color_norm(original).permute(1,2,0)[:,:,[5,3,2]].numpy())
    plt.show()

    plt.imshow(color_norm(tensor).permute(1,2,0)[:,:,[5,3,2]].numpy())
    plt.show()

    plt.imshow(original.permute(1,2,0)[:,:,-1].numpy())
    plt.show()
    plt.imshow(tensor.permute(1,2,0)[:,:,-1].numpy())
    plt.show()
