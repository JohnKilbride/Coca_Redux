import torch
from barbar import Bar
import os
import numpy as np
from glob import glob
from osgeo import gdal
import pandas as pd
import os

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, valid_ids):
        '''
        Args:
            root_dir (string): Directory with all of the images
            transform (calllable, optional): Optional transforms that can be 
                applied to the images. 
        '''
        # Initalize the class attributes
        self.root_dir = root_dir
        self.valid_ids = valid_ids
        self.paths = None
        
        # Get the file names
        self.__get_file_list()
        
        return
    
    def __get_file_list(self):
        '''Get a list of the tensors in the target directory.'''
        
        # Glob all of the tiffs in the root dir
        all_paths = glob(self.root_dir + "/*.tif")
                
        # Check if the image is in the list
        valid_paths = []
        
        # Loop over all paths
        for path in all_paths:
            
            # print(path)
            
            # Get just the partition name
            file_name = path.split('/')[-1].split('.')[0]
            
            # Check if the partition name is in the list of valid partition names
            for grid_id in self.valid_ids:
                if file_name == grid_id:
                    valid_paths.append(path)
                    break
                    
        self.paths = valid_paths
        if len(self.paths) == 0:
            raise ValueError("Dataset found no `.tif` files in specified directory.")
        
        return None

    def __getitem__(self, idx):
        '''
        Args:
            idx (int): Index of tensor to retrieve
        
        Returns:
            torch.tensor: DL Tensor [row, col, band]
        
        '''  
        # if the idx provided is a tensor convert it to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load in the image
        tiff_path = os.path.join(self.root_dir, self.paths[idx])
                
        # Read the landsat time-series
        raster = gdal.Open(tiff_path)
        image_data = np.array(raster.ReadAsArray())
                    
        # Split the features and the label
        spectral_bands = image_data[0:66,:,:]
        dem_band = np.expand_dims(image_data[-1,:,:], axis=0)
        
        # Reshape the features
        spectral_reshaped = spectral_bands.reshape((int(spectral_bands.shape[0]/6), 6, spectral_bands.shape[1], spectral_bands.shape[2]))
        spectral_avg = np.mean(spectral_reshaped, axis=(0))
        
        # Join the two labels
        features = np.concatenate((spectral_avg, dem_band), axis=0)

        return features 

    def __len__(self):
        return len(self.paths)
    
if __name__ == "__main__":

    # Path to the tiffs
    tiff_path = "/home/john/datasets/coca_data_2022/coca_tiles"
    
    # Get the list of tensors to process
    grid_to_use = pd.read_csv("/home/john/datasets/coca_data_2022/csvs/grid_samples_to_keep.csv")
    valid_ids = grid_to_use["grid_id"].tolist()

    # Define a path to the output directory
    output_dir = "/home/john/datasets/coca_data_2022/csvs"
    
    # Load teh dataset
    print('Getting dataloader...')
    dataset = Dataset(tiff_path, valid_ids)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        num_workers = os.cpu_count() - 4,
        shuffle = False
    )
    pop_mean = []
    pop_std0 = []
    print("Initiating processing...")
    for i, data in enumerate(Bar(loader)):

        # get the inputs
        numpy_image = data
        numpy_image = numpy_image.numpy()
                
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))
        
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
                
    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    np.set_printoptions(suppress=True)
    pop_mean_out = np.array(pop_mean).mean(axis=0).round(decimals=5)
    pop_std0_out = np.array(pop_std0).mean(axis=0).round(decimals=5)
    
    # Creat the output dataframe
    output_name = output_dir + "/norm_stats.csv"
    out_df = pd.DataFrame(data={'means': [pop_mean_out], 'stds': [pop_std0_out]})
    out_df.to_csv(output_name, index=False) 
    
    print("\nProgram Completed\n")