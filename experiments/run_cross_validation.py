import itertools
from os import system
import glob
from random import shuffle
from pathlib import Path

def load_all_checkpoints(chkp_dir, exp_name):
    
    # Get all of the checkpoint files in the logger directory
    all_checkpoint_files = list(glob.glob(chkp_dir + '/**/*.ckpt', recursive=True))
    
    # Get the loggers that are part of the cross-validation experiment
    checkpoint_files = []
    for file in all_checkpoint_files:
        if exp_name in file:
            checkpoint_files.append(file)
            
    # Raise a value error if no checkpoints are found
    if len(checkpoint_files) == 0:
        raise ValueError("Error: no checkpoints found for the specified experiment in the specified logger directory.")

    return checkpoint_files

if __name__ == "__main__":
    
    # Set constant parameters
    py_file_name = "/home/john/git/Coca_Redux/project/model_applicator.py"
    encoder = "timm-mobilenetv3_large_100"
    decoder = "Unet"
    time_series_dir = "/media/john/Expansion/coca_composites_small"
    output_dir = '/media/john/Expansion/coca_classifications'
    chunk_size = 128
    scale_factor = 2
    device = 0
    bands = "[0,1,2,3,4,5,6]"

    # Define the parameters related to loading the checkpoint files
    logger_dir = "/home/john/datasets/coca_data_2022/resnest_cross_val/"
    experiment_name = "timm-mobilenetv3_large_100-Unet-jaccard-0.005-128-128-False-Experiment3"
    
    # Get the checkpoint files
    checkpoint_paths = load_all_checkpoints(logger_dir, experiment_name)
    
    # Fop over checkpoint files
    for fold_i, checkpoint_paths in enumerate(checkpoint_paths):
        
        # Loop over the year
        for year in range(1984, 2020):
            
            # Format the output name
            out_name = "test_" + str(year) + "_fold_" + str(fold_i)
            
            # Run the model
            com2 = 'python ' + py_file_name + ' '
            com2 += '--encoder ' + encoder + ' '
            com2 += '--decoder ' + decoder + ' '
            com2 += '--model_path ' + checkpoint_paths + ' '
            com2 += '--time_series_dir ' + time_series_dir + ' '
            com2 += '--output_dir ' + output_dir + ' ' 
            com2 += '--chunk_size ' + str(chunk_size) + ' '
            com2 += '--scale_factor ' + str(scale_factor) + ' '
            com2 += '--out_name ' + out_name + ' '
            com2 += '--device ' + str(device) + ' '
            com2 += '--bands ' + bands + ' '
            com2 += '--year ' + str(year) + ' '
            com2 += '--silent ' + str(True) + ' '
            com2 += '--in_memory ' + str(True) + ' '
            system(com2)    
            
            break



