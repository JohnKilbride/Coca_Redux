import itertools
from os import system
from random import shuffle

if __name__ == "__main__":
    
    # Get the python file
    file_name = "/home/john/git/coca_project_2022/project/segmentation_3yr.py"
    
    # Set the experiment name
    experiment_name = "debugging"
    epochs = 50
    
    # Train 
    learning_rate = [0.0001]
    batch_size = [8]
    encoder = ['timm-mobilenetv3_large_100']
    loss_function = ['jaccard']
    
    
    # Get the args
    args = [learning_rate, batch_size, encoder, loss_function]
    args = list(itertools.product(*args)) 
    shuffle(args)
    
    # Loop over the command
    print('\nBeginning experiments...')
    for i, arg in enumerate(args):
        
        print("\n=================================================")
        print("         Experiment {} of {}".format(i+1, len(args)))
        print("=================================================\n")
        
        # Unpack the args
        cur_lr = arg[0]
        cur_batch = arg[1]
        cur_encoder = arg[2]
        cur_loss = arg[3]

        # Run the model
        com2 = 'python ' + file_name + ' '
        com2 += '--seed 7385817 '
        com2 += '--data_dir "/media/john/linux_ssd/coca_tensor_256_3yr/" '
        com2 += '--log_dir "/home/john/datasets/coca_data_2022/tensorboard_logs/" '
        com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
        com2 += '--max_epochs ' + str(epochs) + ' '
        com2 += '--encoder ' + str(cur_encoder) + ' '
        com2 += '--loss ' + str(cur_loss) + ' '
        com2 += '--batch_size ' + str(cur_batch) + ' '
        com2 += '--learning_rate ' + str(cur_lr) + ' '
        com2 += '--num_nodes 1 '
        com2 += '--devices 2 '
        com2 += '--accelerator "gpu" '
        com2 += '--num_workers 1 '
        com2 += '--precision 16 '
        system(com2)