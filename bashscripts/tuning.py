import itertools
from os import system
from random import shuffle

if __name__ == "__main__":
    
    # Get the python file
    file_name = "/home/john/git/coca_project_2022/project/segmentation.py"
    
    # Train    
    learning_rate = [0.003, 0.001, 0.005, 0.0001]
    batch_size = [32, 64, 128]
    encoder = ['timm-resnest50d']
    decoder = ["Unet"]
    loss_name = ['jaccard']
    epochs = 60
    suffix = "overlap16"
    resolution = 128
    use_noise = "True"
    
    # Get the args
    args = [learning_rate, batch_size, encoder, decoder, loss_name]
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
        cur_decoder = arg[3]
        cur_loss = arg[4]

        # Run the model
        com2 = 'python ' + file_name + ' '
        com2 += '--seed 343468 '
        com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
        com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
        com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
        com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
        com2 += '--max_epochs ' + str(epochs) + ' '
        com2 += '--encoder ' + str(cur_encoder) + ' '
        com2 += '--decoder ' + str(cur_decoder) + ' '
        com2 += '--loss_name ' + str(cur_loss) + ' '
        com2 += '--batch_size ' + str(cur_batch) + ' '
        com2 += '--learning_rate ' + str(cur_lr) + ' '
        com2 += '--resolution ' + str(resolution) + ' '
        com2 += '--suffix ' + suffix + ' '
        com2 += '--num_nodes 1 '
        com2 += '--gpus 2 '
        com2 += '--use_noise ' + str(use_noise) + ' '
        com2 += '--accelerator "gpu" '
        com2 += '--num_workers 3  '
        com2 += '--precision 16 '
        system(com2)
