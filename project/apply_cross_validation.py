import itertools
from os import system
from random import shuffle

if __name__ == "__main__":
    
    # Parameters
    experiment_log = 
    experiment_name = 
    
    
    
    
    # Provide a list of values to use for inferenmce
    tmp_chp_dir = "/home/john/datasets/coca_data_2022/debugging_logs/timm-mobilenetv3_large_100-Unet-jaccard-0.001-128-128-True-TuningTest/version_0/checkpoints"
    tmp_model_path = tmp_chp_dir + '/' + 'timm-mobilenetv3_large_100-Unet-jaccard-0.001-128-128-True-TuningTest-epoch=48-val_logger_loss=0.32.ckpt'
    
    
    # # Get the python file
    # file_name = "/home/john/git/Coca_Redux/project/model_applicator.py"

    # # Train 
    # learning_rate = 0.01
    # batch_size = 128
    # epochs = 60
    # encoder = 'timm-resnest14d'
    # decoder = "Unet"
    # loss_name = 'lovasz'
    # suffix = "tinkering"
    # # suffix = "test_lovasz+Warp"
    # resolution = 128
    # use_noise = "True"

    # # Run the model
    # com2 = 'python ' + file_name + ' '
    # com2 += '--seed 343468 '
    # com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    # com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    # # com2 += '--deformed_data_dir "/media/john/linux_ssd/coca_tensors_128_elastic/" '
    # com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    # com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    # com2 += '--max_epochs ' + str(epochs) + ' '
    # com2 += '--encoder ' + str(encoder) + ' '
    # com2 += '--decoder ' + str(decoder) + ' '
    # com2 += '--loss_name ' + str(loss_name) + ' '
    # com2 += '--batch_size ' + str(batch_size) + ' '
    # com2 += '--learning_rate ' + str(learning_rate) + ' '
    # com2 += '--resolution ' + str(resolution) + ' '
    # com2 += '--suffix ' + suffix + ' '
    # com2 += '--num_nodes 1 '
    # com2 += '--gpus 2 '
    # com2 += '--use_noise ' + str(use_noise) + ' '
    # com2 += '--accelerator "gpu" '
    # com2 += '--num_workers 2 '
    # com2 += '--precision 32 '
    # system(com2)
