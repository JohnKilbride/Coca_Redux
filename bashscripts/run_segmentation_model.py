import itertools
from os import system
from random import shuffle

if __name__ == "__main__":
    
    # Get the python file
    file_name = "/home/john/git/coca_project_2022/project/segmentation.py"
    
    # Train 
    learning_rate = 0.001
    decay = 0
    batch_size = 128
    epochs = 60
    encoder = 'timm-mobilenetv3_large_100'
    decoder = "Unet"
    loss_name = 'jaccard'
    suffix = "Geo"
    resolution = 128
    use_noise = "False"

    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 343468 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)

    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 234213 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)

    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 543254342 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)
    
    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 75436 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)
    
    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 1234321 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)

    # Train 
    learning_rate = 0.001
    decay = 0
    batch_size = 128
    epochs = 60
    encoder = 'timm-mobilenetv3_large_100'
    decoder = "Unet"
    loss_name = 'jaccard'
    suffix = "Geo+Noise"
    resolution = 128
    use_noise = "True"

    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 343468 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)

    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 52341 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)
    
    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 1123497 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)
    
    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 89343 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)
    
     # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 2756134 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/debugging_logs/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--decay ' + str(decay) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus 2 '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" '
    com2 += '--num_workers 3  '
    com2 += '--precision 16 '
    system(com2)
 