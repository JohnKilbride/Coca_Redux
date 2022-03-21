from os import system

if __name__ == "__main__":
    
    # Get the python file
    file_name = "/home/john/git/Coca_Redux/project/cross_validation.py"

    # Train 
    num_folds = 5
    learning_rate = 0.0025
    batch_size = 128
    epochs = 30
    encoder = 'timm-mobilenetv3_large_100'
    decoder = "Unet"
    loss_name = 'jaccard'
    suffix = "5FoldMobileNet"
    resolution = 128
    use_noise = "True"
    gpus = 2
    num_workers = 4
    precision = 32

    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--seed 654365 '
    com2 += '--train_data_dir "/media/john/linux_ssd/coca_tensors_128_train/" '
    com2 += '--test_data_dir "/media/john/linux_ssd/coca_tensors_128_test/" '
    com2 += '--num_folds ' + str(num_folds) + ' '
    com2 += '--log_dir "/home/john/datasets/coca_data_2022/resnest_cross_val/" '
    com2 += '--norm_stats_csv "/home/john/datasets/coca_data_2022/csvs/norm_stats.csv" '
    com2 += '--max_epochs ' + str(epochs) + ' '
    com2 += '--encoder ' + str(encoder) + ' '
    com2 += '--decoder ' + str(decoder) + ' '
    com2 += '--loss_name ' + str(loss_name) + ' '
    com2 += '--batch_size ' + str(batch_size) + ' '
    com2 += '--learning_rate ' + str(learning_rate) + ' '
    com2 += '--resolution ' + str(resolution) + ' '
    com2 += '--suffix ' + suffix + ' '
    com2 += '--num_nodes 1 '
    com2 += '--gpus ' + str(gpus) + ' '
    com2 += '--num_workers ' + str(num_workers) + ' '
    com2 += '--use_noise ' + str(use_noise) + ' '
    com2 += '--accelerator "gpu" ' + ' '
    com2 += '--precision ' + str(precision) + ' '
    system(com2)
