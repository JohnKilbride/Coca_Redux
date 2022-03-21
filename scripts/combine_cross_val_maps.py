from os import system

if __name__ == "__main__":
    
    # Get the python file
    file_name = "/home/john/git/Coca_Redux/project/fuse_time_series_maps.py"

    # Train 
    raster_dir = "/media/john/Expansion/coca_classifications"
    output_dir =  '/media/john/Expansion/coca_cv_mode_maps'
    output_name = "coca_mode_test"
    start_year = 1984
    end_year = 2019
    silent = 'False'

    # Run the model
    com2 = 'python ' + file_name + ' '
    com2 += '--raster_dir ' + raster_dir + ' '
    com2 += '--output_dir ' + output_dir + ' '
    com2 += '--output_name ' + output_name + ' '
    com2 += '--start_year ' + str(start_year) + ' '
    com2 += '--end_year ' + str(end_year) + ' '
    com2 += '--silent ' + silent + ' '
    system(com2)
