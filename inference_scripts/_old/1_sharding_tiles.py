import glob
import os
import pandas as pd
from os import system
import rasterio
from osgeo import gdal 

def get_band_names (filename):
    """
    Get the names of the bands using RasterIO
    """
    ds = rasterio.open(filename)
    bandnames = list(ds.descriptions)
    ds.close()
    return bandnames

def set_band_descriptions(filepath, bands):
    """
    filepath: path to geotiff
    bands:    list of band names
    Based on: https://gis.stackexchange.com/a/290806/177680
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc in enumerate(bands):
        rb = ds.GetRasterBand(band+1)
        rb.SetDescription(desc)
    del ds
    return None

if __name__ == "__main__":
    
    # Fire atlas directory
    tile_dir = "/media/john/Expansion/coca_tiles"
    
    # Set output file information
    shard_dir = "/media/john/Expansion/yearly_coca_shards"
    output_base = "sharded_coca"
    
    # Define the start year and the end year
    start_year = 1984
    end_year = 2019
    
    # Get all of the file names in the directory
    file_paths = glob.glob(tile_dir+"/*.tif")
    
    # Get the band names as a list
    band_names = ['B1','B2','B3','B4','B5','B7','Elevation']
    
    # Create the folders needed
    for year in range(start_year, end_year+1):
        dir_path = shard_dir + "/" + 'tiles_' +str(year)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        
    # Loop over each tile
    for file_path in file_paths:
    
        # Loop over the indices
        for current_index in range(0, (end_year-start_year) + 1):
        
            # Get the current year
            current_year = start_year + current_index
            
            # Get the current file name
            file_name = file_path.split('/')[-1]
           
            # Generate the "master" vrt which glues all the tiles together
            out_file = shard_dir + "/tiles_" + str(current_year) + "/" + file_name
            
            # Get the band numbers
            start_index = current_index * 6
            end_index = start_index + 6
            elevation_index = (((end_year-start_year) + 1) * 6) 
            band_nums = list(range(start_index, end_index)) + [elevation_index]
            
            # Run the translation to mosaic the images
            com2 = "gdal_translate -of GTiff " + file_path + " " + out_file
            com2 += ' -co "TILED=YES" -co "BIGTIFF=YES" -co "COMPRESS=LZW"' 
            for band_num in band_nums:
                com2 += " -b " + str(band_num+1)
            os.system(com2)
        
            # Set the band names with GDAL
            set_band_descriptions(out_file, band_names)
    
        break
    

            
            
    







