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
    output_dir = "/media/john/Expansion/coca_composites"
    output_base = "all_coca_features"
    
    # Get all of the file names in the directory
    file_paths = glob.glob(tile_dir+"/*.tif")
    
    # Create the VRT
    vrt_name =  output_dir + '/master_inference.vrt'
    vrt_options = gdal.BuildVRTOptions()
    my_vrt = gdal.BuildVRT(vrt_name, file_paths, options=vrt_options)
    my_vrt = None

    # Loop over the years of the dataset
    start_year = 1984
    end_year = 2018
    for current_index in range(0, (end_year-start_year) + 1):
        
        if current_index in [0, 5, 10, 15, 20, 25, 30, 34]:
        
            # Get the current year
            current_year = start_year + current_index
            
            # Get the band names as a list
            band_names = get_band_names(file_paths[0])
        
            # Generate the "master" vrt which glues all the tiles together
            out_file = output_dir + "/composite_" + str(current_year) + ".tif"
            
            # Get the band numbers
            start_index = current_index * 6
            end_index = start_index + 6
            elevation_index = (((end_year-start_year) + 1) * 6) 
            band_nums = list(range(start_index, end_index)) + [elevation_index]
            
            # Run the translation to mosaic the images
            com2 = "gdal_translate -of GTiff " + vrt_name + " " + out_file
            com2 += ' -co "TILED=YES" -co "BIGTIFF=YES" -co "COMPRESS=LZW" -co "NUM_THREADS=ALL_CPUS"' 
            for band_num in band_nums:
                com2 += " -b " + str(band_num+1)
            os.system(com2)
        
            # # # Set the band names with GDAL
            # # set_band_descriptions(out_file, band_names)
            
            
    







