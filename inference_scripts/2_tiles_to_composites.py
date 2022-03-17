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
    
    # Set output file information
    output_dir = "/media/john/Expansion/coca_composites"
    output_base = "composite"
    
    # Loop over the years of the time series
    start_year = 1984
    end_year = 2019
    for year in range(start_year, end_year + 1):
    
        # Fire atlas directory
        current_tile_dir = "/media/john/Expansion/yearly_coca_shards/tiles_" + str(year)
        
        # Get all of the file names in the directory
        file_paths = glob.glob(current_tile_dir+"/*.tif")
        
        # Create the VRT
        vrt_name =  output_dir + '/inference.vrt'
        vrt_options = gdal.BuildVRTOptions()
        my_vrt = gdal.BuildVRT(vrt_name, file_paths, options=vrt_options)
        my_vrt = None
        
        # Format the output file name
        out_file = output_dir + "/" + output_base + "_" + str(year) + '.tif'
                
        # Run the translation to mosaic the images
        com2 = "gdal_translate -of GTiff " + vrt_name + " " + out_file
        com2 += ' -co "TILED=YES" -co "BIGTIFF=YES" -co "COMPRESS=LZW" -co "NUM_THREADS=ALL_CPUS"' 
        os.system(com2)
    
    
    
    
        
        
        
