import os

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
    
    # Define the working directory -- here it is set as the top level of your Google Drive
    wdir = "/content/drive/MyDrive"
    
    # Specificy the directory with the Shards and where you want the outputs
    shard_dir = wdir + '/FOLDER_WITH_SHARDS'
    output_dir = wdir +'/OUTPUT_FOLDER'
    
    # Aggregate all of the file names
    tif_files = glob.glob(shard_dir + "/*.tif")
    
    # print('Number of raster shards found:{num_files}'.format(num_files = len(tif_files)))
    
    # output_names = []
    # for file in tif_files:
    #     output_names.append(file.split('/')[-1].split("-")[0])
    # output_names = sorted(set(list(output_names)))
    
    # print('Files to be created:{l}'.format(l=output_names))

    # # Loop over the unique file names and generate the mosaics
    # for i, output_base in enumerate(output_names):
        
    #     print("Exporting: {name}".format(name=output_base))
        
    #     # Get of the the files needed to produce the current mosaic
    #     component_files = [z for z in tif_files if output_base in z]
    
    #     # Get the band names as a list
    #     band_names = get_band_names(component_files[0])
        
    #     # Create the output image name
    #     out_file = output_dir + "/" + output_base + ".tif"
        
    #     # Create the VRT
    #     com1 = "gdalbuildvrt " + output_dir + '/' + output_base + ".vrt "
    #     for file in component_files:
    #         com1 += file + " "
    #     os.system(com1)
        
    #     # Run the translation to mosaic the images
    #     com2 = "gdal_translate -of GTiff " + output_dir + '/' + output_base + ".vrt " 
    #     com2 += out_file+ " -co BIGTIFF=YES -co COMPRESS=LZWA -co NUM_THREADS=ALL_CPUS --config GDAL_CACHEMAX 512"
    #     os.system(com2)
    
    #     # Set the band names with GDAL
    #     set_band_descriptions(out_file, band_names)