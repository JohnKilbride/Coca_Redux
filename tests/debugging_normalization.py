from osgeo import gdal
import numpy as np

if __name__ == "__main__":
    
    # Read the landsat time-series
    raster = gdal.Open(tiff_path)
    image_data = np.array(raster.ReadAsArray())
    
    # create the input tensor and the target labels
    x = image_data[0:-1,:,:]
