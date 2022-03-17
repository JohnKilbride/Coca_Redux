import ee

ee.Initialize()

def load_image_data():
    '''
    Loads in the medoid composites, elevation, and slope predictors.
    '''
    # Load in the response images
    field_images = ee.Image("users/murillop/CH3/CNN_trainning/field_data_2008_2018")
    
    # Load in the SRTM DEM and compute a slope layer
    elevation = ee.Image("USGS/SRTMGL1_003").toInt16()
    
    # Load in the medoids
    image_stack = ee.Image("users/colombia_andes_amazon/AA-AA-NBR-7-19842019-01011231")
    
    # Define the out band names
    new_bands = ['B1','B2','B3','B4','B5','B7']
    
    # Loop over the years
    all_images = []
    for year in range(1984, 2019 + 1):
        
        # Get the current image band names
        year_str = str(year)
        band_year = [
            'b1_ftv_' + year_str, 'b2_ftv_' + year_str, 'b3_ftv_' + year_str,
            'b4_ftv_' + year_str, 'b5_ftv_' + year_str,'b7_ftv_' + year_str
            ]
        
        # Create the metadata date
        image_date = ee.Date.fromYMD(year, 8, 1).millis()
        
        # Extract the year and rename the bands and set metadata
        image = image_stack.select(band_year, new_bands).set('system:time_start', image_date).toInt16()
        
        # Append to the collection
        all_images.append(image)

    # cast the image list as a collection
    all_images = ee.ImageCollection.fromImages(all_images)        
    
    # Generate the final layer
    outputs = all_images.map(clamp_unmask_image_values)
    
    return ee.ImageCollection(outputs)

def clamp_unmask_image_values (image):
    '''
    Unmasks and then clamps the values of the input image between 0 and 10000.
    '''
    return image.unmask(0, False) \
        .clamp(0, 10000) \
        .set('system:time_start', image.date().millis())

if __name__ == "__main__":
    
    # Load in dataset
    images = load_image_data()
    
    print(images.size().getInfo())
    print(images.first().bandNames().getInfo())