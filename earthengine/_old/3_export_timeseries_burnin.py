import ee
import random
import string
from utils import task_monitor

ee.Initialize()

# Global variables
PRJ = 'EPSG:4326'
MONITOR = task_monitor.GEETaskMonitor()

def  main ():
    '''
    Define the main logic def 
    '''
    # Load in the amazon rain forest
    grid = ee.FeatureCollection("users/kilbridj/Coca_Project/export_grid_38460m") \
        .filterBounds(ee.Geometry.Point([-73.43885234811381, 2.2673007569681944]))
        # .filterBounds(ee.Geometry.Polygon([[
        #     [-77.52982948959972, 0.16196088330412164],
        #     [-76.05766152084972, 0.09054992849034865],
        #     [-75.79828270895013, -0.1511919565680721],
        #     [-75.1104335669343, -0.23912048121607002],
        #     [-74.77412215595446, -0.40943048320020836],
        #     [-74.15212828799524, -1.2497971838994655],
        #     [-73.8251117553942, -1.312861095738866],
        #     [-73.73296264259294, -1.9483666503689512],
        #     [-73.32206581772472, -2.342269040998568],
        #     [-72.23454136019059, -2.693896169203317],
        #     [-71.23466347397472, -2.649594027300353],
        #     [-70.60332527416706, -2.6218696656742053],
        #     [-70.32582540210694, -2.7944690986983405],
        #     [-70.73478554428722, -4.015046654293857],
        #     [-69.91081093491222, -4.376623172092754],
        #     [-69.89526209327592, -3.703191355159535],
        #     [-69.61642836227085, -3.6113644490433208],
        #     [-69.52154128605385, -1.634848544277264],
        #     [-69.36161306397695, -1.5194887928992855],
        #     [-69.26219111378555, -0.8661113628241425],
        #     [-69.579541647445, -0.4649192997674257],
        #     [-69.95244030830828, -0.3741845926838406],
        #     [-69.93532348878284, 0.41966670985511356],
        #     [-69.3078383796014, 0.5187782298581101],
        #     [-68.98246620834972, 0.4860519488073514],
        #     [-68.57605994353959, 1.5022569645123012],
        #     [-67.68056407247614, 1.46115816851156],
        #     [-67.19989440580285, 1.0999966935044703],
        #     [-66.78520058334972, 1.1341770601681007],
        #     [-67.19169472397472, 3.3077052583128905],
        #     [-67.83988808334972, 5.0169878766536415],
        #     [-69.58671425522472, 4.57907579359782],
        #     [-70.35405567564558, 4.3337538989514055],
        #     [-70.98908849241835, 4.000448940353316],
        #     [-71.40404587561099, 3.7434575026331087],
        #     [-71.9066686083425, 3.6176781606833353],
        #     [-72.39255780342404, 3.30493437801244],
        #     [-73.0100613091228, 3.233156138041634],
        #     [-74.48661659897472, 3.351576385802497],
        #     [-75.9320077267985, 1.9746580277190309],
        #     [-76.84867714584972, 1.8919638563338637],
        #     [-77.62870644272472, 0.5080237781549672]
        #     ]])
        #     )
    
    # Define the start year and the end years of the time-series
    sub_start_year = 1984
    sub_end_year = 1989
    start_year = 1990
    end_year = 2021
    
    # Define the start and end of the time-period to composite -- pick the good season
    # Source: https://www.tandfonline.com/doi/pdf/10.1080/01431160010006926
    start_julian = 1
    end_julian = 365
    
    # Define the directory part (after the username) to the 
    # GEE Image Collection that will be used to store the outputs
    # of the compositing and interpolation
    dir_base = "coca_inference_tiles"
    
    # Covert the grid strata to a list and get length for looping
    grid_list = grid.toList(1e6)
    list_length = grid_list.length().getInfo()
    
    # Loop over the individual perimeters
    for index in range(0, list_length):
        
        print("Processing perimeter {x} of {y}".format(x=index+1, y=list_length))
        
        # Monitoring to prevent too many GEE tasks from being executed
        if index != 0 and index % 500 == 0:
            run_monitoring()
        
        # Get the current perimeter from the list
        current_perimeter = ee.Feature(grid_list.get(index))
        
        # Get the ID of the current perimeter
        perimeter_id = current_perimeter.getString('grid_id') \
            .replace("\\+", "p", 'g').replace("\\-", "m", 'g') \
            .getInfo()
        
        # Create the folder name and the file name
        dir_name = dir_base
        file_name = perimeter_id
        
        # Process the tile -- generate a time-series of interpolated medoids for that tile
        # And export to google drive
        process_tile(current_perimeter.geometry(), dir_name, file_name, sub_start_year, sub_end_year, start_year, end_year, start_julian, end_julian)
    
    return None

def process_tile (study_area, asset_dir, file_prefix, sub_start_year, sub_end_year, start_year, end_year, start_julian, end_julian):
    '''
    Wrapper def  for all the key image processing tasks.
    '''
    # Create the study area mask
    study_area_mask = generate_geo_mask(study_area)
    
    # Load in all of the imagery
    raw_imagery = load_imagery(start_julian, end_julian, sub_start_year, end_year, study_area, study_area_mask) \
        .select(['B1','B2','B3','B4','B5','B7'])
    
    # Generate annual composites using the masked imagery using the burn in proceedure
    composited_images = medoid_timeseries_burnin (raw_imagery, sub_start_year, sub_end_year, start_year, end_year) \
        .sort('system:time_start')

    # Calculate NBR
    nbr_images = composited_images.map(calculate_nbr)
    
    # Perform the landtrendr segmentation of the time-series
    landtrendr = ee.Algorithms.TemporalSegmentation.LandTrendr(
        timeSeries = nbr_images,
        maxSegments = 10,
        spikeThreshold = 0.9,
        vertexCountOvershoot = 3,
        preventOneYearRecovery = True,
        recoveryThreshold = 0.75,
        pvalThreshold = 0.05,
        bestModelProportion = 0.75,
        minObservationsNeeded = 6
        )
    
    # Construct the fitted images
    fitted_images = ee.ImageCollection(flatten_ltr_fits(landtrendr, sub_end_year, end_year))

    # Create the export image
    export_image = fitted_images.toBands() \
        .addBands(load_srtm()) \
        .toInt16()
    
    # Initiate the export task
    task = ee.batch.Export.image.toDrive(
        image = export_image, 
        description = file_prefix + "-StudyArea", 
        folder = asset_dir, 
        fileNamePrefix = file_prefix,
        region = study_area,
        crs = PRJ, 
        scale = 30, 
        maxPixels = 1e13
        )
      
    # Initiate the tasks
    task.start()
    MONITOR.add_task(randomword(25), task)
        
    return None
  
def  generate_geo_mask (input_geometry):
    '''
    Generate a mask to reduce image processing time
    '''
    out_mask = ee.Image.constant(0).byte().paint(input_geometry, 1)
    return out_mask

def  load_imagery(start_day, end_day, start_year, end_year, geometry, study_area_mask):
    '''
    Load in the imagery
    '''
    # Define a def  with the scope needed to loop over the collections
    def  apply_mask (image):
        return ee.Image(image).updateMask(study_area_mask)
    
    # Load in the landsat data
    ls4 = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2") \
        .filterBounds(geometry) \
        .filterDate(ee.Date.fromYMD(start_year,1,1), ee.Date.fromYMD(end_year,12,31)) \
        .filter(ee.Filter.calendarRange(start_day, end_day)) \
        .select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','QA_PIXEL'],
                ['B1','B2','B3','B4','B5','B7','QA_PIXEL']) \
        .map(apply_qa_mask_tm) \
        .map(scale_sr_values) \
        .map(rgb_sanity_check) \
        .map(apply_mask)
    ls5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2") \
        .filterBounds(geometry) \
        .filterDate(ee.Date.fromYMD(start_year,1,1), ee.Date.fromYMD(end_year,12,31)) \
        .filter(ee.Filter.calendarRange(start_day, end_day)) \
        .select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','QA_PIXEL'],
                ['B1','B2','B3','B4','B5','B7','QA_PIXEL']) \
        .map(apply_qa_mask_tm) \
        .map(scale_sr_values) \
        .map(rgb_sanity_check) \
        .map(apply_mask)
    ls7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
        .filterBounds(geometry) \
        .filterDate(ee.Date.fromYMD(start_year,1,1), ee.Date.fromYMD(end_year,12,31)) \
        .filter(ee.Filter.calendarRange(start_day, end_day)) \
        .select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','QA_PIXEL'],
                ['B1','B2','B3','B4','B5','B7','QA_PIXEL']) \
        .map(apply_qa_mask_tm) \
        .map(scale_sr_values) \
        .map(rgb_sanity_check) \
        .map(apply_mask)
    ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(geometry) \
        .filterDate(ee.Date.fromYMD(start_year,1,1), ee.Date.fromYMD(end_year,12,31)) \
        .filter(ee.Filter.calendarRange(start_day, end_day)) \
        .select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','QA_PIXEL'],
                ['B1','B2','B3','B4','B5','B7','QA_PIXEL']) \
        .map(apply_qa_mask_oli) \
        .map(scale_sr_values) \
        .map(rgb_sanity_check) \
        .map(apply_mask)
      
    return ee.ImageCollection(ls4.merge(ls5).merge(ls7).merge(ls8))

def flatten_ltr_fits (ltr_output, start_year, end_year):
    '''
    Flattens the fitted NBR values produced by LandTrendr 
    '''
    # Cast the years as ee.Numbers to get the index values
    start_year = ee.Number(start_year)
    end_year = ee.Number(end_year)
    
    # Select the fitted bands of the LandTrendr output
    def year_label (year):
        return ee.Number(year).toInt16().format()
    band_names = ee.List(['B1','B2','B3','B4','B5','B7'])
    years = ee.List.sequence(start_year, end_year).map(year_label)
    fitted_values = ltr_output.select(['B1_fit','B2_fit','B3_fit','B4_fit','B5_fit','B7_fit']) \
        .toArray(1) \
        .arrayTranspose(0,1) \
        .arrayFlatten([band_names, years])
    
    # Flatten a single from the LandTrendr fitted values
    def loop_over_years (year):
      
        # Cast the input
        year = ee.Number(year).toInt16()
          
        # Create the band name
        def loop_over_bands (band):
            return ee.String(band).cat(ee.Number(year).toInt16().format())
        band_names = ee.List(['B1_','B2_','B3_','B4_','B5_','B7_']).map(loop_over_bands)
          
        # Flatten the fitted spectal values for a single year
        return fitted_values.select(band_names, ['B1','B2','B3','B4','B5','B7']) \
          .set('system:time_start', ee.Date.fromYMD(year, 8, 1).millis()) \
          .toFloat()

    outputs = ee.List.sequence(start_year, end_year).map(loop_over_years)
    
    return ee.ImageCollection.fromImages(outputs)

def  scale_sr_values (image):
    '''
    Scale the Collection 2 data 0-1 SR value
    '''  
    # Compute the rescaled values
    scaled = image.select(['B1','B2','B3','B4','B5','B7']) \
        .multiply(0.0000275).add(-0.2) \
        .multiply(10000)
      
    return image.addBands(scaled, None, True) \
        .set('system:time_start', image.date().millis()) \
        .toInt16()


def  apply_qa_mask_tm (img):
    '''
    def  applies the Landsat SR cloud mask for Landsat 4-7
    '''    
    # Isolate the QA band
    qa = img.select('QA_PIXEL')
    
    # Do the bitwise operations needed to obtain the mask
    mask = qa.bitwiseAnd(1 << 1).eq(0) \
        .And(qa.bitwiseAnd(1 << 2).eq(0)) \
        .And(qa.bitwiseAnd(1 << 3).eq(0)) \
        .And(qa.bitwiseAnd(1 << 4).eq(0)) \
        .And(qa.bitwiseAnd(1 << 5).eq(0))  
  
    # Define the area
    output = img.select(['B1','B2','B3','B4','B5','B7']) \
        .updateMask(mask)
    
    return output.toFloat()

def  apply_qa_mask_oli (img):
    '''
    Applies the Landsat SR cloud mask for Landsat 4-7
    '''
    # Isolate the QA band
    qa = img.select('QA_PIXEL')
    
    # Do the bitwise operations needed to obtain the mask
    mask = qa.bitwiseAnd(1 << 1).eq(0) \
        .And(qa.bitwiseAnd(1 << 2).eq(0)) \
        .And(qa.bitwiseAnd(1 << 3).eq(0)) \
        .And(qa.bitwiseAnd(1 << 4).eq(0)) \
        .And(qa.bitwiseAnd(1 << 5).eq(0))  
  
    # Define the area
    output = img.select(['B1','B2','B3','B4','B5','B7']).updateMask(mask)
    
    return output.toFloat()

def  rgb_sanity_check (image):
    '''
    Create a sanity check mask -- drop very bright pixels that are probably clouds
    '''
    # Sum the RGB bands and then keep those with a values less than 6000
    rgb_sum_mask = image.select(['B1','B2','B3']) \
        .reduce('sum') \
        .lt(6000) \
        .rename('rgb_sum')
      
    return image.updateMask(rgb_sum_mask).toInt16()

def  medoid_timeseries_burnin (scenes, sub_start_year, sub_end_year, start_year, end_year):
    '''
    This def  fuses the first several years of the time-series to create "composited"
    starting image. This is done to prevent LandTrendr fitting errors that can occur when
    the first year(s) of the timeseries are very noisey.
    '''
  
    # Create a bunch of dates
    sub_date_1 = ee.Date.fromYMD(sub_start_year, 1, 1)
    sub_date_2 = ee.Date.fromYMD(sub_end_year, 12, 31)
    sub_meta_date = ee.Date.fromYMD(sub_end_year, 8, 1).millis()
    
    # Create the "false" start image
    start_image = medoid_mosaic(scenes.filterDate(sub_date_1, sub_date_2), sub_end_year) \
        .set('system:time_start', sub_meta_date)  
    
    # Create the annual collection
    medoids = generate_annual_collection (scenes, start_year, end_year)
    
    # Create the new time series
    new_time_series = ee.ImageCollection([start_image]).merge(medoids)
    
    return new_time_series
  
def  medoid_mosaic (inCollection, image_year):
    '''
    Medoid compositing function
    '''
    # fill in missing years with the dummy collection
    dummy_image = ee.Image([0,0,0,0,0,0]).mask(ee.Image(0)).rename(['B1','B2','B3','B4','B5','B7']) \
        .set('system:time_start', ee.Date.fromYMD(image_year, 1, 1).millis())
    dummyCollection = ee.ImageCollection([dummy_image])
    finalCollection = inCollection.merge(dummyCollection)
    
    # calculate median across images in collection per band
    median = finalCollection.median()                                                                       
     
    # calculate the difference between the median and the observation per image per band
    def comute_diff (img):
        diff = ee.Image(img).subtract(median).pow(ee.Image.constant(2))                                       
        return diff.reduce('sum').addBands(img).set('system:time_start', img.date().millis())
    difFromMedian = finalCollection.map(comute_diff)
    
    # Get the medoid by selecting the image pixel with the smallest difference 
    # between median and observation per band 
    return ee.ImageCollection(difFromMedian) \
        .reduce(ee.Reducer.min(7)) \
        .select([1,2,3,4,5,6], ['B1','B2','B3','B4','B5','B7']) 


def  generate_annual_collection (input_collection, start_year, end_year):
    '''
    Loops over each year of the dataset and generates a medoid composites
    This modified variant uses several years worth of imagery to compute the median
    during the medoid calculation -- this helps stabalize the median of the compositing values
    '''
    # Generate the annual composites using the medoid method
    def  loop_over_year (year):
      
        # Cast the input
        year = ee.Number(year)
        
        # Filter the image collection to isolate a particular year
        start_single_year = ee.Date.fromYMD(year, 1, 1)
        end_single_year = ee.Date.fromYMD(year, 12, 31)
        metadata_year = ee.Date.fromYMD(year, 8, 1).millis()
        collection_single_year = input_collection.filterDate(start_single_year, end_single_year)
        
        # Generate the composite and push into the client side list
        annual_composite = medoid_mosaic(collection_single_year, year)
        
        return annual_composite.set('system:time_start', metadata_year)
    
    composited_images = ee.List.sequence(start_year, end_year).map(loop_over_year)
    
    return ee.ImageCollection.fromImages(composited_images)

def load_srtm ():
    '''
    Load the SRTN elevation dataset.
    '''
    srtm = ee.Image("USGS/SRTMGL1_003").select(['elevation'])
    return srtm.toInt16()

def run_monitoring():
    '''
    Check the status of the task monitor. If full, being monitoring
    '''
    # Clear any completed tasks
    MONITOR.check_status()
    
    # Check the capacity and hold if it is full
    while MONITOR.get_monitor_capacity() >= 0.90:
        print("Monitoring tasks on GEE server...")
        MONITOR.check_status()
    
    return None

def calculate_nbr (image):
    '''
    Calculates and appends an inverted Normalized Burn ratio band to each image.
    This band will be used during the LandTrendr fitting process. 
    '''
    nbr = image.normalizedDifference(['B4', 'B7']) \
        .multiply(-1) \
        .rename('NBR')
    return nbr.addBands(image).set('system:time_start', image.get('system:time_start')).toFloat()

def randomword(length):
    '''Generate a random length string'''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

if __name__ == "__main__":
    
    main()

