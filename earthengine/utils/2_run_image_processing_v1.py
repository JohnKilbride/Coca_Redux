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
    grid = ee.FeatureCollection("users/kilbridj/Coca_Project/export_grid_19230m") \
        .filterBounds(ee.Geometry.MultiPoint([
            [-74.19613842960771, 2.5014176498348286],
            [-74.02173047062334, 2.493871724595353],
            [-74.02173047062334, 2.3299088848862524],
            [-74.20025830265459, 2.3312810399470765],
            [-73.85006909367021, 2.328536728488412],
            [-73.85006909367021, 2.666044985976053],
            [-73.6681080340999, 2.672218091122908],
            [-73.67428784367021, 2.5069055682020713],
            [-73.67085461613115, 2.3347114217444482],
            [-73.5033131122249, 2.327164570754344],
            [-73.33371167179521, 2.326478491386449],
            [-73.15312390324053, 2.325106331649484],
            [-73.16548352238115, 2.5144514183699855],
            [-73.33577160831865, 2.5130794488543504],
            [-72.99382214542803, 2.5130794488543504],
            [-72.99656872745928, 2.677019373668668],
            [-72.82490735050615, 2.679762955250769],
            [-73.1572437762874, 2.690051331368096],
            [-73.33989148136553, 2.68593599131875],
            [-73.33645825382646, 2.8491669586623964],
            [-71.95037821954058, 1.9933909720886758],
            [-72.12821940606402, 1.9927047419641173],
            [-72.12478617852496, 2.162880804416775],
            [-72.1323392791109, 2.332351703190632],
            [-71.9510648650484, 2.34195674378983],
            [-71.95037821954058, 2.1532745854940245],
            [-71.77734355157183, 2.160822334044777],
            [-71.59881571954058, 2.1649392719957117],
            [-71.43196086114214, 2.165625427234212],
            [-71.77665690606402, 2.328921315640273],
            [-71.7767563262799, 1.6379381101612356],
            [-71.60784153135802, 1.4629076757932202],
            [-71.43274692686583, 1.4539841759316712],
            [-75.23309069728552, 0.42335141163606876],
            [-75.40612536525427, 0.4240380383696766],
            [-75.5832799062699, 0.4336508062009487],
            [-75.58121996974646, 0.08895886582211805],
            [-75.4081853017777, 0.08964551049589893],
            [-75.22897082423864, 0.08071912876842118],
            [-75.2324040517777, -0.07637716833975695],
            [-75.4074986562699, -0.07706381323199106],
            [-76.27400709869154, 1.1162055759025662],
            [-76.10097243072279, 1.2967532349661932],
            [-75.92050833973673, 1.4670599685695433],
            [-75.40813600515784, 1.6318920433838098],
            [-75.06138002371253, 1.6435602487439862],
            [-74.71531068777503, 1.986710219030956],
            [-74.37610780691566, 1.802789915607866],
            [-74.37061464285316, 1.4554882625345296],
            [-74.21131288504066, 1.650423867137477],
            [-74.21680604910316, 1.8261241571215312],
            [-74.54502260183753, 2.32841675738376],
            [-75.23538071903987, 1.1358536729593869],
            [-75.59243638310237, 0.7623704268233376],
            [-75.75860459599299, 0.9408784660809675],
            [-74.72314317021174, 0.9422515716673389],
            [-74.72180477858993, 1.2972210820799632],
            [-75.24640194655868, 1.4702052264046874],
            [-73.68485439153763, 2.1629230774450674],
            [-72.8210543427095, 2.161550764212142],
            [-72.64252651067825, 2.338568647628113],
            [-72.474985006772, 1.998236852387942],
            [-71.9537995051632, 1.2990864740881067],
            [-72.14331366531945, 1.4665791243992616],
            [-74.36793004706703, 0.6179790552267552],
            [-75.06281530097328, 0.7800152819595801],
            [-76.79041539862953, 0.6042469262386243],
            [-76.60364782050453, 0.4312193882927864],
            [-76.61738073066078, 0.9342970543566457],
            [-76.79330891508182, 0.42877969854656584],
            [-76.44723957914432, 0.7693367535611983],
            [-76.44174641508182, 0.43427270680977664],
            [-75.75784748930057, 1.3020846941246778],
            [-74.72398858938273, 1.815503745360871],
            [-74.89427667532023, 1.4668314523153565],
            [-75.06731134328898, 1.993932865331188],
            [-75.58968327988393, 1.1258217334974046],
            [-74.02962468613393, 2.160843190005808],
            [-72.3003818093988, 2.5063190647992557],
            [-73.6791659890863, 1.990360897403594]
            ]))
    
    # Define the start year and the end years of the time-series
    sub_start_year = 1984
    sub_end_year = 1990
    start_year = 1991
    end_year = 2018
    
    # Define the start and end of the time-period to composite -- pick the good season
    # Source: https://www.tandfonline.com/doi/pdf/10.1080/01431160010006926
    start_julian = 1
    end_julian = 365
    
    # Define the directory part (after the username) to the 
    # GEE Image Collection that will be used to store the outputs
    # of the compositing and interpolation
    dir_base = "coca_tiles"
    
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
    
        # break
    
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
    composited_images = medoid_timeseries_burnin(raw_imagery, sub_start_year, sub_end_year, start_year, end_year) \
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
    fitted_images = ee.ImageCollection(flatten_ltr_fits(landtrendr, sub_end_year, end_year)) \
        .filterDate("2006-01-01", "2018-12-31")

    # Create the export image
    export_image = fitted_images.toBands() \
        .addBands(load_reference_labels()) \
        .addBands(load_srtm()) \
        .toInt16()
    
    # Initiate the export task
    task = ee.batch.Export.image.toDrive(
        image = export_image, 
        description = file_prefix + "-FeaturesAndLoss", 
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
  
def  generate_geo_mask (input_geometry):
    '''
    Generate a mask to reduce image processing time
    '''
    out_mask = ee.Image.constant(0).byte().paint(input_geometry, 1)
    return out_mask

def load_reference_labels ():
    '''
    Load in the reference labels
    '''
    labels = ee.Image("users/murillop/CH3/CNN_trainning/field_data_2008_2018") \
        .select(
            ["class","class_1","class_2","class_3","class_4","class_5","class_6","class_7","class_8","class_9","class_10"], 
            ["label_2008","label_2009","label_2010","label_2011", "label_2012","label_2013","label_2014","label_2015","label_2016","label_2017","label_2018"]
            ) \
        .unmask(0, False) \
        .toInt16()        
    return labels

def load_srtm ():
    '''
    Load the SRTN elevation dataset.
    '''
    srtm = ee.Image("USGS/SRTMGL1_003").select(['elevation'])
    return srtm.toInt16()

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

# Interpolate a time series of landsat images
def interpolate_collection (image_collection, start_year, end_year):
  
    # Calculate NBR
    nbr_images = image_collection.map(calculate_nbr_ltr)
    
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
    fitted_images = flatten_ltr_fits(landtrendr, start_year, end_year)
    
    return fitted_images

# Calculates and appends an inverted Normalized Burn ratio band to each image.
# This band will be used during the LandTrendr fitting process. 
def calculate_nbr (image):
    nbr = image.normalizedDifference(['B4', 'B7']).multiply(-1).rename('NBR')
    return nbr.addBands(image).set('system:time_start', image.get('system:time_start')).toFloat()

# Flattens the fitted NBR values produced by LandTrendr 
def flatten_ltr_fits (ltr_output, start_year, end_year):
  
    # Cast the years as ee.Numbers to get the index values
    start_year = ee.Number(start_year)
    end_year = ee.Number(end_year)
  
    # Select the fitted bands of the LandTrendr output
    def loop_years (year):
        return ee.Number(year).toInt16().format()
    band_names = ee.List(['B1','B2','B3','B4','B5','B7'])
    years = ee.List.sequence(start_year, end_year).map(loop_years) 
    fitted_values = ltr_output.select(['B1_fit','B2_fit','B3_fit','B4_fit','B5_fit','B7_fit']) \
        .toArray(1) \
        .arrayTranspose(0,1) \
        .arrayFlatten([band_names, years])
  
    # Flatten a single from the LandTrendr fitted values
    def loop_over_years (year):
      
        # Cast the input
        year = ee.Number(year).toInt16()
    
        # Create the band name
        def loop_bands (band):
            return ee.String(band).cat(ee.Number(year).toInt16().format())
        band_names = ee.List(['B1_','B2_','B3_','B4_','B5_','B7_']).map(loop_bands)
    
        # Flatten the fitted spectal values for a single year
        return fitted_values.select(band_names, ['B1','B2','B3','B4','B5','B7']).set({
            'system:time_start': ee.Date.fromYMD(year, 8, 1).millis()
            }).toFloat()
      
    outputs = ee.List.sequence(start_year, end_year).map(loop_over_years)
  
    return ee.ImageCollection.fromImages(outputs)


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

def randomword(length):
    '''Generate a random length string'''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

if __name__ == "__main__":
    
    main()

