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
    # Load in the amazon rain forest
    grid = ee.FeatureCollection("users/kilbridj/Coca_Project/export_grid_38460m") \
        .filterBounds(ee.Geometry.Polygon([[
            [-77.52982948959972, 0.16196088330412164],
            [-76.05766152084972, 0.09054992849034865],
            [-75.79828270895013, -0.1511919565680721],
            [-75.1104335669343, -0.23912048121607002],
            [-74.77412215595446, -0.40943048320020836],
            [-74.15212828799524, -1.2497971838994655],
            [-73.8251117553942, -1.312861095738866],
            [-73.73296264259294, -1.9483666503689512],
            [-73.32206581772472, -2.342269040998568],
            [-72.23454136019059, -2.693896169203317],
            [-71.23466347397472, -2.649594027300353],
            [-70.60332527416706, -2.6218696656742053],
            [-70.32582540210694, -2.7944690986983405],
            [-70.73478554428722, -4.015046654293857],
            [-69.91081093491222, -4.376623172092754],
            [-69.89526209327592, -3.703191355159535],
            [-69.61642836227085, -3.6113644490433208],
            [-69.52154128605385, -1.634848544277264],
            [-69.36161306397695, -1.5194887928992855],
            [-69.26219111378555, -0.8661113628241425],
            [-69.579541647445, -0.4649192997674257],
            [-69.95244030830828, -0.3741845926838406],
            [-69.93532348878284, 0.41966670985511356],
            [-69.3078383796014, 0.5187782298581101],
            [-68.98246620834972, 0.4860519488073514],
            [-68.57605994353959, 1.5022569645123012],
            [-67.68056407247614, 1.46115816851156],
            [-67.19989440580285, 1.0999966935044703],
            [-66.78520058334972, 1.1341770601681007],
            [-67.19169472397472, 3.3077052583128905],
            [-67.83988808334972, 5.0169878766536415],
            [-69.58671425522472, 4.57907579359782],
            [-70.35405567564558, 4.3337538989514055],
            [-70.98908849241835, 4.000448940353316],
            [-71.40404587561099, 3.7434575026331087],
            [-71.9066686083425, 3.6176781606833353],
            [-72.39255780342404, 3.30493437801244],
            [-73.0100613091228, 3.233156138041634],
            [-74.48661659897472, 3.351576385802497],
            [-75.9320077267985, 1.9746580277190309],
            [-76.84867714584972, 1.8919638563338637],
            [-77.62870644272472, 0.5080237781549672]
            ]])
            )
    
    # Define the directory part (after the username) to the 
    # GEE Image Collection that will be used to store the outputs
    # of the compositing and interpolation
    dir_base = "coca_inference"
    
    # Covert the grid strata to a list and get length for looping
    grid_list = grid.toList(1e6)
    list_length = grid_list.length().getInfo()
    
    # Load the image collection
    images = load_image_data()
    
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
        process_tile(images, current_perimeter.geometry(), dir_name, file_name)
            
    return None

def process_tile (images, study_area, asset_dir, file_prefix):
    '''
    Wrapper def  for all the key image processing tasks.
    '''
    # Create the study area mask
    study_area_mask = generate_geo_mask(study_area)

    # Create the export image
    export_image = images.toBands() \
        .addBands(load_srtm()) \
        .updateMask(study_area_mask) \
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
  
def  generate_geo_mask (input_geometry):
    '''
    Generate a mask to reduce image processing time
    '''
    out_mask = ee.Image.constant(0).byte() \
        .paint(input_geometry.buffer(300), 1)
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

