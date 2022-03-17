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
          [-73.6791659890863, 1.990360897403594],
          [-76.95677538600745, 0.43161009215791124],
          [-76.95746203151526, 0.608070654722247],
          [-76.60769184332847, 0.617736153795764],
          [-76.44839008551597, 0.6170495481515365],
          [-76.27741535407065, 0.4206770814648492],
          [-76.61763224742407, 0.7864226556724432],
          [-76.77418742320532, 0.7980945133646382],
          [-77.10858378551, 0.6146960980408462],
          [-77.13261637828344, 0.44853509600563446],
          [-76.0907221752237, 0.42149807486091295],
          [-75.94927320061433, 0.4269910883376127],
          [-75.91768750725495, 0.27318541194280327],
          [-75.40812154306842, 0.6028901606626443],
          [-75.24126668466998, 0.5980839060818408],
          [-75.57840962900592, 0.6125026570629118],
          [-75.75573144286982, 0.7753929122549764],
          [-75.91228661865107, 0.9600792913926729],
          [-76.2659090551745, 0.9525272428496591],
          [-76.44684201160156, 0.9391197348307557],
          [-76.4324224559375, 1.133065283403911],
          [-76.09104798391986, 1.119698645800212],
          [-75.23714131108461, 1.286421646063049],
          [-75.06479328862368, 1.2802433865970102],
          [-75.06190411370363, 1.4768521695435433],
          [-74.88992094638793, 1.6349239731896084],
          [-74.88374113681762, 1.8113120016996491],
          [-74.88957864317305, 1.9801240970260374],
          [-74.7126185225758, 1.4603026900883553],
          [-74.56176351955959, 1.3024020470413091],
          [-74.70626792578152, 1.1328049684192198],
          [-74.52705344824246, 0.9426354451586689],
          [-74.53117332128933, 1.1314319454790276],
          [-74.87449607519558, 0.9604857662581021],
          [-75.04203757910183, 0.9584261185489561],
          [-75.03791770605496, 1.1273128727626016],
          [-75.24398733772142, 0.9507573271107188],
          [-72.46788030623617, 2.15449155099249],
          [-72.64984136580648, 2.1613531366423153],
          [-72.30583196639242, 2.1654700731525987],
          [-72.29982278291888, 2.0023229507758935],
          [-72.47217080537982, 1.8273261865823422],
          [-72.63902566377826, 1.987912136322776],
          [-72.81892678682513, 1.9851672049923241],
          [-72.47563290466933, 2.330639853396664],
          [-72.1336834417787, 2.521356128775139],
          [-72.46242348753385, 2.507310448790765],
          [-73.00833827017952, 2.330313818307203],
          [-73.67341842495775, 2.8467934203888396],
          [-73.500383756989, 2.858451929442637],
          [-71.60525289754175, 3.014742416531854],
          [-71.7673012373855, 3.029827606494761],
          [-71.77554098347925, 3.201235199218116],
          [-71.78103414754175, 3.369872287412958],
          [-71.597013151448, 3.389064927700176],
          [-71.44942220476008, 3.3637014566545353],
          [-71.44598897722102, 3.5288837070767136],
          [-71.29698690202571, 3.3787815160463026],
          [-71.91257601144709, 3.194709143631521],
          [-71.45076413544268, 3.0183200740257323],
          [-71.32579465302081, 3.01283451651988],
          [-71.60617761730592, 2.8528822513445267],
          [-72.1087194753307, 2.673240407692302],
          [-71.92263854271351, 2.6656955034688927],
          [-72.28381407982289, 2.656092831208443],
          [-72.29023944943403, 2.3341518348623826],
          [-72.98506027679925, 2.167054169093373],
          [-72.30276271702418, 1.8160808254029686],
          [-72.47305080296168, 1.6606174889752152],
          [-72.2878337631565, 1.4737650104669886],
          [-71.96236379245337, 1.4717057544060155],
          [-71.78006702483736, 1.4733707004789154],
          [-71.78783672278186, 1.2964845138399044],
          [-71.60518901770374, 1.2999168601420776],
          [-71.96430461828967, 1.1143696377656063],
          [-76.27275139647381, 0.6064187314497518],
          [-76.95042277672113, 0.7825345291281037],
          [-76.7734655946255, 0.9448562979533273],
          [-74.2134025884956, 0.6042067002983627],
          [-74.36377795470653, 0.41676000039871397],
          [-74.17907031310497, 0.43873204419576345],
          [-74.01648639895268, 0.43606107559577517],
          [-74.19707416750737, 0.25822322461673125],
          [-75.05555568228172, 0.419905673088134],
          [-75.72601552422896, 0.08767196489026435],
          [-75.92308731022793, 0.7657077042119934],
          [-76.0920541179905, 0.9394438811581495],
          [-75.40826645734647, 0.9468408936362409],
          [-76.79386034339615, 0.27820593267639837],
          [-76.59988298743912, 0.2806091633776248],
          [-76.4344014200563, 0.2864455787311053],
          [-76.97170152991959, 0.2977750823487211],
          [-74.88460091848783, 1.3087506423280697],
          [-74.88711926682882, 1.1237184522539103],
          [-70.22438270941855, 2.3324355908085197],
          [-70.39386805910209, 3.7061624495595855],
          [-70.38734492677787, 3.885326296071771],
          [-70.3898793207398, 3.5454164501307486],
          [-70.2092915521851, 3.8941821591229053],
          [-70.2202360182124, 4.048515186427878],
          [-70.55332151589867, 3.715006209701778],
          [-70.55538145242211, 3.8897153165941862],
          [-70.0548168772268, 4.04794999502541],
          [-70.05253133194002, 3.8939898727155766]
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
    dir_base = "coca_tiles_mmu"
    
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
    imagery = load_image_data()

    # Create the export image
    export_image = imagery.toBands() \
        .addBands(load_reference_labels()) \
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

def load_image_data ():
    '''
    Loads in the medoid composites, elevation, and slope predictors.
    '''
    # Load in the response images
    field_images = ee.Image("users/murillop/CH3/CNN_trainning/field_data_2008_2018")
    
    # Load in the SRTM DEM and compute a slope layer
    elevation = ee.Image("USGS/SRTMGL1_003").toInt16()
    
    # Load in the medoids
    m = ee.Image("users/colombia_andes_amazon/AA-AA-NBR-7-19842019-01011231")
    new_bands = ['B1','B2','B3','B4','B5','B7']

    DEF_BANDS_2006 = ['b1_ftv_2006','b2_ftv_2006','b3_ftv_2006','b4_ftv_2006','b5_ftv_2006','b7_ftv_2006']
    DEF_BANDS_2007 = ['b1_ftv_2007','b2_ftv_2007','b3_ftv_2007','b4_ftv_2007','b5_ftv_2007','b7_ftv_2007']    
    DEF_BANDS_2008 = ['b1_ftv_2008','b2_ftv_2008','b3_ftv_2008','b4_ftv_2008','b5_ftv_2008','b7_ftv_2008']
    DEF_BANDS_2009 = ['b1_ftv_2009','b2_ftv_2009','b3_ftv_2009','b4_ftv_2009','b5_ftv_2009','b7_ftv_2009']
    DEF_BANDS_2010 = ['b1_ftv_2010','b2_ftv_2010','b3_ftv_2010','b4_ftv_2010','b5_ftv_2010','b7_ftv_2010']
    DEF_BANDS_2011 = ['b1_ftv_2011','b2_ftv_2011','b3_ftv_2011','b4_ftv_2011','b5_ftv_2011','b7_ftv_2011']
    DEF_BANDS_2012 = ['b1_ftv_2012','b2_ftv_2012','b3_ftv_2012','b4_ftv_2012','b5_ftv_2012','b7_ftv_2012']
    DEF_BANDS_2013 = ['b1_ftv_2013','b2_ftv_2013','b3_ftv_2013','b4_ftv_2013','b5_ftv_2013','b7_ftv_2013']
    DEF_BANDS_2014 = ['b1_ftv_2014','b2_ftv_2014','b3_ftv_2014','b4_ftv_2014','b5_ftv_2014','b7_ftv_2014']
    DEF_BANDS_2015 = ['b1_ftv_2015','b2_ftv_2015','b3_ftv_2015','b4_ftv_2015','b5_ftv_2015','b7_ftv_2015']
    DEF_BANDS_2016 = ['b1_ftv_2016','b2_ftv_2016','b3_ftv_2016','b4_ftv_2016','b5_ftv_2016','b7_ftv_2016']
    DEF_BANDS_2017 = ['b1_ftv_2017','b2_ftv_2017','b3_ftv_2017','b4_ftv_2017','b5_ftv_2017','b7_ftv_2017']
    DEF_BANDS_2018 = ['b1_ftv_2018','b2_ftv_2018','b3_ftv_2018','b4_ftv_2018','b5_ftv_2018','b7_ftv_2018']

    y2006 = m.select(DEF_BANDS_2006, new_bands).set({'system:time_start': ee.Date.fromYMD(2006, 8, 1).millis()}).toInt16()
    y2007 = m.select(DEF_BANDS_2007, new_bands).set({'system:time_start': ee.Date.fromYMD(2007, 8, 1).millis()}).toInt16()    
    y2008 = m.select(DEF_BANDS_2008, new_bands).set({'system:time_start': ee.Date.fromYMD(2008, 8, 1).millis()}).toInt16()
    y2009 = m.select(DEF_BANDS_2009, new_bands).set({'system:time_start': ee.Date.fromYMD(2009, 8, 1).millis()}).toInt16()
    y2010 = m.select(DEF_BANDS_2010, new_bands).set({'system:time_start': ee.Date.fromYMD(2010, 8, 1).millis()}).toInt16()
    y2011 = m.select(DEF_BANDS_2011, new_bands).set({'system:time_start': ee.Date.fromYMD(2011, 8, 1).millis()}).toInt16()
    y2012 = m.select(DEF_BANDS_2012, new_bands).set({'system:time_start': ee.Date.fromYMD(2012, 8, 1).millis()}).toInt16()
    y2013 = m.select(DEF_BANDS_2013, new_bands).set({'system:time_start': ee.Date.fromYMD(2013, 8, 1).millis()}).toInt16()
    y2014 = m.select(DEF_BANDS_2014, new_bands).set({'system:time_start': ee.Date.fromYMD(2014, 8, 1).millis()}).toInt16()
    y2015 = m.select(DEF_BANDS_2015, new_bands).set({'system:time_start': ee.Date.fromYMD(2015, 8, 1).millis()}).toInt16()
    y2016 = m.select(DEF_BANDS_2016, new_bands).set({'system:time_start': ee.Date.fromYMD(2016, 8, 1).millis()}).toInt16()
    y2017 = m.select(DEF_BANDS_2017, new_bands).set({'system:time_start': ee.Date.fromYMD(2017, 8, 1).millis()}).toInt16()
    y2018 = m.select(DEF_BANDS_2018, new_bands).set({'system:time_start': ee.Date.fromYMD(2018, 8, 1).millis()}).toInt16()
    
    # Generate the final layer
    outputs = ee.ImageCollection([y2006, y2007, y2008,y2009,y2010,y2011,y2012,y2013,y2014,y2015,y2016,y2017,y2018]) \
        .map(clamp_unmask_image_values)
    
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
    # labels = ee.Image("users/murillop/CH3/CNN_trainning/field_data_2008_2018") \
    #     .select(
    #         ["class","class_1","class_2","class_3","class_4","class_5","class_6","class_7","class_8","class_9","class_10"], 
    #         ["label_2008","label_2009","label_2010","label_2011", "label_2012","label_2013","label_2014","label_2015","label_2016","label_2017","label_2018"]
    #         ) \
    #     .unmask(0, False) \
    #     .toInt16()    
    labels = ee.Image("users/kilbridj/Coca_Project/labels_mmu_5")
    
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

