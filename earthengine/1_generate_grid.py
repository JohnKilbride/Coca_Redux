import ee

ee.Initialize()

def main ():
  
    # Define the username
    username = "kilbridj"
    
    # Define the area over which to tile the study area
    study_area = ee.Geometry.Polygon([[
        [-78.03392155552253, 5.391582891571006],
        [-78.03392155552253, -4.877742854845319],
        [-66.47630436802253, -4.877742854845319],
        [-66.47630436802253, 5.391582891571006]
        ]])

    # Define the export collection
    export_collection = 'Coca_Project'
    
    generate_study_area_fishnet(study_area, username, export_collection)
    
    return None
    
# Run fishnet generation logic
def generate_study_area_fishnet (study_area, username, export_collection):
    
    # First generate a fishnet of the study area
    def assign_grid_id (feat):
        feat = ee.Feature(feat)
        id = feat.getString('system:index')
        return feat.set('grid_id', id)
    sample_grid = generate_fishnet(study_area, 19230).map(assign_grid_id)
    
    # Export the image to an image collection
    task = ee.batch.Export.table.toAsset(
        collection = sample_grid, 
        description = "Export-Grid-Strata", 
        assetId = "users/" + username + "/" + export_collection + "/export_grid_19230m"
        )
    task.start()
        
    return None

# Generate a fishnet of a given size of a target geometry
def generate_fishnet (study_area, grid_size_m):

    # functionine the projection with linear unit set at a 1m scale.
    proj = ee.Projection('EPSG:4326').atScale(1)
    
    # Create the image to vectorize
    im = ee.Image.pixelLonLat().clip(study_area)
    im2 = im.select([0]).add(im.select([1])).multiply(-1000000).int()
    im2 = im2.reproject(proj, None, grid_size_m)
    
    # Vectorize the image
    fishnet = im2.reduceToVectors(
        geometry = study_area, 
        scale = grid_size_m, 
        geometryType = 'polygon',
        eightConnected = False,
        bestEffort = True,
        geometryInNativeProjection = True
        )
    
    return fishnet

if __name__ == "__main__":
    
    main()