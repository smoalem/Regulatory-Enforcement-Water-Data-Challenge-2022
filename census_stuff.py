import json
from census_area import Census
import os
import pandas as pd
import numpy as np
import time
import regex as re
from pygit2 import Repository
import geopandas as gpd
import wdc_lib as wdc
import concurrent.futures


# Pip install the census_area package from here: https://github.com/datamade/census_area

# Example script: https://github.com/datamade/chicago-community-area-ward-demographics/blob/master/scripts/calculate_demographics.py
# Documentation: https://census-area.readthedocs.io/en/latest/

# Description of the summary file (sf1) variables:
# main page: https://www.census.gov/data/datasets/2010/dec/summary-file-1.html
# List of tables: https://www2.census.gov/programs-surveys/decennial/2010/technical-documentation/complete-tech-docs/summary-file/sf1.pdf

# Description of the ACS 5-year (acs5) variables:
# main page: https://www.census.gov/data/developers/data-sets/acs-5year.html
# variables: https://api.census.gov/data/2020/acs/acs5/variables.html

start = time.perf_counter()

# Load up the census data
# API KEY REDACTED
api_key = 'ad3130119c535c4902b79422cdacefe1bd3a7190'  # api key for sarmad
c = Census(api_key, year=2020)

# BACKGROUND: We are attempting to obtain demographic information for California water systems.
# These system boundaries are available in the SABL dataset:
# https://gispublic.waterboards.ca.gov/portal/apps/webappviewer/index.html?id=272351aa7db14435989647a86e6d3ad8
# Note: The shapefile is only available to download by request.

# NECESSARY CONVERSIONS
# You only need to do this once to produce the geojson file in the proper format. Comment out afterwards
# SABL to geojson
# will need a custom function in wdc to get all devs hyperlink variations
# filepath to
os.chdir(r'C:\Users\sarma\Dropbox\Water Data Challenge 2022\GIS\public_water_systems')
sfile = gpd.read_file("SABL_Public_220615.shp")
print(type(sfile))
sfile.to_file("SABL.json", driver="GeoJSON")
print('this step done')


# Note: I had to manually add the ogr2ogr script to my Library folder in my environment.
# Get it here: http://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/ogr2ogr.py

# To get help on usage: ogr2ogr.Usage()

# This is the step to adjust the geojson to the correct projection

command = "ogr2ogr SABL_layer.geojson -t_srs \"EPSG:4326\" SABL.json"
os.system(command)
print('command complete')

# Open the geojson
with open('SABL_layer.geojson') as infile:
    print('start infile')
    my_shape_geojson = json.load(infile)
    print('finished infile')

# Let's say we want only a subset of water systems. Make a list.
# pwsid_list = ['CA0010005', 'CA4010011', 'CA3010037', 'CA1910065', 'CA2410005']


pwsid_list = ['CA4810701', 'CA1510003']
print(pwsid_list)

pre_pwsid_iteration = time.perf_counter()
print(
    f'Time to prep for pwsid iteration (seconds): {pre_pwsid_iteration - start}')
# We're going to create a smaller geojson of just these shapes
subset_json = {
    'type': my_shape_geojson['type'], 'crs': my_shape_geojson['crs']}
print('subset_json made')
subset_list = []
f_counter = 0  # deleting this after testing
for f in my_shape_geojson['features']:
    f_counter += 1
    pwsid = f['properties']['SABL_PWSID']
    if pwsid in pwsid_list:
        subset_list += [f]
print(f'f_counter: {f_counter}')
subset_json['features'] = subset_list

# Here we make a list of the ACS fields that we will want to retrieve from the census
acs_keys = {'B19001_001E': 'n_hh_income', 'B19001_002E': 'n_hh_income_lt_10k', 'B19001_003E': 'n_hh_income_10k_15k',
            'B19001_004E': 'n_hh_income_15k_20k', 'B19001_005E': 'n_hh_income_20k_25k', 'B19001_006E': 'n_hh_income_25k_30k',
            'B19001_007E': 'n_hh_income_30k_35k', 'B19001_008E': 'n_hh_income_35k_40k', 'B19001_009E': 'n_hh_income_40k_45k',
            'B19001_010E': 'n_hh_income_45k_50k', 'B19001_011E': 'n_hh_income_50k_60k', 'B19001_012E': 'n_hh_income_60k_75k',
            'B19001_013E': 'n_hh_income_75k_100k', 'B19001_014E': 'n_hh_income_100k_125k', 'B19001_015E': 'n_hh_income_125k_150k',
            'B19001_016E': 'n_hh_income_150k_200k', 'B19001_017E': 'n_hh_income_gt_200k', 'B07412_002E': 'n_100pct_pov_lvl',
            'B07412_003E': 'n_101_149pct_pov_lvl', 'B07412_004E': 'n_150pct_pov_lvl', 'B25032_002E': 'n_hh_own',
            'B25032_013E': 'n_hh_rent', 'B08202_001E': 'hh_size', 'B08202_003E': 'hh_1worker', 'B08202_004E': 'hh_2worker',
            'B08202_005E': 'hh_3+worker', 'B08202_013E': 'n_hh_3ppl', 'B08202_018E': 'n_hh_4+ppl', 'B11011_001E': 'hh_type'}

vlist = tuple(['NAME']+list(acs_keys.keys()))


# OPTION 1: Store as a Pandas dataframe to export to excel

df_store = pd.DataFrame()

# Iterate over each water system boundary in the subset geojson
n_counter = 0  # deleting this after testing
tract_counter = 0
for n in range(0, len(subset_list)):
    n_counter += 1
    print('begin test')
    print(type(subset_json['features']))  # list
    # len equals number of water systems you're evaluating
    print(len(subset_json['features']))
    print(subset_json['features'])
    print('$$$$$$$$$')
    print(type(subset_json['features'][n]))  # dict
    print(subset_json['features'][n])
    # dict_keys(['type', 'crs', 'features'])
    # dict_keys(['type', 'properties', 'geometry'])
    print(subset_json['features'][n].keys())
    raise ValueError
    system_area = subset_json['features'][n]
    # Generate the overlap
    overlap_features = c.acs5.geo_tract(vlist, system_area['geometry'], 2020)
    # Iterate over each overlapped census tract
    for tract_geojson, tract_data, tract_proportion in overlap_features:
        tract_counter += 1
        # Include the tract properties obtained from the overlap
        tract_geojson['properties'].update(tract_data)
        # add the proportion of overlap between the census tract and the water system
        tract_geojson['properties'].update({'proportion': tract_proportion})
        # Add information about the water system ID, name, regulating agency, classification, and total population
        for k in ['SABL_PWSID', 'WATER_SY_1', 'REGULATING', 'STATE_CLAS', 'POPULATION']:
            tract_geojson['properties'].update(
                {k: system_area['properties'][k]})
        # Append these properties as a new row to the dataframe
        df_store = df_store.append(
            tract_geojson['properties'], ignore_index=True)

print(f'n_counter: {n_counter}')
print(f'tract_counter: {tract_counter}')

# Rename the columns using the acs_keys dictonary
df_final = df_store.rename(columns=acs_keys)
print(df_final)
print(time.perf_counter() - start)
raise ValueError
# Output to excel
df_final.to_excel('acs_SABL_overlaps.xlsx', index=False)
# Or, output to csv
df_final.to_csv('acs_SABL_overlaps.csv', index=False)


# OPTION 2 (not fully tested): Create a list of geojsons (possibly able to re-export to shapefile?)

# List to store the resulting overlap jsons
store_jsons = []
for n in range(0, len(subset_list)):
    system_area = subset_json['features'][n]
    # Generate the overlap
    overlap_features = c.acs5.geo_tract(vlist, system_area['geometry'], 2020)
    # Create the features list for the overlaps
    features = []
    for tract_geojson, tract_data, tract_proportion in overlap_features:
        tract_geojson['properties'].update(tract_data)
        tract_geojson['properties'].update({'proportion': tract_proportion})
        for k in ['SABL_PWSID', 'WATER_SY_1', 'REGULATING', 'STATE_CLAS', 'POPULATION']:
            tract_geojson['properties'].update(
                {k: system_area['properties'][k]})
        features.append(tract_geojson)
    # Create a new geojson and add the features to it
    overlap_geojson = {
        'type': 'FeatureCollection',
        'crs': my_shape_geojson['crs'],
        'features': features
    }
    # Store the new geojson in the list
    store_jsons += [overlap_geojson]