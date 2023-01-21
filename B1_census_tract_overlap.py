import difflib
import json
from random import SystemRandom
from census_area import Census
import os
import pandas as pd
import numpy as np
import time
import geopandas as gpd
import wdc_lib as wdc
import concurrent.futures
import cProfile
import pstats
import io
from pstats import SortKey


# Credit to Marielle Rhodeiro for sharing her census-to-system-area-boundaries overlap script


# Pip install the census_area package from here: https://github.com/datamade/census_area

# Example script: https://github.com/datamade/chicago-community-area-ward-demographics/blob/master/scripts/calculate_demographics.py
# Documentation: https://census-area.readthedocs.io/en/latest/

# Description of the summary file (sf1) variables:
# main page: https://www.census.gov/data/datasets/2010/dec/summary-file-1.html
# List of tables: https://www2.census.gov/programs-surveys/decennial/2010/technical-documentation/complete-tech-docs/summary-file/sf1.pdf

# Description of the ACS 5-year (acs5) variables:
# main page: https://www.census.gov/data/developers/data-sets/acs-5year.html
# variables: https://api.census.gov/data/2020/acs/acs5/variables.html

def census_tracts_geojson(api_key, c):
    # Load up the census data
    # API KEY REDACTED

    # BACKGROUND: We are attempting to obtain demographic information for California water systems.
    # These system boundaries are available in the SABL dataset:
    # https://gispublic.waterboards.ca.gov/portal/apps/webappviewer/index.html?id=272351aa7db14435989647a86e6d3ad8
    # Note: The shapefile is only available to download by request.
    # NECESSARY CONVERSIONS
    # You only need to do this once to produce the geojson file in the proper format. Comment out afterwards
    # SABL to geojson

    # filepath to folder with SABL shape file for Jae
    os.chdir(
        r'C:\Users\hoonje92\Dropbox\Water Data Challenge 2022\GIS\public_water_systems')
    # filepath to folder with SABL shape file for Sarmad
    # os.chdir(
    #     r'C:\Users\sarma\Dropbox\Water Data Challenge 2022\GIS\public_water_systems')

    sfile = gpd.read_file("SABL_Public_220615.shp")
    sfile.to_file("SABL.json", driver="GeoJSON")
    # Note: I had to manually add the ogr2ogr script to my Library folder in my environment.
    # Get it here: http://svn.osgeo.org/gdal/trunk/gdal/swig/python/samples/ogr2ogr.py
    # To get help on usage: ogr2ogr.Usage()
    # This is the step to adjust the geojson to the correct projection
    command = "ogr2ogr SABL_layer.geojson -t_srs \"EPSG:4326\" SABL.json"
    os.system(command)
    # Open the geojson
    with open('SABL_layer.geojson') as infile:
        my_shape_geojson = json.load(infile)
    return my_shape_geojson


def overlap_generator(system_area, vlist, c):
    if system_area['geometry'] == None:
        return 'null geometry'
    else:
        overlap_features = c.acs5.geo_tract(
            vlist, system_area['geometry'], 2020)
        for tract_geojson, tract_data, tract_proportion in overlap_features:
            try:
                # Include the tract properties obtained from the overlap
                tract_geojson['properties'].update(tract_data)
                # add the proportion of overlap between the census tract and the water system
                tract_geojson['properties'].update(
                    {'proportion': tract_proportion})
                # Add information about the water system ID, name, regulating agency, classification, and total population
                for k in ['SABL_PWSID', 'WATER_SY_1', 'REGULATING', 'STATE_CLAS', 'POPULATION']:
                    tract_geojson['properties'].update(
                        {k: system_area['properties'][k]})
                return tract_geojson['properties']
            except:
                print('failed:')
                print(system_area)
                print(overlap_features)
                print(type(overlap_features))
                raise ValueError


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()
    # REDACTED, request API key from:
    # https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_api_handbook_2020_ch02.pdf#:~:text=%E2%80%A2%20Click%20on%20the%20Request%20a%20KEY%20box,Bureau%20data%20sets%20using%20a%20variety%20of%20tools
    api_key = '########################################'
    c = Census(api_key, year=2020)
    my_shape_geojson = census_tracts_geojson(api_key, c)
    # my_shape_geojson is a dictionary with dict_keys(['type', 'name', 'crs', 'features'])
    # 'type': 'FeatureCollection'
    # 'name': 'SABL'
    # 'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}
    # > 'features': returns a list of dictionaries. Each item in my_shape_geojson{'features'} list has dictionary with dict_keys(['type','properties','geometry']):
    # >> 'type': 'Feature'
    # >> 'properties': dict of all the different columns of properties {'OBJECTID_1': 2709255, 'SABL_PWSID': 'CA4900745', 'BOUNDARY_F': 'WBT Tool', 'BOUNDARY_T': 'Water Service Area', 'VERIFIED_S': 'Verified'........}
    # >> 'geometry': 'dict with keys 'type' and 'coordinates' or can also be null
    # >>> 'type': 'Polygon' or 'Multipolygon'
    # >>> 'coordinates': list of list of list of list if multipolygon, otherwise list of list of list if just polygon

    conn = wdc.sql_query_conn()
    ws_primary = pd.read_sql_query(
        "SELECT * from water_system_primary", conn)
    ws_score = pd.read_sql_query(
        "SELECT * from score_and_percentile_ave_ws", conn)
    conn.close()

    df_pws = pd.merge(ws_score, ws_primary, left_on='ws_id',
                      right_on='id', how='left')

    df_pws = df_pws[(df_pws['ave_red_lean_score'] != 'TBD') & (
        df_pws['ave_red_lean_score'] != 'PMD') & (df_pws['ave_red_lean_score'] != 'NA')]
    pwsid_list = df_pws['water_system_number'].values.tolist()

    # Let's say we want only a subset of water systems. Make a list.
    # This example list includes a "no shapefile", "no overlap", "overlap", and "null geometry"
    # pwsid_list = ['CA4200885', 'CA3600027', 'CA1910021', 'CA3010023']

    pre_pwsid_iteration = time.perf_counter()
    print(
        f'Time to prep for pwsid iteration (seconds): {pre_pwsid_iteration - start}')
    # We're going to create a smaller geojson of just these shapes
    comm_water_systems_with_shape_file = []
    subset_json = {
        'type': my_shape_geojson['type'], 'crs': my_shape_geojson['crs']}
    subset_list = []
    f_counter = 0  # deleting this after testing
    for f in my_shape_geojson['features']:
        f_counter += 1
        pwsid = f['properties']['SABL_PWSID']
        if pwsid in pwsid_list:
            comm_water_systems_with_shape_file.append(pwsid)
            subset_list += [f]
    subset_json['features'] = subset_list
    comm_water_systems_with_no_shape_file = list(
        set(pwsid_list) - set(comm_water_systems_with_shape_file))
    df_skipped = pd.DataFrame()
    if len(comm_water_systems_with_no_shape_file) > 0:
        for ws in comm_water_systems_with_no_shape_file:
            df_skipped = df_skipped.append(
                {'pwsid': ws, 'reason': 'No shapefile match.'}, ignore_index=True)
        print(comm_water_systems_with_no_shape_file)
        print(['No shapefile match.']*len(comm_water_systems_with_no_shape_file))
        print(df_skipped)

    print(f'f_counter: {f_counter}')

    # Here we make a list of the ACS fields that we will want to retrieve from the census
    acs_keys = {'B19001_001E': 'n_hh_income', 'B19001_002E': 'n_hh_income_lt_10k', 'B19001_003E': 'n_hh_income_10k_15k',
                'B19001_004E': 'n_hh_income_15k_20k', 'B19001_005E': 'n_hh_income_20k_25k', 'B19001_006E': 'n_hh_income_25k_30k',
                'B19001_007E': 'n_hh_income_30k_35k', 'B19001_008E': 'n_hh_income_35k_40k', 'B19001_009E': 'n_hh_income_40k_45k',
                'B19001_010E': 'n_hh_income_45k_50k', 'B19001_011E': 'n_hh_income_50k_60k', 'B19001_012E': 'n_hh_income_60k_75k',
                'B19001_013E': 'n_hh_income_75k_100k', 'B19001_014E': 'n_hh_income_100k_125k', 'B19001_015E': 'n_hh_income_125k_150k',
                'B19001_016E': 'n_hh_income_150k_200k', 'B19001_017E': 'n_hh_income_gt_200k',
                'B07412_001E': 'n_pov_lvl', 'B07412_002E': 'n_100pct_pov_lvl',
                'B07412_003E': 'n_101_149pct_pov_lvl', 'B07412_004E': 'n_150pct_pov_lvl',
                'B25032_001E': 'n_hh_housing_units', 'B25032_002E': 'n_hh_own', 'B25032_013E': 'n_hh_rent',
                'B08202_001E': 'hh_size', 'B08202_003E': 'hh_1worker', 'B08202_004E': 'hh_2worker', 'B08202_005E': 'hh_3+worker',
                'B08202_013E': 'n_hh_3ppl', 'B08202_018E': 'n_hh_4+ppl',
                'B11011_001E': 'n_hh_type', 'B11011_002E': 'n_hh_type_fam', 'B11011_003E': 'n_hh_type_fam_mcf', 'B11011_004E': 'n_hh_type_fam_mcf_1unit', 'B11011_005E': 'n_hh_type_fam_mcf_2unit',
                'B11011_006E': 'n_hh_type_fam_mcf_mh_and_other', 'B11011_007E': 'n_hh_type_fam_other', 'B11011_008E': 'n_hh_type_fam_other_mhh_nsp',
                'B11011_009E': 'n_hh_type_fam_other_mhh_nsp_1unit', 'B11011_010E': 'n_hh_type_fam_other_mhh_nsp_2unit', 'B11011_011E': 'n_hh_type_fam_other_mhh_nsp_mh_and_other',
                'B11011_012E': 'n_hh_type_fam_other_fhh_nsp', 'B11011_013E': 'n_hh_type_fam_other_fhh_nsp_1unit', 'B11011_014E': 'n_hh_type_fam_other_fhh_nsp_2unit',
                'B11011_015E': 'n_hh_type_fam_other_fhh_nsp_mh_and_other', 'B11011_016E': 'n_hh_type_nonfam', 'B11011_017E': 'n_hh_type_nonfam_1unit',
                'B11011_018E': 'n_hh_type_nonfam_2unit', 'B11011_019E': 'n_hh_type_nonfam_mh_and_other',
                'B02001_001E': 'n_race', 'B02001_002E': 'n_white_alone', 'B02001_003E': 'n_black_alone', 'B02001_004E': 'n_ai_and_an_alone',
                'B02001_005E': 'n_asian_alone', 'B02001_006E': 'n_nh_and_opi_alone', 'B02001_007E': 'n_other_alone', 'B02001_008E': 'n_two_or_more_races',
                'B15012_001E': 'n_bachelors_deg', 'B15012_002E': 'n_seng_compt_mat_stat_deg', 'B15012_003E': 'n_seng_bio_ag_env_deg', 'B15012_004E': 'n_seng_phys_sci_deg',
                'B15012_005E': 'n_seng_psych_deg', 'B15012_006E': 'n_seng_soc_sci_deg', 'B15012_007E': 'n_seng_eng_deg', 'B15012_008E': 'n_seng_mds_deg',
                'B15012_009E': 'n_seng_rltd_deg', 'B15012_010E': 'n_bus_deg', 'B15012_011E': 'n_edu_deg', 'B15012_012E': 'n_aho_lit_lang_deg',
                'B15012_013E': 'n_aho_lib_arts_and_hist_deg', 'B15012_014E': 'n_aho_vis_perf_art_deg', 'B15012_015E': 'n_aho_comm_deg',
                'B15012_016E': 'n_aho_other_deg',
                'B992701_001E': 'n_insurance', 'B992701_002E': 'n_have_insurance', 'B992701_003E': 'n_no_insurance',
                'B25070_001E': 'n_rent_as_pct', 'B25070_002E': 'n_rent_lt_10pct', 'B25070_003E': 'n_rent_10_14.9pct', 'B25070_004E': 'n_rent_15_19.9pct',
                'B25070_005E': 'n_rent_20_24.9pct', 'B25070_006E': 'n_rent_25_29.9pct', 'B25070_007E': 'n_rent_30_34.9pct', 'B25070_008E': 'n_rent_35_39.9pct',
                'B25070_009E': 'n_rent_40_49.9pct', 'B25070_010E': 'n_rent_gt_50pct', 'B25070_011E': 'n_rent_not_computed'}
    vlist = tuple(['NAME']+list(acs_keys.keys()))
    # Note on good ACS groups to include:
    # B25032 was missing question 1E, 1 field added
    # B02001 includes race (black, white, asian, american indian/alaskan native,pacific islander, etc), 8 fields added
    # B15012 includes what was their college major (of course not everyone goes to college so will have to adjust numbers to a percentage), 16 fields added
    # B992701 includes health insurance allocation, 3 fields added
    # B25070 is rent as a percentage of household income, 11 fields added
    # B11011 was in it but didn't have all the keys. Now has 19 keys
    # Initially had 28 fields, now 28+8+16+3+11 == 66 fields

    # Iterate over each water system boundary in the subset geojson
    start = time.perf_counter()
    total_counter = 0
    no_overlap_counter = 0
    ws_no_overlap = []
    null_geometry_counter = 0
    ws_null_geometry = []
    df_store = pd.DataFrame()

    old_len_df_store = len(df_store)
    for system in subset_json['features']:
        total_counter += 1
        overlap = overlap_generator(system, vlist, c)
        old_len_df_store = len(df_store)
        water_sytem_id = system['properties']['SABL_PWSID']
        if overlap != 'null geometry':
            try:
                df_store = df_store.append(overlap, ignore_index=True)
            except:
                print('error')
                print(overlap)
                print(df_store)
                raise ValueError
        if overlap == None:
            no_overlap_counter += 1
            df_skipped = df_skipped.append(
                {'pwsid': water_sytem_id, 'reason': 'No overlap with census tracts.'}, ignore_index=True)
            if len(df_store) == old_len_df_store:
                print(f'No overlap with df_store staying the same')
            else:
                print(
                    f'!!! No overlap but df_store incrased by {len(df_store) - old_len_df_store}')
        elif overlap == 'null geometry':
            null_geometry_counter += 1
            df_skipped = df_skipped.append(
                {'pwsid': water_sytem_id, 'reason': 'Null geometry in shapefile.'}, ignore_index=True)
            print(
                f'$$$ Null geometry at water system: {water_sytem_id}')
        else:
            print(
                f'Overlap with df_store increase of: {len(df_store) - old_len_df_store}')
        current_time = time.perf_counter()
        print(
            f'Time so far: {current_time-start}. Counter: {total_counter}. Average time per system: {(current_time-start)/total_counter}.')
        print(
            f'Total: {total_counter}. No overlap: {no_overlap_counter}. Null geometry: {null_geometry_counter}.')
    comm_water_systems_with_no_shape_file_dict = {'pwsid': comm_water_systems_with_no_shape_file, 'reason': [
        'No shapefile match.']*len(comm_water_systems_with_no_shape_file)}

    df_final = df_store.rename(columns=acs_keys)
    df_final.drop(['COUNTY', 'STATE', 'TRACT'], axis=1, inplace=True)
    df_final.columns = df_final.columns.str.lower()
    print(df_final)
    print(df_final.columns.tolist())
    print(df_skipped)
    conn = wdc.sql_query_conn()
    df_final.to_sql('census_tract_overlap', conn,
                    if_exists='replace', index=False)
    df_skipped.to_sql('census_tract_overlap_skipped', conn,
                      if_exists='replace', index=False)
    conn.close()
    print(time.perf_counter() - start)

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
