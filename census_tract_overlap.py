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
    # filepath to folder with SABL shape file for Sarmad
    os.chdir(
        r'C:\Users\sarma\Dropbox\Water Data Challenge 2022\GIS\public_water_systems')
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
    api_key = 'ad3130119c535c4902b79422cdacefe1bd3a7190'  # API key for Sarmad
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

# 'coordinates': [[[[-118.24620330993622, 33.92047127986991], [-118.24621147023227, 33.91729464978054], [-118.24894969960994, 33.91729078009003], [-118.25164818033295, 33.91728800995508], [-118.25434867957038, 33.91728356998626], [-118.25434807141094, 33.91689355032028], [-118.25377698992695, 33.916887088629736], [-118.25378281999315, 33.91637891032095], [-118.25454570975661, 33.91637655016457], [-118.25455355025241, 33.915163330197885], [-118.25463321015685, 33.91510939022284], [-118.25473887000057, 33.909236289692466], [-118.26091094009452, 33.9091700700284], [-118.26094341958193, 33.906272299872285], [-118.25867218928617, 33.9062812197261], [-118.25867205004731, 33.906117639716626], [-118.2576230199164, 33.90611993007452], [-118.25763481030451, 33.904386960002604], [-118.25709003957685, 33.904405951270526], [-118.25700901063992, 33.90337537064241], [-118.25550417982541, 33.903439020311176], [-118.25548931001254, 33.90436085967711], [-118.25448325977638, 33.90435682015392], [-118.25447743959165, 33.90484738969467], [-118.25432540961127, 33.90615074999519], [-118.25059658012202, 33.906154459896385], [-118.25059792040842, 33.90623237002017], [-118.24551477013556, 33.90629838992031], [-118.24561838002379, 33.90340192980312], [-118.24563685028434, 33.90326004971905], [-118.24493947016299, 33.90333129026079], [-118.24188502931372, 33.903338119806214], [-118.24170317975748, 33.9006489902274], [-118.24162937956363, 33.89957482988902], [-118.24653782009014, 33.899545890273224], [-118.24682212969104, 33.90317571962942], [-118.25253557030658, 33.90291934016627], [-118.25248595006522, 33.90227152010698], [-118.25317661977142, 33.90217801972896], [-118.25366442024206, 33.902839720364], [-118.25328891008027, 33.901695450005676], [-118.25232961984959, 33.89770716029792], [-118.25870197921762, 33.897485859932445], [-118.25954149078326, 33.90110339028371], [-118.26054866032023, 33.9010733387466], [-118.26082629993013, 33.902221030034866], [-118.26090677011497, 33.90248816973065], [-118.26091916956086, 33.90555016064573], [-118.26231590969049, 33.90552900973275], [-118.26301421956815, 33.90820205988484], [-118.26304104056759, 33.90845804994585], [-118.26305687966267, 33.90913834015832], [-118.26531226996507, 33.9091165899816], [-118.26532221970513, 33.90556891009175], [-118.26576880996213, 33.905576699744316], [-118.2658707301193, 33.90559227979279], [-118.26609335959618, 33.90568689029738], [-118.26625294979799, 33.90551770026612], [-118.26746423992371, 33.90551993024938], [-118.2674587602005, 33.90370973124803], [-118.2693684895613, 33.90371863869519], [-118.26963469990586, 33.90496972875126], [-118.26962933965856, 33.905530720057406], [-118.27399061005731, 33.90552626978364], [-118.27398929941532, 33.910200810223444], [-118.27183280015099, 33.910205260253086], [-118.27181671042592, 33.9091679392964], [-118.26962044993053, 33.909146839977325], [-118.26959500964166, 33.91278517002909], [-118.26530080027551, 33.91276291011477], [-118.26528697071171, 33.91155987010389], [-118.26309561142449, 33.9115754498043], [-118.26307893959113, 33.9163408697512], [-118.25614022028422, 33.91635683025807], [-118.25616343005625, 33.91913496028508], [-118.25612341999177, 33.92309732018482], [-118.25362091932443, 33.92311290001209], [-118.2536262885549, 33.92372718019349], [-118.25357263936951, 33.92372718019349], [-118.25356995969499, 33.924190109836175], [-118.25350826968949, 33.92424130024287], [-118.25336343072297, 33.92423685020071], [-118.25336611039747, 33.92433699994536], [-118.2534090292068, 33.92441045012274], [-118.25347876991388, 33.92443716000787], [-118.25358136021427, 33.924438269906936], [-118.25355052015226, 33.9272995497675], [-118.25374832019467, 33.9272995497675], [-118.25374892026926, 33.92731498869594], [-118.25378168991253, 33.927315398651515], [-118.2537788494396, 33.92747536987518], [-118.25384473996735, 33.92870664010923], [-118.2541619997729, 33.92903851991354], [-118.25424322993246, 33.92946386971618], [-118.24927577997326, 33.92943494024876], [-118.24491986982179, 33.929492799919196], [-118.24355730969042, 33.92949726012168], [-118.23915178019594, 33.929424930127496], [-118.23915446076877, 33.92048351988971], [-118.24620330993622, 33.92047127986991]]], [[[-118.25224797017678, 33.898019620866386], [-118.25217196731386, 33.89802113524096], [-118.25154664285807, 33.89802541068346], [-118.2510604144392, 33.895838435716755], [-118.25192676676718, 33.895827303920456], [-118.25151382380749, 33.8942504381176], [-118.2516211122968, 33.894205910094556], [-118.25164256855736, 33.89409458918948], [-118.25128851824938, 33.892607327034504], [-118.25131534014714, 33.89248709766031], [-118.25139044199983, 33.892469285638185], [-118.2532182477017, 33.892436812076056], [-118.25318972798806, 33.89235326766785], [-118.25293237143896, 33.89126581084635], [-118.25426728771407, 33.89124209718434], [-118.25424778528925, 33.891159027356125], [-118.25327178908925, 33.887038686962455], [-118.25046923494732, 33.88826644968996], [-118.2501741922754, 33.888351059253], [-118.2499381581379, 33.888337701266835], [-118.24953046223777, 33.88837777671051], [-118.24251645063534, 33.88842732270251], [-118.24251194827913, 33.88834453872455], [-118.24239308409905, 33.8866013033472], [-118.2434847806127, 33.88660282911049], [-118.24338634142723, 33.88517503532517], [-118.24339707809152, 33.88508255427648], [-118.24333806955714, 33.884975675047855], [-118.24330856439161, 33.884790862312975], [-118.24322429972302, 33.884793491066496], [-118.24227900165121, 33.884792239705234], [-118.24118656402746, 33.884790308223934], [-118.24111507340223, 33.88372993029973], [-118.24111116213747, 33.883653113028956], [-118.2409118601119, 33.880623124913406], [-118.24390538081188, 33.880544625342566], [-118.24417121655895, 33.884795119775035], [-118.24426274769989, 33.88479539868392], [-118.24916994495806, 33.88480057863308], [-118.24968214457012, 33.88700088982953], [-118.24957073640691, 33.88701676335968], [-118.24916203529196, 33.88706546896339], [-118.24914506791288, 33.88712352593518], [-118.2489133924011, 33.8871519142013], [-118.24891487102806, 33.88721755364111], [-118.24887262595519, 33.88722284231893], [-118.24904027763837, 33.88792073537089], [-118.24927027779212, 33.887881770087624], [-118.24926290172534, 33.88785727769996], [-118.24988079621541, 33.88785226124888], [-118.25000093779985, 33.88783014023955], [-118.24925059480594, 33.884779626907616], [-118.24888045028132, 33.88313187970226], [-118.24881874859773, 33.88047208525525], [-118.24942090460402, 33.88044647728536], [-118.24941822313289, 33.88037299295169], [-118.2612491081883, 33.88002792232886], [-118.26147429427047, 33.880935563049675], [-118.25779420557187, 33.883986415928646], [-118.25694796921805, 33.884012579167546], [-118.25695829265727, 33.88407312529978], [-118.25747934067489, 33.88627762614121], [-118.257969971737, 33.886257205740286], [-118.25800459370635, 33.88639057101213], [-118.25946303372295, 33.89265985626009], [-118.257621081352, 33.89271102533944], [-118.25754028058715, 33.89236240671631], [-118.25653378119337, 33.892385228980174], [-118.25655280032456, 33.892455371274984], [-118.25727416366738, 33.8956341626612], [-118.25691154343158, 33.89570860915993], [-118.25665405033854, 33.895733100038726], [-118.25626513012617, 33.895744231847324], [-118.25176839557922, 33.895853896749664], [-118.25224797017678, 33.898019620866386]]], [[[-118.2391588885648, 33.91714465013144], [-118.23919373062138, 33.91628530986916], [-118.24265041998692, 33.91635652983376], [-118.24264823977572, 33.91608702125256], [-118.2426482505555, 33.914652799296825], [-118.23924183989642, 33.91465404872893], [-118.23915601036262, 33.91465571935981], [-118.23915633016287, 33.913713339689515], [-118.2389604902444, 33.91370714986275], [-118.23605337040678, 33.91373917995642], [-118.23605866058547, 33.9087590502476], [-118.23605802008669, 33.907077049745446], [-118.23622833976623, 33.90702250970262], [-118.23645364981591, 33.906995800322846], [-118.23882176970328, 33.906965979747135], [-118.23914949937344, 33.907039509761795], [-118.23911267024343, 33.90595835009472], [-118.24331773978392, 33.9059553700889], [-118.24363825957563, 33.90632101013611], [-118.24425283029916, 33.90631524995911], [-118.24502421992183, 33.90664562000946], [-118.24500141978162, 33.90674691025212], [-118.24445228144506, 33.90679717025006], [-118.24427794000799, 33.907072090341224], [-118.2447403900213, 33.90762604998761], [-118.24475083024154, 33.90739779032784], [-118.24621837019197, 33.909142120009534], [-118.2460145199961, 33.90914101065676], [-118.24821052986158, 33.91176334011083], [-118.24816090962021, 33.91176223004665], [-118.2482621596342, 33.91195867017458], [-118.24837139028118, 33.912243989823445], [-118.24903907030264, 33.914649710754254], [-118.24506867035873, 33.91465070970358], [-118.24506802985992, 33.917296010247654], [-118.24619451992116, 33.917296010247654], [-118.24620390012936, 33.91782019028359], [-118.24506577958013, 33.91783020027763], [-118.24504298033824, 33.917822410254104], [-118.24096668968359, 33.91783242993883], [-118.24097474038516, 33.91738727033377], [-118.23966314078154, 33.917393940715336], [-118.23966314078154, 33.91714242865237], [-118.2391588885648, 33.91714465013144]]]]

#     # pwsid_list = ['CA4810701']
#          AREALAND  AREAWATER n_100pct_pov_lvl n_101_149pct_pov_lvl n_150pct_pov_lvl  hh_size  hh_1worker  hh_2worker  hh_3+worker  n_hh_3ppl  ...  SABL_PWSID  STATE  STATE_CLAS   TRACT    UR                               WATER_SY_1  county  proportion  state   tract
# 0   4164175.0     4279.0             None                 None             None   1176.0       327.0       480.0        225.0      332.0  ...   CA4810701     06   COMMUNITY  252706  None  CALIFORNIA WATER SERVICE CO.-TRAVIS AFB     095    0.341018     06  252706
# 1   2552436.0    10113.0             None                 None             None    758.0       430.0       314.0          0.0      176.0  ...   CA4810701     06   COMMUNITY  252801  None  CALIFORNIA WATER SERVICE CO.-TRAVIS AFB     095    0.883196     06  252801
# 2   3603664.0        0.0             None                 None             None    158.0        74.0        79.0          5.0       62.0  ...   CA4810701     06   COMMUNITY  252802  None  CALIFORNIA WATER SERVICE CO.-TRAVIS AFB     095    0.996825     06  252802
# 3  13952563.0   167451.0             None                 None             None      0.0         0.0         0.0          0.0        0.0  ...   CA4810701     06   COMMUNITY  980000  None  CALIFORNIA WATER SERVICE CO.-TRAVIS AFB     095    0.997367     06  980000

    conn = wdc.sql_query_conn()
    ws_primary = pd.read_sql_query(
        "SELECT * from water_system_primary", conn)
    ws_score = pd.read_sql_query(
        "SELECT * from score_and_percentile_ave_ws", conn)
    conn.close()

    df_pws = pd.merge(ws_score, ws_primary, left_on='ws_id',
                      right_on='id', how='left')
    # print(df_pws)

    df_pws = df_pws[(df_pws['ave_red_lean_score'] != 'TBD') & (
        df_pws['ave_red_lean_score'] != 'PMD') & (df_pws['ave_red_lean_score'] != 'NA')]
    pwsid_list = df_pws['water_system_number'].values.tolist()

    # Let's say we want only a subset of water systems. Make a list.
    # This example list includes a "no shapefile", "no overlap", "overlap", and "null geometry"
    # pwsid_list = ['CA4200885', 'CA3600027', 'CA1910021', 'CA3010023']

    # print(len(pwsid_list))
    # raise ValueError

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

    # print(acs_keys.keys())
    # print(type(acs_keys.keys()))
    # print(len(acs_keys))
    # raise ValueError

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
            # ws_no_overlap.append(water_sytem_id)
            if len(df_store) == old_len_df_store:
                print(f'No overlap with df_store staying the same')
            else:
                print(
                    f'!!! No overlap but df_store incrased by {len(df_store) - old_len_df_store}')
        elif overlap == 'null geometry':
            null_geometry_counter += 1
            df_skipped = df_skipped.append(
                {'pwsid': water_sytem_id, 'reason': 'Null geometry in shapefile.'}, ignore_index=True)
            # ws_null_geometry.append(water_sytem_id)
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
    # print(comm_water_systems_with_no_shape_file_dict)
    # no_overlap_dict = {'pwsid': ws_no_overlap, 'reason': [
    #     'No overlap with census tracts.']*len(ws_no_overlap)}
    # print(no_overlap_dict)
    # null_geometry_dict = {'pwsid': ws_null_geometry, 'reason': [
    #     'Null geometry in shapefil.']*len(ws_null_geometry)}
    # print(null_geometry_dict)

    # if len(ws_no_overlap) > 0:
    #     df_skipped = df_skipped.append(no_overlap_dict, ignore_index=True)
    # if len(ws_null_geometry) > 0:
    #     df_skipped = df_skipped.append(null_geometry_dict, ignore_index=True)
    df_final = df_store.rename(columns=acs_keys)
    # print(df_final)
    # print(df_final.columns.tolist())
    # county_state_tract__df = df_final[[
    #     'COUNTY', 'county', 'STATE', 'state', 'TRACT', 'TRACT']]
    # print(county_state_tract__df)
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

# Time so far: 5127.1588622. Counter: 2268. Average time per system: 2.260652055643739.
# Total: 2268. No overlap: 1271. Null geometry: 5.


#     raise ValueError
# raise ValueError

# # Output to excel
# # df_final.to_excel('acs_SABL_overlaps.xlsx', index=False)
# # # Or, output to csv
# # df_final.to_csv('acs_SABL_overlaps.csv', index=False)


# # OPTION 2 (not fully tested): Create a list of geojsons (possibly able to re-export to shapefile?)

# # List to store the resulting overlap jsons
# store_jsons = []
# for n in range(0, len(subset_list)):
#     system_area = subset_json['features'][n]
#     # Generate the overlap
#     overlap_features = c.acs5.geo_tract(vlist, system_area['geometry'], 2020)
#     # Create the features list for the overlaps
#     features = []
#     for tract_geojson, tract_data, tract_proportion in overlap_features:
#         tract_geojson['properties'].update(tract_data)
#         tract_geojson['properties'].update({'proportion': tract_proportion})
#         for k in ['SABL_PWSID', 'WATER_SY_1', 'REGULATING', 'STATE_CLAS', 'POPULATION']:
#             tract_geojson['properties'].update(
#                 {k: system_area['properties'][k]})
#         features.append(tract_geojson)
#     # Create a new geojson and add the features to it
#     overlap_geojson = {
#         'type': 'FeatureCollection',
#         'crs': my_shape_geojson['crs'],
#         'features': features
#     }
#     # Store the new geojson in the list
#     store_jsons += [overlap_geojson]
