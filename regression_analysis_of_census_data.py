# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:12:19 2022

@author: sarma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wdc_lib as wdc
import time
import cProfile
import pstats
import io
from pstats import SortKey


pr = cProfile.Profile()
pr.enable()
start = time.perf_counter()


# Importing the dataset
conn = wdc.sql_query_conn()
df_census = pd.read_sql_query(
    "SELECT * from census_tract_overlap", conn)
df_wsp = pd.read_sql_query(
    "SELECT * from water_system_primary", conn)
df_ws_compliance = pd.read_sql_query(
    "SELECT * from score_and_percentile_ave_ws", conn)
df_ws_overage = pd.read_sql_query(
    "SELECT * from overage_count_and_percentile_ws", conn)
conn.close()

df_ws_compliance_and_overage = pd.merge(df_ws_compliance, df_ws_overage, left_on='ws_id', right_on='ws_id', how='left')
df_wsp_and_scores = pd.merge(df_ws_compliance_and_overage, df_wsp, left_on='ws_id', right_on='id', how='left')
df_wsp_score_census = pd.merge(df_census, df_wsp_and_scores, left_on='sabl_pwsid', right_on='water_system_number', how='left')
df_wsp_score_census = df_wsp_score_census[(df_wsp_score_census['ave_red_lean_score'] != 'PMD') & (df_wsp_score_census['ave_red_lean_score'] != 'TBD') & (df_wsp_score_census['ave_red_lean_score'] != 'NA')]

df_wsp_score_census.drop(['n_100pct_pov_lvl', 'n_101_149pct_pov_lvl', 'n_150pct_pov_lvl','id','pserved','type','primary_source_water_type','ur','water_sy_1', 'pop100'], axis=1, inplace=True)
df_wsp_score_census = df_wsp_score_census[['n_race', 'n_white_alone', 'n_black_alone', 'n_ai_and_an_alone', 'n_asian_alone', 'n_nh_and_opi_alone', 'n_other_alone', 'n_two_or_more_races', 
                                           'hh_size', 'hh_1worker', 'hh_2worker', 'hh_3+worker', 'n_hh_3ppl', 'n_hh_4+ppl', 
                                           'n_hh_type', 'n_hh_type_fam', 'n_hh_type_fam_mcf', 'n_hh_type_fam_mcf_1unit', 'n_hh_type_fam_mcf_2unit', 'n_hh_type_fam_mcf_mh_and_other', 'n_hh_type_fam_other', 'n_hh_type_fam_other_mhh_nsp', 'n_hh_type_fam_other_mhh_nsp_1unit', 'n_hh_type_fam_other_mhh_nsp_2unit', 'n_hh_type_fam_other_mhh_nsp_mh_and_other', 'n_hh_type_fam_other_fhh_nsp', 'n_hh_type_fam_other_fhh_nsp_1unit', 'n_hh_type_fam_other_fhh_nsp_2unit', 'n_hh_type_fam_other_fhh_nsp_mh_and_other', 'n_hh_type_nonfam', 'n_hh_type_nonfam_1unit', 'n_hh_type_nonfam_2unit', 'n_hh_type_nonfam_mh_and_other',
                                           'n_bachelors_deg', 'n_seng_compt_mat_stat_deg', 'n_seng_bio_ag_env_deg', 'n_seng_phys_sci_deg', 'n_seng_psych_deg', 'n_seng_soc_sci_deg', 'n_seng_eng_deg', 'n_seng_mds_deg', 'n_seng_rltd_deg', 'n_bus_deg', 'n_edu_deg', 'n_aho_lit_lang_deg', 'n_aho_lib_arts_and_hist_deg', 'n_aho_vis_perf_art_deg', 'n_aho_comm_deg', 'n_aho_other_deg', 
                                           'n_hh_income', 'n_hh_income_lt_10k', 'n_hh_income_10k_15k', 'n_hh_income_15k_20k', 'n_hh_income_20k_25k', 'n_hh_income_25k_30k', 'n_hh_income_30k_35k', 'n_hh_income_35k_40k', 'n_hh_income_40k_45k', 'n_hh_income_45k_50k', 'n_hh_income_50k_60k', 'n_hh_income_60k_75k', 'n_hh_income_75k_100k', 'n_hh_income_100k_125k', 'n_hh_income_125k_150k', 'n_hh_income_150k_200k', 'n_hh_income_gt_200k', 
                                           'n_hh_housing_units', 'n_hh_own', 'n_hh_rent', 
                                           'n_rent_as_pct', 'n_rent_lt_10pct', 'n_rent_10_14.9pct', 'n_rent_15_19.9pct', 'n_rent_20_24.9pct', 'n_rent_25_29.9pct', 'n_rent_30_34.9pct', 'n_rent_35_39.9pct', 'n_rent_40_49.9pct', 'n_rent_gt_50pct', 'n_rent_not_computed', 
                                           'n_insurance', 'n_have_insurance', 'n_no_insurance', 
                                           'number_gw', 'number_sw',
                                           'ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches',
                                           'regulating',
                                           'arealand', 'areawater',
                                           'population',
                                           'basename', 'centlat', 'centlon', 'funcstat', 'geoid', 'geo_id', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'mtfcc', 'name', 'objectid', 'oid', 'sabl_pwsid', 'state_clas', 'county', 'proportion', 'state', 'tract', 'water_system_number', 
                                           'water_system_name','ws_id', 'water_system_number', 'water_system_name','ave_red_lean_score', 'ave_score_red_lean_percentile', 'ave_overage_rate', 'overage_percentile']]

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def linear_regression(X,y):
    # # # Multiple Linear Regression:
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # Training the Multiple Linear Regression model on the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2) # precision argument indicates number of digits of precision for floating point output (default 8)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    # Getting the final linear regression equation with the values of the coefficients
    # print(regressor.coef_)
    # print(regressor.intercept_)    
    return r2_score(y_test, y_pred)

def polynomial_regression(X,y):
    # # # Polynomial Regression:
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    
    # Training the Polynomial Regression model on the Training dataset
    poly_reg = PolynomialFeatures(degree = 3)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(poly_reg.transform(X_test))
    np.set_printoptions(precision=2)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))    
    return r2_score(y_test, y_pred)

def support_vector_regression(X,y):
    # # # SVR:
    y = y.reshape(len(y),1)
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    # Training the SVR model on the Training set
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    # # Predicting the Test set results
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
    np.set_printoptions(precision=2)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))  
    return r2_score(y_test, y_pred)
    
def decision_tree_regression(X,y):
    # # # Decision Tree:
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # Training the Decision Tree Regression model on the Training set
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))  
    return r2_score(y_test, y_pred)
    
def random_forest_regression(X,y):
    # # # Random Forest:
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # Training the Random Forest Regression model on the whole dataset
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))    
    return r2_score(y_test, y_pred)


# Converting to fractions as some census questions may have varying answer rates
dataset_columns = ['fract_white_alone','fract_black_alone','fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races',
                   'fract_hh_1worker', 'fract_hh_2worker', 'fract_hh_3+worker', 'fract_hh_3ppl', 'fract_hh_4+ppl',
                   'fract_hh_type_fam', 'fract_hh_type_fam_mcf', 'fract_hh_type_fam_mcf_1unit', 'fract_hh_type_fam_mcf_2unit', 'fract_hh_type_fam_mcf_mh_and_other', 'fract_hh_type_fam_other', 'fract_hh_type_fam_other_mhh_nsp', 'fract_hh_type_fam_other_mhh_nsp_1unit', 'fract_hh_type_fam_other_mhh_nsp_2unit', 'fract_hh_type_fam_other_mhh_nsp_mh_and_other', 'fract_hh_type_fam_other_fhh_nsp', 'fract_hh_type_fam_other_fhh_nsp_1unit', 'fract_hh_type_fam_other_fhh_nsp_2unit', 'fract_hh_type_fam_other_fhh_nsp_mh_and_other', 'fract_hh_type_nonfam', 'fract_hh_type_nonfam_1unit', 'fract_hh_type_nonfam_2unit', 'fract_hh_type_nonfam_mh_and_other',
                   'fract_seng_compt_mat_stat_deg', 'fract_seng_bio_ag_env_deg', 'fract_seng_phys_sci_deg', 'fract_seng_psych_deg', 'fract_seng_soc_sci_deg', 'fract_seng_eng_deg', 'fract_seng_mds_deg', 'fract_seng_rltd_deg', 'fract_bus_deg', 'fract_edu_deg', 'fract_aho_lit_lang_deg', 'fract_aho_lib_arts_and_hist_deg', 'fract_aho_vis_perf_art_deg', 'fract_aho_comm_deg', 'fract_aho_other_deg',
                   'fract_hh_income_lt_10k', 'fract_hh_income_10k_15k', 'fract_hh_income_15k_20k', 'fract_hh_income_20k_25k', 'fract_hh_income_25k_30k', 'fract_hh_income_30k_35k', 'fract_hh_income_35k_40k', 'fract_hh_income_40k_45k', 'fract_hh_income_45k_50k', 'fract_hh_income_50k_60k', 'fract_hh_income_60k_75k', 'fract_hh_income_75k_100k', 'fract_hh_income_100k_125k', 'fract_hh_income_125k_150k', 'fract_hh_income_150k_200k', 'fract_hh_income_gt_200k',
                   'fract_hh_own', 'fract_hh_rent',
                   'fract_rent_lt_10pct', 'fract_rent_10_14.9pct', 'fract_rent_15_19.9pct', 'fract_rent_20_24.9pct', 'fract_rent_25_29.9pct', 'fract_rent_30_34.9pct', 'fract_rent_35_39.9pct', 'fract_rent_40_49.9pct', 'fract_rent_gt_50pct', 'fract_rent_not_computed', 
                   'fract_have_insurance', 'fract_no_insurance',
                   'num_gw', 'num_sw', 
                   'ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches',
                   'regulating',
                   'arealand', 'areawater',
                   'population',
                   'ws_id', 'water_system_number','water_system_name',
                   'ave_red_lean_score', 'ave_score_red_lean_percentile', 
                   'ave_overage_rate', 'overage_percentile']

data_array = []
for i,j in df_wsp_score_census.iterrows():
    max_answers = max(j['n_race'], j['hh_size'], j['n_hh_type'],j['n_bachelors_deg'], j['n_hh_income'], j['n_hh_housing_units'],j['n_rent_as_pct'], j['n_insurance'])
    
    race_fractions = [None]*7
    hh_size_fractions = [None]*5
    hh_type_fractions = [None]*18
    bdeg_fractions = [None]*15
    hh_income_fractions = [None]*16
    hh_ownership_fractions = [None]*2
    rent_as_pct_fractions = [None]*10
    insurance_fractions = [None]*2
    gw_sw = [None]*2
    timeline_characteristics = [None]*4
    regulator = [None]*1
    area = [None]*2
    population = [None]*1
    identity = [None]*3
    compliance = [None]*2
    overage = [None]*2
    
    if j['n_race'] > 0:
        race_fractions = [j['n_white_alone']/j['n_race'], j['n_black_alone']/j['n_race'], j['n_ai_and_an_alone']/j['n_race'], j['n_asian_alone']/j['n_race'], j['n_nh_and_opi_alone']/j['n_race'], j['n_other_alone']/j['n_race'], j['n_two_or_more_races']/j['n_race']]
    if j['hh_size'] > 0:
        hh_size_fractions = [j['hh_1worker']/j['hh_size'], j['hh_2worker']/j['hh_size'], j['hh_3+worker']/j['hh_size'], j['n_hh_3ppl']/j['hh_size'], j['n_hh_4+ppl']/j['hh_size']]
    if j['n_hh_type'] > 0:
        hh_type_fractions = [j['n_hh_type_fam']/j['n_hh_type'], j[ 'n_hh_type_fam_mcf']/j['n_hh_type'], j[ 'n_hh_type_fam_mcf_1unit']/j['n_hh_type'], j[ 'n_hh_type_fam_mcf_2unit']/j['n_hh_type'], j[ 'n_hh_type_fam_mcf_mh_and_other']/j['n_hh_type'], j[ 'n_hh_type_fam_other']/j['n_hh_type'], j[ 'n_hh_type_fam_other_mhh_nsp']/j['n_hh_type'], j[ 'n_hh_type_fam_other_mhh_nsp_1unit']/j['n_hh_type'], j[ 'n_hh_type_fam_other_mhh_nsp_2unit']/j['n_hh_type'], j[ 'n_hh_type_fam_other_mhh_nsp_mh_and_other']/j['n_hh_type'], j[ 'n_hh_type_fam_other_fhh_nsp']/j['n_hh_type'], j[ 'n_hh_type_fam_other_fhh_nsp_1unit']/j['n_hh_type'], j[ 'n_hh_type_fam_other_fhh_nsp_2unit']/j['n_hh_type'], j[ 'n_hh_type_fam_other_fhh_nsp_mh_and_other']/j['n_hh_type'], j[ 'n_hh_type_nonfam']/j['n_hh_type'], j[ 'n_hh_type_nonfam_1unit']/j['n_hh_type'], j[ 'n_hh_type_nonfam_2unit']/j['n_hh_type'], j[ 'n_hh_type_nonfam_mh_and_other']/j['n_hh_type']]
    if j['n_bachelors_deg'] > 0:
        bdeg_fractions = [j['n_seng_compt_mat_stat_deg']/j['n_bachelors_deg'], j[ 'n_seng_bio_ag_env_deg']/j['n_bachelors_deg'], j[ 'n_seng_phys_sci_deg']/j['n_bachelors_deg'], j[ 'n_seng_psych_deg']/j['n_bachelors_deg'], j[ 'n_seng_soc_sci_deg']/j['n_bachelors_deg'], j[ 'n_seng_eng_deg']/j['n_bachelors_deg'], j[ 'n_seng_mds_deg']/j['n_bachelors_deg'], j[ 'n_seng_rltd_deg']/j['n_bachelors_deg'], j[ 'n_bus_deg']/j['n_bachelors_deg'], j[ 'n_edu_deg']/j['n_bachelors_deg'], j[ 'n_aho_lit_lang_deg']/j['n_bachelors_deg'], j[ 'n_aho_lib_arts_and_hist_deg']/j['n_bachelors_deg'], j[ 'n_aho_vis_perf_art_deg']/j['n_bachelors_deg'], j[ 'n_aho_comm_deg']/j['n_bachelors_deg'], j[ 'n_aho_other_deg']/j['n_bachelors_deg']]
        bdeg_answer_rate = j['n_bachelors_deg']/max_answers
        bdeg_fractions = [bdeg*bdeg_answer_rate for bdeg in bdeg_fractions] # Necessary b/c this was the only census question with no negative answer
    if j['n_hh_income'] > 0:
        hh_income_fractions = [j['n_hh_income_lt_10k']/j['n_hh_income'], j[ 'n_hh_income_10k_15k']/j['n_hh_income'], j[ 'n_hh_income_15k_20k']/j['n_hh_income'], j[ 'n_hh_income_20k_25k']/j['n_hh_income'], j[ 'n_hh_income_25k_30k']/j['n_hh_income'], j[ 'n_hh_income_30k_35k']/j['n_hh_income'], j[ 'n_hh_income_35k_40k']/j['n_hh_income'], j[ 'n_hh_income_40k_45k']/j['n_hh_income'], j[ 'n_hh_income_45k_50k']/j['n_hh_income'], j[ 'n_hh_income_50k_60k']/j['n_hh_income'], j[ 'n_hh_income_60k_75k']/j['n_hh_income'], j[ 'n_hh_income_75k_100k']/j['n_hh_income'], j[ 'n_hh_income_100k_125k']/j['n_hh_income'], j[ 'n_hh_income_125k_150k']/j['n_hh_income'], j[ 'n_hh_income_150k_200k']/j['n_hh_income'], j[ 'n_hh_income_gt_200k']/j['n_hh_income']]
    if j['n_hh_housing_units'] > 0:
        hh_ownership_fractions = [j['n_hh_own']/j['n_hh_housing_units'], j['n_hh_rent']/j['n_hh_housing_units']]
    if j['n_rent_as_pct'] > 0:
        rent_as_pct_fractions = [j['n_rent_lt_10pct']/j['n_rent_as_pct'], j[ 'n_rent_10_14.9pct']/j['n_rent_as_pct'], j[ 'n_rent_15_19.9pct']/j['n_rent_as_pct'], j[ 'n_rent_20_24.9pct']/j['n_rent_as_pct'], j[ 'n_rent_25_29.9pct']/j['n_rent_as_pct'], j[ 'n_rent_30_34.9pct']/j['n_rent_as_pct'], j[ 'n_rent_35_39.9pct']/j['n_rent_as_pct'], j[ 'n_rent_40_49.9pct']/j['n_rent_as_pct'], j[ 'n_rent_gt_50pct']/j['n_rent_as_pct'], j[ 'n_rent_not_computed']/j['n_rent_as_pct']]
        rent_rate = j['n_rent_as_pct']*(j['n_hh_rent']/j['n_hh_housing_units'])
        rent_as_pct_fractions = [rpct*rent_rate for rpct in rent_as_pct_fractions] # Necessary b/c own vs rent ratio can vary      
    if j['n_insurance'] > 0:
        insurance_fractions = [j['n_have_insurance']/j['n_insurance'], j['n_no_insurance']/j['n_insurance']]
    gw_sw = [j['number_gw'], j['number_sw']]
    timeline_characteristics = [j['ave_target_timeline'], j['ave_method_priority_level'], j['ave_num_time_segments'], j['ave_num_track_switches']]
    regulator = [j['regulating']]
    area = [j['arealand'], j['areawater']]
    population = [j['population']]
    identity = [j['ws_id'], j['water_system_number'], j['water_system_name']]
    compliance = [float(j['ave_red_lean_score']), float(j['ave_score_red_lean_percentile'])]
    if j['overage_percentile'] != 'NA':
        overage = [float(j['ave_overage_rate']), float(j['overage_percentile'])]
    
    data_list = []
    data_list.extend(race_fractions)
    data_list.extend(hh_size_fractions)
    data_list.extend(hh_type_fractions)
    data_list.extend(bdeg_fractions)
    data_list.extend(hh_income_fractions)
    data_list.extend(hh_ownership_fractions)    
    data_list.extend(rent_as_pct_fractions)
    data_list.extend(insurance_fractions)
    data_list.extend(gw_sw)
    data_list.extend(timeline_characteristics)
    data_list.extend(regulator)
    data_list.extend(area)
    data_list.extend(population)
    data_list.extend(identity)
    data_list.extend(compliance)
    data_list.extend(overage)
    
    data_array.append(data_list)

dataset = pd.DataFrame(data_array, columns=dataset_columns)

dataset = dataset.replace(' ','_', regex=True)



def data_and_regression_selector(data, independent_sets, dependent_variable, regression, reg_arg = None):
    ind_variables = []
    ind_set_dict = {'race': ['fract_white_alone','fract_black_alone','fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races'],
    'hh_size': ['fract_hh_1worker', 'fract_hh_2worker', 'fract_hh_3+worker', 'fract_hh_3ppl', 'fract_hh_4+ppl'],
    'bdeg': ['fract_seng_compt_mat_stat_deg', 'fract_seng_bio_ag_env_deg', 'fract_seng_phys_sci_deg', 'fract_seng_psych_deg', 'fract_seng_soc_sci_deg', 'fract_seng_eng_deg', 'fract_seng_mds_deg', 'fract_seng_rltd_deg', 'fract_bus_deg', 'fract_edu_deg', 'fract_aho_lit_lang_deg', 'fract_aho_lib_arts_and_hist_deg', 'fract_aho_vis_perf_art_deg', 'fract_aho_comm_deg', 'fract_aho_other_deg'],
    'hh_income': ['fract_hh_income_lt_10k', 'fract_hh_income_10k_15k', 'fract_hh_income_15k_20k', 'fract_hh_income_20k_25k', 'fract_hh_income_25k_30k', 'fract_hh_income_30k_35k', 'fract_hh_income_35k_40k', 'fract_hh_income_40k_45k', 'fract_hh_income_45k_50k', 'fract_hh_income_50k_60k', 'fract_hh_income_60k_75k', 'fract_hh_income_75k_100k', 'fract_hh_income_100k_125k', 'fract_hh_income_125k_150k', 'fract_hh_income_150k_200k', 'fract_hh_income_gt_200k'],
    'hh_own': ['fract_hh_own', 'fract_hh_rent'],
    'rent_as_pct': ['fract_rent_lt_10pct', 'fract_rent_10_14.9pct', 'fract_rent_15_19.9pct', 'fract_rent_20_24.9pct', 'fract_rent_25_29.9pct', 'fract_rent_30_34.9pct', 'fract_rent_35_39.9pct', 'fract_rent_40_49.9pct', 'fract_rent_gt_50pct', 'fract_rent_not_computed'],
    'insurance': ['fract_have_insurance', 'fract_no_insurance'],
    'gw_sw': ['num_gw', 'num_sw'],
    'timeline_characteristics': ['ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches'],
    'regulating': ['regulating'],
    'area': ['arealand', 'areawater'],
    'population': ['population']}

    # print(independent_sets)
    if independent_sets == 'all':
        independent_sets = list(ind_set_dict.keys())
    
    for ind in independent_sets:
        ind_variables.extend(ind_set_dict[ind])
    filtered_data = data[ind_variables]
    X = filtered_data.iloc[:, :].values
    
    dep_var_dict = {'compliance_score': -4, 'compliance_percentile': -3, 'overage_rate': -2, 'overage_percentile': -1}
    y = data.iloc[:, dep_var_dict[dependent_variable]].values
    
    ind_var_columns_list = filtered_data.columns.to_list()

    # Encoding categorical data
    if 'ave_target_timeline' in ind_var_columns_list:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [ind_var_columns_list.index('ave_target_timeline')])], remainder='passthrough')
        X = np.array(ct.fit_transform(X)) 
    if 'regulating' in ind_var_columns_list:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [ind_var_columns_list.index('regulating')])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        
    # Taking care of missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Create imputer object that will replace any nan values with average of the column
    imputer.fit(X[:]) # Call fit method of imputer to create averages for all columns. Arguments specify all rows and the Age + Salary columns (ignore country column to avoid error since country is all strings)
    X[:] = imputer.transform(X[:]) # Transform method then does the replacement of all the nan with mean

    
    if regression == 'linear':
        # print('linear')
        r2_score = linear_regression(X,y)
    elif regression == 'polynomial':
        # print('polynomial')
        r2_score = polynomial_regression(X,y)
    elif regression == 'svr':
        # print('svr')
        r2_score = support_vector_regression(X,y)
    elif regression == 'decision_tree':
        # print('decision_tree')
        r2_score = decision_tree_regression(X,y)
    elif regression == 'rand_forest':
        # print('rand_forest')
        r2_score = random_forest_regression(X,y)
    print(f'finished {reg} with r2: {r2_score}')
    return r2_score
        

# data_and_regression_selector(dataset,['race', 'hh_size', 'bdeg', 'hh_income', 'hh_own', 'rent_as_pct', 'insurance', 'gw_sw', 'timeline_characteristics', 'regulating', 'area', 'population'],'compliance_percentile','linear')
# data_and_regression_selector(dataset,'all','compliance_percentile','linear')
start = time.perf_counter()
output = []
for reg in ['linear', 'polynomial', 'svr', 'decision_tree', 'rand_forest']:
    r2 = data_and_regression_selector(dataset,['bdeg', 'hh_income', 'hh_own', 'rent_as_pct', 'insurance', 'gw_sw', 'timeline_characteristics', 'area', 'population'],'compliance_percentile',reg)
    # r2 = data_and_regression_selector(dataset,['regulating'],'compliance_percentile',reg)
    output.append(f'{reg} got r2 score of: {r2}')
finish = time.perf_counter()
    
print(output)

# powerset (leaving off 'regulating' for now):
from itertools import combinations

variable_sets = ['race', 'hh_size', 'bdeg', 'hh_income', 'hh_own', 'rent_as_pct', 'insurance', 'gw_sw', 'timeline_characteristics', 'area', 'population']
var_combinations = list()

for n in range(len(variable_sets) + 1):
    var_combinations += list(combinations(variable_sets, n))
    
print(var_combinations)

print(len(var_combinations))

print(f'Timer took: {finish-start}')

raise ValueError






# all would be dataset.iloc[:, :-7].values
# race would be dataset.iloc[:, :7].values
# hh_size would be dataset.iloc[:, 7:12].values
# bdeg would be dataset.iloc[:, 12:27].values
# hh_income would be dataset.iloc[:, 27:43].values
# rent_as_pct would be dataset.iloc[:, 43:53].values
# insurance would be dataset.iloc[:, 53:55].values
# gw_sw would be dataset.iloc[:, 55:57].values
X = dataset.iloc[:, :-7].values

# -1 is overage percentile
# -2 is ave_overage_rate
# -3 is compliance percentile
# -4 is compliance score
y = dataset.iloc[:, -3].values

# Multiple Linear Regression R2 score is: 0.0756890029201619 for compliance percentile
# Multiple Linear Regression R2 score is: 0.10779996911329026 for compliance score
# Polynomial Regression R2 score is: 0.07568900292014868 for compliance percentile with degree = 1
# Polynomial Regression R2 score is: 0.10779996911328948 for compliance score with degree = 1
# Support Vector Regression R2 score is: 0.12551090319984126 for compliance percentile
# Support Vector Regression R2 score is: 0.11023953102159001 for compliance score
# Decision Tree Regression R2 score is: -0.5117175837846952 for compliance percentile
# Decision Tree Regression R2 score is: -0.36346148453332594 for compliance score
# Random Forest Regression R2 score is: 0.1248655102559384 for compliance percentile with n_estimators = 400
# Random Forest Regression R2 score is: 0.028948059577550977 for compliance score with n_estimators = 100



# Multiple Linear Regression R2 score is:  for overage percentile
# Multiple Linear Regression R2 score is:  for overage score
# Polynomial Regression R2 score is:  for overage percentile with degree = 1
# Polynomial Regression R2 score is:  for overage score with degree = 1
# Support Vector Regression R2 score is:  for overage percentile
# Support Vector Regression R2 score is:  for overage score
# Decision Tree Regression R2 score is:  for overage percentile
# Decision Tree Regression R2 score is:  for overage score
# Random Forest Regression R2 score is:  for overage percentile with n_estimators = 400
# Random Forest Regression R2 score is:  for overage score with n_estimators = 100



    

























finish = time.perf_counter()
print(f'Seconds: {finish - start}')
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
# print(s.getvalue())
