import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wdc_lib as wdc
import time
import cProfile
import pstats
import io
from pstats import SortKey
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
import concurrent.futures
from numba import jit
import pandas as pd
from functools import partial
import keras_tuner as kt
import random

# build ANN by Friday
# two approaches:
# use ideal variables seen in wdc - regressions - random forest results
# use dimensionality reduction - feed in all variables

# multiple indpdt var - increase in processing time
# can see how long each epoch lasts and decide gpu acceleration
# November 15th - to make initial prototype of ANN - decide on next steps in terms of boosting efficiency and accuracy

# Dec 1st - A1 - A10 in vapyr and the cesus steps to refresh the data and feed into neural network


# from sklearn.metrics import mean_absolute_percentage_error

# # Defining a function to find the best parameters for ANN
# def FunctionFindBestParams(X_train, y_train, X_test, y_test):

#     # Defining the list of hyper parameters to try
#     batch_size_list=[1, 5, 10, 15, 20, 32]
#     epoch_list  =   [100, 1000, 2000, 10000]

#     import pandas as pd
#     SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

#     # initializing the trials
#     TrialNumber=0
#     for batch_size_trial in batch_size_list:
#         for epochs_trial in epoch_list:
#             TrialNumber+=1
#             # create ANN model
#             model = Sequential()
#             # Defining the first layer of the model
#             model.add(Dense(units=6, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

#             # Defining the Second layer of the model
#             model.add(Dense(units=6, kernel_initializer='normal', activation='relu'))

#             # The output neuron is a single fully connected node
#             # Since we will be predicting a single number
#             model.add(Dense(1, kernel_initializer='normal'))

#             # Compiling the model
#             model.compile(loss='mean_squared_error', optimizer='adam')

#             # Fitting the ANN to the Training set
#             model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)

#             MAPE = 100 * mean_absolute_percentage_error(y_test, model.predict(X_test))


#             # printing the results of the current iteration
#             print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)

#             SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
#                                                                     columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
#     return(SearchResultsData)


from sklearn.metrics import mean_absolute_percentage_error

# Defining a function to find the best parameters for ANN



@jit(forceobj=True, parallel=True)
def FunctionFindBestParams(X_train, y_train, X_test, y_test):

    # Defining the list of hyper parameters to try
    epoch_list = [i for i in range(101)]
    units_list = [i for i in range(68, 200)]
    hidden_layer_list = [i for i in range(2, 201)]
    activation_function_name_list = ["relu", "relu6", "elu", "selu", "swish"]
    activation_function_name_list_mix = [(a, b) for a in activation_function_name_list for b in activation_function_name_list]
    activation_function_name_list_mix_clean = [*set(activation_function_name_list_mix)]
    # print(activation_function_name_list_mix_clean)
    combination_list_single_hidden = [(e,u,1,(a,"N/A")) for e in epoch_list for u in units_list for a in activation_function_name_list]
    combination_list_multi_hidden = [(e,u,h,a) for e in epoch_list for u in units_list for h in hidden_layer_list for a in activation_function_name_list_mix_clean]
    print(combination_list_single_hidden)
    print(combination_list_multi_hidden)
    print(len(combination_list_single_hidden))
    print(len(combination_list_multi_hidden))

    raise ValueError

    combination_list = combination_list_single_hidden + combination_list_multi_hidden
    SearchResultsData = pd.DataFrame(
        columns=['trial_number', 'epoch', 'hidden_layers', 'units', 'activation_function', 'accuracy', 'r2', 'adj_r2'])

    # initializing the trials
    print(len(combination_list))
    TrialNumber = 0
    for combo in combination_list: 
        TrialNumber += 1
        epochs_trial = combo[0]
        units_trial = combo[1]
        hidden_trial = combo[2] 
        activation_trial_initial = combo[3][0]
        activation_trial_body = combo[3][1]
        
        # create ANN model
        model = tf.keras.models.Sequential()
        # Defining the first layer of the model
        model.add(tf.keras.layers.Dense(units= units_trial, input_dim = 68, activation=activation_trial_initial))
        if hidden_trial > 1:
            for _ in range(hidden_trial - 1):
                model.add(tf.keras.layers.Dense(
                    units=units_trial, activation= activation_trial_body))

    

        # The output neuron is a single fully connected node
        # Since we will be predicting a single number
        model.add(tf.keras.layers.Dense(1))

        # Compiling the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Fitting the ANN to the Training set
        model.fit(X_train, y_train, batch_size=1,
                epochs=epochs_trial, verbose=0)
        y_pred = model.predict(X_test)
        MAPE = mean_absolute_percentage_error(y_test, model.predict(X_test))
        MAPE_100 = 100 * MAPE
        # MAPE = "N/A"
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])
        
        # printing the results of the current iteration
        print("Trial Number: ", TrialNumber, 
            ' / ', 'epochs:', epochs_trial,' / ', 'hidden layers: ', hidden_trial,' / ', 'units: ', units_trial, ' / ', 'activation function initial: ', activation_trial_initial, ' / ','activation function body: ', activation_trial_body,' / ', 'Accuracy:', 100 - MAPE_100,' / ', "R2/Adj R2: ", str(r2) + ' / ' + str(adj_r2))
        load = {'trial_number': [TrialNumber], 'epoch': [epochs_trial], 'hidden_layers': [hidden_trial], 'units': [units_trial], 'activation_function': [combo[3]],  'accuracy': [100-MAPE_100], "r2" : [r2],  'adj_r2' : [adj_r2]}
        load_df = pd.DataFrame.from_dict(load)
        SearchResultsData = pd.concat([SearchResultsData, load_df])

        # print(SearchResultsData)
    return SearchResultsData 


def random_forest_regression(X, y):
    # # # Random Forest:
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)
    # Training the Random Forest Regression model on the whole dataset
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])
    return [r2, adj_r2]


def dataset_filter(data, independent_sets, dependent_variable):
    # start_func = time.perf_counter()
    if len(independent_sets) == 0:
        return [[None]*6, [None]*6]
    ind_variables = []
    ind_set_dict = {'race': ['fract_white_alone', 'fract_black_alone', 'fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races'],
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

    dep_var_dict = {'compliance_score': -4, 'compliance_percentile': -
                    3, 'overage_rate': -2, 'overage_percentile': -1}
    y = data.iloc[:, dep_var_dict[dependent_variable]].values

    ind_var_columns_list = filtered_data.columns.to_list()

    # Encoding categorical data
    if 'ave_target_timeline' in ind_var_columns_list:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [
                               ind_var_columns_list.index('ave_target_timeline')])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
    if 'regulating' in ind_var_columns_list:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [
                               ind_var_columns_list.index('regulating')])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

    # Taking care of missing data
    # Create imputer object that will replace any nan values with average of the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:])  # Call fit method of imputer to create averages for all columns. Arguments specify all rows and the Age + Salary columns (ignore country column to avoid error since country is all strings)
    # Transform method then does the replacement of all the nan with mean
    X[:] = imputer.transform(X[:])

    return [ind_var_columns_list, X, y]


def random_forest_regression(X, y):
    # # # Random Forest:
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)
    # Training the Random Forest Regression model on the whole dataset
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])
    return [r2, adj_r2]


if __name__ == '__main__':
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

    df_ws_compliance_and_overage = pd.merge(
        df_ws_compliance, df_ws_overage, left_on='ws_id', right_on='ws_id', how='left')
    df_wsp_and_scores = pd.merge(
        df_ws_compliance_and_overage, df_wsp, left_on='ws_id', right_on='id', how='left')
    df_wsp_score_census = pd.merge(
        df_census, df_wsp_and_scores, left_on='sabl_pwsid', right_on='water_system_number', how='left')
    df_wsp_score_census = df_wsp_score_census[(df_wsp_score_census['ave_red_lean_score'] != 'PMD') & (
        df_wsp_score_census['ave_red_lean_score'] != 'TBD') & (df_wsp_score_census['ave_red_lean_score'] != 'NA')]

    df_wsp_score_census.drop(['n_100pct_pov_lvl', 'n_101_149pct_pov_lvl', 'n_150pct_pov_lvl', 'id',
                             'pserved', 'type', 'primary_source_water_type', 'ur', 'water_sy_1', 'pop100'], axis=1, inplace=True)
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
                                               'water_system_name', 'ws_id', 'water_system_number', 'water_system_name', 'ave_red_lean_score', 'ave_score_red_lean_percentile', 'ave_overage_rate', 'overage_percentile']]

    # Converting to fractions as some census questions may have varying answer rates
    dataset_columns = ['fract_white_alone', 'fract_black_alone', 'fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races',
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
                       'ws_id', 'water_system_number', 'water_system_name',
                       'ave_red_lean_score', 'ave_score_red_lean_percentile',
                       'ave_overage_rate', 'overage_percentile']

    data_array = []
    for i, j in df_wsp_score_census.iterrows():
        max_answers = max(j['n_race'], j['hh_size'], j['n_hh_type'], j['n_bachelors_deg'],
                          j['n_hh_income'], j['n_hh_housing_units'], j['n_rent_as_pct'], j['n_insurance'])

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
        #7
        if j['n_race'] > 0:
            race_fractions = [j['n_white_alone']/j['n_race'], j['n_black_alone']/j['n_race'], j['n_ai_and_an_alone']/j['n_race'], j['n_asian_alone'] /
                              j['n_race'], j['n_nh_and_opi_alone']/j['n_race'], j['n_other_alone']/j['n_race'], j['n_two_or_more_races']/j['n_race']]
        #5
        if j['hh_size'] > 0:
            hh_size_fractions = [j['hh_1worker']/j['hh_size'], j['hh_2worker']/j['hh_size'],
                                 j['hh_3+worker']/j['hh_size'], j['n_hh_3ppl']/j['hh_size'], j['n_hh_4+ppl']/j['hh_size']]
        # 9
        if j['n_hh_type'] > 0:
            hh_type_fractions = [j['n_hh_type_fam']/j['n_hh_type'], j['n_hh_type_fam_mcf']/j['n_hh_type'], j['n_hh_type_fam_mcf_1unit']/j['n_hh_type'], j['n_hh_type_fam_mcf_2unit']/j['n_hh_type'], j['n_hh_type_fam_mcf_mh_and_other']/j['n_hh_type'], j['n_hh_type_fam_other']/j['n_hh_type'], j['n_hh_type_fam_other_mhh_nsp']/j['n_hh_type'], j['n_hh_type_fam_other_mhh_nsp_1unit']/j['n_hh_type'], j['n_hh_type_fam_other_mhh_nsp_2unit']/j['n_hh_type'],
                                 j['n_hh_type_fam_other_mhh_nsp_mh_and_other']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp_1unit']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp_2unit']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp_mh_and_other']/j['n_hh_type'], j['n_hh_type_nonfam']/j['n_hh_type'], j['n_hh_type_nonfam_1unit']/j['n_hh_type'], j['n_hh_type_nonfam_2unit']/j['n_hh_type'], j['n_hh_type_nonfam_mh_and_other']/j['n_hh_type']]
        # 15
        if j['n_bachelors_deg'] > 0:
            bdeg_fractions = [j['n_seng_compt_mat_stat_deg']/j['n_bachelors_deg'], j['n_seng_bio_ag_env_deg']/j['n_bachelors_deg'], j['n_seng_phys_sci_deg']/j['n_bachelors_deg'], j['n_seng_psych_deg']/j['n_bachelors_deg'], j['n_seng_soc_sci_deg']/j['n_bachelors_deg'], j['n_seng_eng_deg']/j['n_bachelors_deg'], j['n_seng_mds_deg']/j['n_bachelors_deg'],
                              j['n_seng_rltd_deg']/j['n_bachelors_deg'], j['n_bus_deg']/j['n_bachelors_deg'], j['n_edu_deg']/j['n_bachelors_deg'], j['n_aho_lit_lang_deg']/j['n_bachelors_deg'], j['n_aho_lib_arts_and_hist_deg']/j['n_bachelors_deg'], j['n_aho_vis_perf_art_deg']/j['n_bachelors_deg'], j['n_aho_comm_deg']/j['n_bachelors_deg'], j['n_aho_other_deg']/j['n_bachelors_deg']]
            bdeg_answer_rate = j['n_bachelors_deg']/max_answers
            # Necessary b/c this was the only census question with no negative answer
            bdeg_fractions = [bdeg*bdeg_answer_rate for bdeg in bdeg_fractions]
        #16
        if j['n_hh_income'] > 0:
            hh_income_fractions = [j['n_hh_income_lt_10k']/j['n_hh_income'], j['n_hh_income_10k_15k']/j['n_hh_income'], j['n_hh_income_15k_20k']/j['n_hh_income'], j['n_hh_income_20k_25k']/j['n_hh_income'], j['n_hh_income_25k_30k']/j['n_hh_income'], j['n_hh_income_30k_35k']/j['n_hh_income'], j['n_hh_income_35k_40k']/j['n_hh_income'], j['n_hh_income_40k_45k']/j['n_hh_income'],
                                   j['n_hh_income_45k_50k']/j['n_hh_income'], j['n_hh_income_50k_60k']/j['n_hh_income'], j['n_hh_income_60k_75k']/j['n_hh_income'], j['n_hh_income_75k_100k']/j['n_hh_income'], j['n_hh_income_100k_125k']/j['n_hh_income'], j['n_hh_income_125k_150k']/j['n_hh_income'], j['n_hh_income_150k_200k']/j['n_hh_income'], j['n_hh_income_gt_200k']/j['n_hh_income']]
        #2
        if j['n_hh_housing_units'] > 0:
            hh_ownership_fractions = [
                j['n_hh_own']/j['n_hh_housing_units'], j['n_hh_rent']/j['n_hh_housing_units']]
        #10
        if j['n_rent_as_pct'] > 0:
            rent_as_pct_fractions = [j['n_rent_lt_10pct']/j['n_rent_as_pct'], j['n_rent_10_14.9pct']/j['n_rent_as_pct'], j['n_rent_15_19.9pct']/j['n_rent_as_pct'], j['n_rent_20_24.9pct']/j['n_rent_as_pct'], j['n_rent_25_29.9pct'] /
                                     j['n_rent_as_pct'], j['n_rent_30_34.9pct']/j['n_rent_as_pct'], j['n_rent_35_39.9pct']/j['n_rent_as_pct'], j['n_rent_40_49.9pct']/j['n_rent_as_pct'], j['n_rent_gt_50pct']/j['n_rent_as_pct'], j['n_rent_not_computed']/j['n_rent_as_pct']]
            rent_rate = j['n_rent_as_pct'] * \
                (j['n_hh_rent']/j['n_hh_housing_units'])
            # Necessary b/c own vs rent ratio can vary
            # 1
            rent_as_pct_fractions = [
                rpct*rent_rate for rpct in rent_as_pct_fractions]
        #2
        if j['n_insurance'] > 0:
            insurance_fractions = [
                j['n_have_insurance']/j['n_insurance'], j['n_no_insurance']/j['n_insurance']]
        #2
        gw_sw = [j['number_gw'], j['number_sw']]
        #4
        timeline_characteristics = [j['ave_target_timeline'], j['ave_method_priority_level'],
                                    j['ave_num_time_segments'], j['ave_num_track_switches']]
        #1
        regulator = [j['regulating']]
        #2
        area = [j['arealand'], j['areawater']]
        #1
        population = [j['population']]
        #3
        identity = [j['ws_id'], j['water_system_number'],
                    j['water_system_name']]
        #2
        compliance = [float(j['ave_red_lean_score']), float(
            j['ave_score_red_lean_percentile'])]
        #2
        if j['overage_percentile'] != 'NA':
            overage = [float(j['ave_overage_rate']),
                       float(j['overage_percentile'])]

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

    dataset = dataset.replace(' ', '_', regex=True)

    # best r2 for compliance_score was with random forest on ['hh_size', 'bdeg', 'insurance', 'gw_sw', 'timeline_characteristics']
    # best adj_r2 for compliance_score was with random forest on ['hh_size', 'insurance', 'gw_sw', 'timeline_characteristics', 'area']
    # if you want all variables EXCEPT for regulating ['race', 'hh_size', 'bdeg', 'hh_income', 'hh_own', 'rent_as_pct', 'insurance', 'gw_sw', 'timeline_characteristics', 'area', 'population']
    filtered_data = dataset_filter(dataset, independent_sets=['race', 'hh_size', 'bdeg', 'hh_income', 'hh_own', 'rent_as_pct',
                                   'insurance', 'gw_sw', 'timeline_characteristics', 'area', 'population'], dependent_variable='compliance_score')
    # filtered_data = dataset_filter(dataset, independent_sets=['hh_size', 'bdeg', 'insurance', 'gw_sw', 'timeline_characteristics'], dependent_variable='compliance_score')



    independent_variables = filtered_data[0]
    X = filtered_data[1]
    y = filtered_data[2]

    # print(independent_variables)
    # print(len(independent_variables))
    # raise ValueError

    # Splitting the dataset into the Training set and Test set
    r2_list = []
    # for i in range(100):
        # print(f"Loop {i}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # FunctionFindBestParams(X_train, y_train, X_test, y_test)

    # # This is just to test that the data was organized correctly
    # # ['hh_size', 'bdeg', 'insurance', 'gw_sw', 'timeline_characteristics']
    # # Should have r2 of 0.728174973621484 & adj_r2 of 0.719301468951892
    # print(random_forest_regression(X, y))
    # raise ValueError
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    # Initializing the ANN
    # print(FunctionFindBestParams(X_train, y_train, X_test, y_test))
    # table_best_params = FunctionFindBestParams(X_train, y_train, X_test, y_test)
    # table_best_params.to_csv("best_params_search.csv")
    # ann = tf.keras.models.Sequential()
    # number_of_layers = 200
    # randomize_unit_list = [random.randrange(100, 200, 1) for _ in range(number_of_layers)]
    # randomized_activation_list = [random.randrange(0, 5, 1) for _ in range(number_of_layers)]
    # activation_list = ["selu", "relu", "relu6", "elu", "swish"]
    # ann.add(tf.keras.layers.Dense(units= randomize_unit_list[0], 
    # input_dim = 68, 
    # activation = activation_list[randomized_activation_list[0]]))
    # for i in range(1, number_of_layers):
    #     ann.add(tf.keras.layers.Dense(units= randomize_unit_list[i],  activation = activation_list[randomized_activation_list[i]]))

    ann = tf.keras.models.Sequential()


    ann.add(tf.keras.layers.Dense(units= 501, input_dim=68, activation="swish"))
    ann.add(tf.keras.layers.Dense(units= 509, activation="relu6"))
    ann.add(tf.keras.layers.Dense(units= 130,  activation="selu"))
    ann.add(tf.keras.layers.Dense(units= 311,  activation="relu"))
    ann.add(tf.keras.layers.Dense(units= 266,  activation="swish"))






    # # # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1))

    # # # Part 3 - Training the ANN
    
    # # # Using Hyperband and HP fit
    # model_builder = 
    # tuner = kt.Hyperband(model_builder,
    #                     objective='val_accuracy',
    #                     max_epochs=10,
    #                     factor=3)
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
    # best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    # model = tuner.hypermodel.build(best_hps)
    # history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

    # val_acc_per_epoch = history.history['val_accuracy']
    # best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    # print('Best epoch: %d' % (best_epoch,))

    # hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    # hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
    # eval_result = hypermodel.evaluate(img_test, label_test)
    # print("[test loss, test accuracy]:", eval_result)


    # # # Compiling the ANN
    ann.compile(optimizer='adam', loss="mean_squared_error")

    # # # Training the ANN on the Training set
    ann.fit(X_train, y_train, batch_size=1, epochs=5)

    


    # 128 hidden layer x 3 
    # 3000 epochs
    # loss: 43.3718
    # normal r2/adj_r2
    # 0.5555102403221246
    # 0.5212022906534577

    # 123 Parameters: batch_size: 1 - epochs: 20 hidden layers:  2 activation function:  relu6 Accuracy: 76.47456847560395 R2/Adj R2:  0.6075984947424813 / 0.5773109778781098

    # Predicting the Test set results
    y_pred = ann.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1),
          y_test.reshape(len(y_test), 1)), 1))
    print('keras.metrics.Accuracy function')
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(y_test, y_pred)
    print(metric.result().numpy())
    print('MAPE:')
    from sklearn.metrics import mean_absolute_percentage_error
    MAPE = mean_absolute_percentage_error(y_test, ann.predict(X_test))
    print(MAPE)
    print('normal r2/adj_r2')

    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])
    print(f"r2 / adj r2: {r2} / {adj_r2}")
    r2_list.append((r2, adj_r2))

    print(r2_list)

    print('ann.evaluate function:')
    print(ann.evaluate(np.asarray(X_test).astype('float32'), np.asarray(y_test).astype('float32')))

    finish = time.perf_counter()
    print(f'Seconds: {finish - start}')
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
