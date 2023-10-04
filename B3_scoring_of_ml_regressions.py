

# Importing the libraries
import numpy as np
import pandas as pd
import wdc_lib as wdc
import time
import cProfile
import pstats
import io
from pstats import SortKey
import concurrent.futures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

# 12.5 to do:
# integrate steps into main function that calls the right regression
# output as a list that includes: [independent_variable_sets, regression, params, cv_r2_score, r2, adj_r2, mae, mape, mse, rmse]
# have testing iterated in order from that which had the highest cv_r2_score to the lowest
# look up what different relationships between scoring differences mean in terms of being better in one score but worse in another

# 12.7 to do:
# remove ave_target_timeline and method_priority_level from timeline_characteristics? maybe just remove timeline_characteristics altogether?
# determine water system averages for all the contaminants that are not a VOC/SOC (and have an appreciable number of samples. See: contam_info table)
# # if that contaminant has never been sampled at the water system, then leave as a blank (imputer will fill with mean)
# # Put all as a new class, something like 'non_organic_contaminant_averages'. Will have a lot of features obviously.
# #Bring that in as another independent_variable set in the regression calculations
# Also figure out what is going on with the 'regulating' ind_var set.


def linear_regression(X, y, params):
    # # # Multiple Linear Regression:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Training the model
    param_dict = eval(params)
    regressor = LinearRegression(fit_intercept=param_dict['fit_intercept'])
    regressor.fit(X_train, y_train)

    # Predicting test set results
    y_pred = regressor.predict(X_test)

    # Scoring the model
    r2 = r2_score(y_test, y_pred)  # r-squared
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])  # adjusted r-squared
    mae = mean_absolute_error(y_test, y_pred)  # mean absolute error
    mape = mae * 100  # mean absolute percentage error
    mse = mean_squared_error(y_test, y_pred)  # mean squared error
    rmse = sqrt(mse)  # root mean squared error

    return [r2, adj_r2, mae, mape, mse, rmse]


def polynomial_regression(X, y, params):
    # # # Polynomial Regression:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Training the model
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    param_dict = eval(params[:-11])
    regressor = LinearRegression(fit_intercept=param_dict['fit_intercept'])
    regressor.fit(X_train_poly, y_train)

    # Predicting test set results
    y_pred = regressor.predict(poly_features.fit_transform(X_test))

    # Scoring the model
    r2 = r2_score(y_test, y_pred)  # r-squared
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])  # adjusted r-squared
    mae = mean_absolute_error(y_test, y_pred)  # mean absolute error
    mape = mae * 100  # mean absolute percentage error
    mse = mean_squared_error(y_test, y_pred)  # mean squared error
    rmse = sqrt(mse)  # root mean squared error

    return [r2, adj_r2, mae, mape, mse, rmse]


def support_vector_regression(X, y, params):
    # # # SVR:
    y = y.reshape(len(y), 1)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)

    # Training the model
    param_dict = eval(params)
    if 'degree' in param_dict.keys():
        regressor = SVR(kernel=param_dict['kernel'],
                        C=param_dict['C'],
                        gamma=param_dict['gamma'],
                        degree=param_dict['degree'])
    elif 'gamma' in param_dict.keys():
        regressor = SVR(kernel=param_dict['kernel'],
                        C=param_dict['C'],
                        gamma=param_dict['gamma'])
    else:
        regressor = SVR(kernel=param_dict['kernel'],
                        C=param_dict['C'])
    regressor.fit(X_train, y_train.ravel())

    # Predicting test set results
    y_pred = regressor.predict(X_test)

    # Scoring the model
    r2 = r2_score(y_test, y_pred)  # r-squared
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])  # adjusted r-squared
    mae = mean_absolute_error(y_test, y_pred)  # mean absolute error
    mape = mae * 100  # mean absolute percentage error
    mse = mean_squared_error(y_test, y_pred)  # mean squared error
    rmse = sqrt(mse)  # root mean squared error

    return [r2, adj_r2, mae, mape, mse, rmse]


def decision_tree_regression(X, y, params):
    # # # Decision Tree:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Training the model
    param_dict = eval(params)
    regressor = DecisionTreeRegressor(criterion=param_dict['criterion'],
                                      max_depth=param_dict['max_depth'])
    regressor.fit(X_train, y_train)

    # Predicting test set results
    y_pred = regressor.predict(X_test)

    # Scoring the model
    r2 = r2_score(y_test, y_pred)  # r-squared
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])  # adjusted r-squared
    mae = mean_absolute_error(y_test, y_pred)  # mean absolute error
    mape = mae * 100  # mean absolute percentage error
    mse = mean_squared_error(y_test, y_pred)  # mean squared error
    rmse = sqrt(mse)  # root mean squared error

    return [r2, adj_r2, mae, mape, mse, rmse]


def random_forest_regression(X, y, params):
    # # # Random Forest:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Training the model
    param_dict = eval(params)
    regressor = RandomForestRegressor(n_estimators=param_dict['n_estimators'],
                                      criterion=param_dict['criterion'],
                                      max_depth=param_dict['max_depth'])
    regressor.fit(X_train, y_train)

    # Predicting test set results
    y_pred = regressor.predict(X_test)

    # Scoring the model
    r2 = r2_score(y_test, y_pred)  # r-squared
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])  # adjusted r-squared
    mae = mean_absolute_error(y_test, y_pred)  # mean absolute error
    mape = mae * 100  # mean absolute percentage error
    mse = mean_squared_error(y_test, y_pred)  # mean squared error
    rmse = sqrt(mse)  # root mean squared error

    return [r2, adj_r2, mae, mape, mse, rmse]


def catboost_regression(X, y, params):
    # # # Cat Boost Regression:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Training the model
    param_dict = eval(params)
    regressor = CatBoostRegressor(silent=True, max_depth=param_dict['max_depth'],
                                  n_estimators=param_dict['n_estimators'])
    regressor.fit(X_train, y_train)

    # Predicting test set results
    y_pred = regressor.predict(X_test)

    # Scoring the model
    r2 = r2_score(y_test, y_pred)  # r-squared
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])  # adjusted r-squared
    mae = mean_absolute_error(y_test, y_pred)  # mean absolute error
    mape = mae * 100  # mean absolute percentage error
    mse = mean_squared_error(y_test, y_pred)  # mean squared error
    rmse = sqrt(mse)  # root mean squared error

    return [r2, adj_r2, mae, mape, mse, rmse]


def xgboost_regression(X, y, params):
    # # # XGBoost Regression

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Training the model
    param_dict = eval(params)
    regressor = XGBRegressor(max_depth=param_dict['max_depth'])
    regressor.fit(X_train, y_train)

    # Predicting test set results
    y_pred = regressor.predict(X_test)

    # Scoring the model
    r2 = r2_score(y_test, y_pred)  # r-squared
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])  # adjusted r-squared
    mae = mean_absolute_error(y_test, y_pred)  # mean absolute error
    mape = mae * 100  # mean absolute percentage error
    mse = mean_squared_error(y_test, y_pred)  # mean squared error
    rmse = sqrt(mse)  # root mean squared error

    return [r2, adj_r2, mae, mape, mse, rmse]


def ws_contam_mean_handler(independent_sets, contam_info_dict):
    ws_contam_means = ['ws_contam_means_sampled_and_reviewed_and_has_mcl',
                       'ws_contam_means_sampled_reviewed_has_mcl_and_ninety_percent', 'ws_contam_means_tol']
    ws_contam_mean_cols = []
    ws_ind_set_name = 'absent'
    for ind_set in independent_sets:
        if ind_set in ws_contam_means:
            ws_ind_set_name = ind_set
            ws_contam_mean_cols = contam_info_dict[ind_set[16:]]
            break
    return [ws_contam_mean_cols, ws_ind_set_name]


# print('contam_mean_handler_test')
# active_sources = wdc.facilities_to_review()
# contam_dict = wdc.contam_info_organizer(len_of_source_facs=len(active_sources))
# start_contam_mean = time.perf_counter()
# ws_contam_mean_columns = ws_contam_mean_handler(['regulating', 'race', 'hh_size', 'bdeg', 'hh_income', 'hh_own',
#                                                  'rent_as_pct', 'insurance', 'gw_sw', 'area', 'population', 'ws_contam_means_sampled_and_reviewed_and_has_mcl'], contam_dict)
# # ws_contam_mean_columns = ws_contam_mean_handler(['regulating', 'race', 'hh_size', 'bdeg', 'hh_income', 'hh_own',
# #                                                  'rent_as_pct', 'insurance', 'gw_sw', 'area', 'population'], contam_dict)
# fin_contam_mean = time.perf_counter()
# print(ws_contam_mean_columns)
# print(fin_contam_mean - start_contam_mean)
# raise ValueError


def ml_reg_scorer(data, ml_reg_output, contam_info_dict):
    # start_func = time.perf_counter()
    independent_sets, dependent_variable, hyperparameters, regression, mean_test_score = ml_reg_output

    if len(independent_sets) == 0:
        return [[None]*6, [None]*6]

    ws_mean_check = ws_contam_mean_handler(
        independent_sets, contam_info_dict)
    ws_mean_headers = ws_mean_check[0]
    if len(ws_mean_headers) > 0:
        ws_mean_headers = [str(i) for i in ws_mean_headers]
    ws_contam_means = ws_mean_check[1]

    ind_variables = []
    ind_set_dict = {'race': ['fract_white_alone', 'fract_black_alone', 'fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races'],
                    'hh_size': ['fract_hh_1worker', 'fract_hh_2worker', 'fract_hh_3+worker', 'fract_hh_3ppl', 'fract_hh_4+ppl'],
                    'bdeg': ['fract_seng_compt_mat_stat_deg', 'fract_seng_bio_ag_env_deg', 'fract_seng_phys_sci_deg', 'fract_seng_psych_deg', 'fract_seng_soc_sci_deg', 'fract_seng_eng_deg', 'fract_seng_mds_deg', 'fract_seng_rltd_deg', 'fract_bus_deg', 'fract_edu_deg', 'fract_aho_lit_lang_deg', 'fract_aho_lib_arts_and_hist_deg', 'fract_aho_vis_perf_art_deg', 'fract_aho_comm_deg', 'fract_aho_other_deg'],
                    'hh_income': ['fract_hh_income_lt_10k', 'fract_hh_income_10k_15k', 'fract_hh_income_15k_20k', 'fract_hh_income_20k_25k', 'fract_hh_income_25k_30k', 'fract_hh_income_30k_35k', 'fract_hh_income_35k_40k', 'fract_hh_income_40k_45k', 'fract_hh_income_45k_50k', 'fract_hh_income_50k_60k', 'fract_hh_income_60k_75k', 'fract_hh_income_75k_100k', 'fract_hh_income_100k_125k', 'fract_hh_income_125k_150k', 'fract_hh_income_150k_200k', 'fract_hh_income_gt_200k'],
                    'hh_own': ['fract_hh_own', 'fract_hh_rent'],
                    'rent_as_pct': ['fract_rent_lt_10pct', 'fract_rent_10_14.9pct', 'fract_rent_15_19.9pct', 'fract_rent_20_24.9pct', 'fract_rent_25_29.9pct', 'fract_rent_30_34.9pct', 'fract_rent_35_39.9pct', 'fract_rent_40_49.9pct', 'fract_rent_gt_50pct', 'fract_rent_not_computed'],
                    'insurance': ['fract_have_insurance', 'fract_no_insurance'],
                    'ws_characteristics': ['number_gw', 'number_sw', 'ave_act_xldate', 'ave_min_xldate', 'ave_max_xldate', 'ave_range_xldate', 'ave_num_unique_contams', 'max_treatment_plant_class'],
                    'regulating': ['regulating'],
                    'area': ['arealand', 'areawater'],
                    'population': ['population'],
                    ws_contam_means: ws_mean_headers
                    }

    for ind in independent_sets:
        ind_variables.extend(ind_set_dict[ind])
    independent_variables = ', '.join(independent_sets)

    filtered_data = data[ind_variables]

    X = filtered_data.iloc[:, :].values

    dep_var_dict = {'compliance_score': -4, 'compliance_percentile': -
                    3, 'overage_rate': -2, 'overage_percentile': -1}
    y = data.iloc[:, dep_var_dict[dependent_variable]].values

    ind_var_columns_list = filtered_data.columns.to_list()

    # Encoding categorical data

    column_transformer_indices = []
    for ind_var in ind_var_columns_list:
        if ind_var == 'max_treatment_plant_class' or ind_var == 'regulating':
            column_transformer_indices.append(
                ind_var_columns_list.index(ind_var))

    if len(column_transformer_indices) > 0:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(
            sparse=False), column_transformer_indices)], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

    # Taking care of missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:])
    # Transform method then does the replacement of all the nan with mean
    X[:] = imputer.transform(X[:])

    independent_variables = ', '.join(independent_sets)
    try:
        if regression == 'linear':
            scores = linear_regression(X, y, hyperparameters)
        elif regression == 'poly':
            scores = polynomial_regression(X, y, hyperparameters)
        elif regression == 'svr':
            scores = support_vector_regression(X, y, hyperparameters)
        elif regression == 'decision_tree':
            scores = decision_tree_regression(X, y, hyperparameters)
        elif regression == 'random_forest':
            scores = random_forest_regression(X, y, hyperparameters)
        elif regression == 'catboost':
            scores = catboost_regression(X, y, hyperparameters)
        elif regression == 'xgboost':
            scores = xgboost_regression(X, y, hyperparameters)
        else:
            print(ml_reg_output)
            print('regression does not exist')
            raise ValueError

        scoring_output = [independent_variables, dependent_variable,
                          hyperparameters, regression, mean_test_score]
        scoring_output.extend(scores)
        # scoring_output will have:
        # [independent_variables, dependent_variable, hyperparameters, regression, mean_test_score, r2, adj_r2, mae, mape, mse, rmse]
        return scoring_output
    except:
        print(ml_reg_output)
        raise ValueError


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
    df_ws_contam_mean_sampled_reviewed_has_mcl = pd.read_sql_query(
        "SELECT * from ws_contam_mean_sampled_reviewed_has_mcl", conn)
    df_ml_reg_gs = pd.read_sql_query(
        "SELECT * from ml_regressions_gridsearch", conn).sort_values(by='mean_fit_time', ascending=False)  # longest times first
    conn.close()
    print(f'Loaded dfs in {time.perf_counter() - start} seconds')

    df_ws_contam_mean = df_ws_contam_mean_sampled_reviewed_has_mcl

    active_sources = wdc.facilities_to_review()['id'].values.tolist()
    contam_dict = wdc.contam_info_organizer(
        len_of_source_facs=len(active_sources))

    df_ws_compliance_and_overage = pd.merge(
        df_ws_compliance, df_ws_overage, left_on='ws_id', right_on='ws_id', how='left')
    df_ws_compliance_overage_and_contam_mean = pd.merge(
        df_ws_compliance_and_overage, df_ws_contam_mean, left_on='ws_id', right_on='ws_id', how='left')
    df_wsp_and_scores = pd.merge(
        df_ws_compliance_overage_and_contam_mean, df_wsp, left_on='ws_id', right_on='id', how='left')
    df_wsp_score_census = pd.merge(
        df_census, df_wsp_and_scores, left_on='sabl_pwsid', right_on='water_system_number', how='left')

    df_wsp_score_census = df_wsp_score_census[(df_wsp_score_census['ave_red_lean_score'] != 'PMD') & (
        df_wsp_score_census['ave_red_lean_score'] != 'TBD') & (df_wsp_score_census['ave_red_lean_score'] != 'NA')]
    df_wsp_score_census = df_wsp_score_census[(df_wsp_score_census['ave_overage_rate'] != 'PMD') & (
        df_wsp_score_census['ave_overage_rate'] != 'TBD') & (df_wsp_score_census['ave_overage_rate'] != 'NA')]

    # Removing contam columns that are empty after filtering for active, raw sources in community water systems
    ws_columns_list = df_wsp_score_census.columns.to_list()
    for column in ws_columns_list:
        unique_column_values = df_wsp_score_census[column].unique()
        unique_val_str = str(unique_column_values[0])
        if len(unique_column_values) == 1 and unique_val_str == 'nan':
            df_wsp_score_census = df_wsp_score_census.drop([column], axis=1)
            df_ws_contam_mean = df_ws_contam_mean.drop([column], axis=1)

    ws_mean_headers = df_ws_contam_mean.columns.to_list()[1:]

    df_wsp_score_census.drop(['n_100pct_pov_lvl', 'n_101_149pct_pov_lvl', 'n_150pct_pov_lvl', 'id',
                             'pserved', 'type', 'primary_source_water_type', 'ur', 'water_sy_1', 'pop100'], axis=1, inplace=True)
    print(f'Merged and filtered dfs in {time.perf_counter() - start} seconds')

    df_wsp_score_census_columns = ['n_race', 'n_white_alone', 'n_black_alone', 'n_ai_and_an_alone', 'n_asian_alone', 'n_nh_and_opi_alone', 'n_other_alone', 'n_two_or_more_races',
                                   'hh_size', 'hh_1worker', 'hh_2worker', 'hh_3+worker', 'n_hh_3ppl', 'n_hh_4+ppl',
                                   'n_hh_type', 'n_hh_type_fam', 'n_hh_type_fam_mcf', 'n_hh_type_fam_mcf_1unit', 'n_hh_type_fam_mcf_2unit', 'n_hh_type_fam_mcf_mh_and_other', 'n_hh_type_fam_other', 'n_hh_type_fam_other_mhh_nsp', 'n_hh_type_fam_other_mhh_nsp_1unit', 'n_hh_type_fam_other_mhh_nsp_2unit', 'n_hh_type_fam_other_mhh_nsp_mh_and_other', 'n_hh_type_fam_other_fhh_nsp', 'n_hh_type_fam_other_fhh_nsp_1unit', 'n_hh_type_fam_other_fhh_nsp_2unit', 'n_hh_type_fam_other_fhh_nsp_mh_and_other', 'n_hh_type_nonfam', 'n_hh_type_nonfam_1unit', 'n_hh_type_nonfam_2unit', 'n_hh_type_nonfam_mh_and_other',
                                   'n_bachelors_deg', 'n_seng_compt_mat_stat_deg', 'n_seng_bio_ag_env_deg', 'n_seng_phys_sci_deg', 'n_seng_psych_deg', 'n_seng_soc_sci_deg', 'n_seng_eng_deg', 'n_seng_mds_deg', 'n_seng_rltd_deg', 'n_bus_deg', 'n_edu_deg', 'n_aho_lit_lang_deg', 'n_aho_lib_arts_and_hist_deg', 'n_aho_vis_perf_art_deg', 'n_aho_comm_deg', 'n_aho_other_deg',
                                   'n_hh_income', 'n_hh_income_lt_10k', 'n_hh_income_10k_15k', 'n_hh_income_15k_20k', 'n_hh_income_20k_25k', 'n_hh_income_25k_30k', 'n_hh_income_30k_35k', 'n_hh_income_35k_40k', 'n_hh_income_40k_45k', 'n_hh_income_45k_50k', 'n_hh_income_50k_60k', 'n_hh_income_60k_75k', 'n_hh_income_75k_100k', 'n_hh_income_100k_125k', 'n_hh_income_125k_150k', 'n_hh_income_150k_200k', 'n_hh_income_gt_200k',
                                   'n_hh_housing_units', 'n_hh_own', 'n_hh_rent',
                                   'n_rent_as_pct', 'n_rent_lt_10pct', 'n_rent_10_14.9pct', 'n_rent_15_19.9pct', 'n_rent_20_24.9pct', 'n_rent_25_29.9pct', 'n_rent_30_34.9pct', 'n_rent_35_39.9pct', 'n_rent_40_49.9pct', 'n_rent_gt_50pct', 'n_rent_not_computed',
                                   'n_insurance', 'n_have_insurance', 'n_no_insurance',
                                   'number_gw', 'number_sw', 'ave_act_xldate', 'ave_min_xldate', 'ave_max_xldate', 'ave_range_xldate', 'ave_num_unique_contams', 'max_treatment_plant_class',
                                   'regulating',
                                   'arealand', 'areawater',
                                               'population',
                                               'basename', 'centlat', 'centlon', 'funcstat', 'geoid', 'geo_id', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'mtfcc', 'name', 'objectid', 'oid', 'sabl_pwsid', 'state_clas', 'county', 'proportion', 'state', 'tract', 'water_system_number',
                                               'water_system_name', 'ws_id', 'water_system_number', 'water_system_name']  # Moved dependent variables from this list to a couple lines down so that they're the last four items
    df_wsp_score_census_columns.extend(ws_mean_headers)
    df_wsp_score_census_columns.extend(
        ['ave_red_lean_score', 'ave_score_red_lean_percentile', 'ave_overage_rate', 'overage_percentile'])
    df_wsp_score_census = df_wsp_score_census[df_wsp_score_census_columns]

    # Converting to fractions as some census questions may have varying answer rates
    dataset_columns = ['fract_white_alone', 'fract_black_alone', 'fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races',
                       'fract_hh_1worker', 'fract_hh_2worker', 'fract_hh_3+worker', 'fract_hh_3ppl', 'fract_hh_4+ppl',
                       'fract_hh_type_fam', 'fract_hh_type_fam_mcf', 'fract_hh_type_fam_mcf_1unit', 'fract_hh_type_fam_mcf_2unit', 'fract_hh_type_fam_mcf_mh_and_other', 'fract_hh_type_fam_other', 'fract_hh_type_fam_other_mhh_nsp', 'fract_hh_type_fam_other_mhh_nsp_1unit', 'fract_hh_type_fam_other_mhh_nsp_2unit', 'fract_hh_type_fam_other_mhh_nsp_mh_and_other', 'fract_hh_type_fam_other_fhh_nsp', 'fract_hh_type_fam_other_fhh_nsp_1unit', 'fract_hh_type_fam_other_fhh_nsp_2unit', 'fract_hh_type_fam_other_fhh_nsp_mh_and_other', 'fract_hh_type_nonfam', 'fract_hh_type_nonfam_1unit', 'fract_hh_type_nonfam_2unit', 'fract_hh_type_nonfam_mh_and_other',
                       'fract_seng_compt_mat_stat_deg', 'fract_seng_bio_ag_env_deg', 'fract_seng_phys_sci_deg', 'fract_seng_psych_deg', 'fract_seng_soc_sci_deg', 'fract_seng_eng_deg', 'fract_seng_mds_deg', 'fract_seng_rltd_deg', 'fract_bus_deg', 'fract_edu_deg', 'fract_aho_lit_lang_deg', 'fract_aho_lib_arts_and_hist_deg', 'fract_aho_vis_perf_art_deg', 'fract_aho_comm_deg', 'fract_aho_other_deg',
                       'fract_hh_income_lt_10k', 'fract_hh_income_10k_15k', 'fract_hh_income_15k_20k', 'fract_hh_income_20k_25k', 'fract_hh_income_25k_30k', 'fract_hh_income_30k_35k', 'fract_hh_income_35k_40k', 'fract_hh_income_40k_45k', 'fract_hh_income_45k_50k', 'fract_hh_income_50k_60k', 'fract_hh_income_60k_75k', 'fract_hh_income_75k_100k', 'fract_hh_income_100k_125k', 'fract_hh_income_125k_150k', 'fract_hh_income_150k_200k', 'fract_hh_income_gt_200k',
                       'fract_hh_own', 'fract_hh_rent',
                       'fract_rent_lt_10pct', 'fract_rent_10_14.9pct', 'fract_rent_15_19.9pct', 'fract_rent_20_24.9pct', 'fract_rent_25_29.9pct', 'fract_rent_30_34.9pct', 'fract_rent_35_39.9pct', 'fract_rent_40_49.9pct', 'fract_rent_gt_50pct', 'fract_rent_not_computed',
                       'fract_have_insurance', 'fract_no_insurance',
                       'number_gw', 'number_sw', 'ave_act_xldate', 'ave_min_xldate', 'ave_max_xldate', 'ave_range_xldate', 'ave_num_unique_contams', 'max_treatment_plant_class',
                       'regulating',
                       'arealand', 'areawater',
                       'population',
                       'ws_id', 'water_system_number', 'water_system_name']  # Moved dependent variables from this list to a couple lines down so that they're the last four items
    dataset_columns.extend(ws_mean_headers)
    dataset_columns.extend(
        ['ave_red_lean_score', 'ave_score_red_lean_percentile', 'ave_overage_rate', 'overage_percentile'])

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
        ws_characteristics = [None]*8
        # timeline_characteristics = [None]*4
        regulator = [None]*1
        area = [None]*2
        population = [None]*1
        identity = [None]*3
        ws_contam_means = [None] * len(ws_mean_headers)
        compliance = [None]*2
        overage = [None]*2

        if j['n_race'] > 0:
            race_fractions = [j['n_white_alone']/j['n_race'], j['n_black_alone']/j['n_race'], j['n_ai_and_an_alone']/j['n_race'], j['n_asian_alone'] /
                              j['n_race'], j['n_nh_and_opi_alone']/j['n_race'], j['n_other_alone']/j['n_race'], j['n_two_or_more_races']/j['n_race']]
        if j['hh_size'] > 0:
            hh_size_fractions = [j['hh_1worker']/j['hh_size'], j['hh_2worker']/j['hh_size'],
                                 j['hh_3+worker']/j['hh_size'], j['n_hh_3ppl']/j['hh_size'], j['n_hh_4+ppl']/j['hh_size']]
        if j['n_hh_type'] > 0:
            hh_type_fractions = [j['n_hh_type_fam']/j['n_hh_type'], j['n_hh_type_fam_mcf']/j['n_hh_type'], j['n_hh_type_fam_mcf_1unit']/j['n_hh_type'], j['n_hh_type_fam_mcf_2unit']/j['n_hh_type'], j['n_hh_type_fam_mcf_mh_and_other']/j['n_hh_type'], j['n_hh_type_fam_other']/j['n_hh_type'], j['n_hh_type_fam_other_mhh_nsp']/j['n_hh_type'], j['n_hh_type_fam_other_mhh_nsp_1unit']/j['n_hh_type'], j['n_hh_type_fam_other_mhh_nsp_2unit']/j['n_hh_type'],
                                 j['n_hh_type_fam_other_mhh_nsp_mh_and_other']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp_1unit']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp_2unit']/j['n_hh_type'], j['n_hh_type_fam_other_fhh_nsp_mh_and_other']/j['n_hh_type'], j['n_hh_type_nonfam']/j['n_hh_type'], j['n_hh_type_nonfam_1unit']/j['n_hh_type'], j['n_hh_type_nonfam_2unit']/j['n_hh_type'], j['n_hh_type_nonfam_mh_and_other']/j['n_hh_type']]
        if j['n_bachelors_deg'] > 0:
            bdeg_fractions = [j['n_seng_compt_mat_stat_deg']/j['n_bachelors_deg'], j['n_seng_bio_ag_env_deg']/j['n_bachelors_deg'], j['n_seng_phys_sci_deg']/j['n_bachelors_deg'], j['n_seng_psych_deg']/j['n_bachelors_deg'], j['n_seng_soc_sci_deg']/j['n_bachelors_deg'], j['n_seng_eng_deg']/j['n_bachelors_deg'], j['n_seng_mds_deg']/j['n_bachelors_deg'],
                              j['n_seng_rltd_deg']/j['n_bachelors_deg'], j['n_bus_deg']/j['n_bachelors_deg'], j['n_edu_deg']/j['n_bachelors_deg'], j['n_aho_lit_lang_deg']/j['n_bachelors_deg'], j['n_aho_lib_arts_and_hist_deg']/j['n_bachelors_deg'], j['n_aho_vis_perf_art_deg']/j['n_bachelors_deg'], j['n_aho_comm_deg']/j['n_bachelors_deg'], j['n_aho_other_deg']/j['n_bachelors_deg']]
            bdeg_answer_rate = j['n_bachelors_deg']/max_answers
            # Necessary b/c this was the only census question with no negative answer
            bdeg_fractions = [bdeg*bdeg_answer_rate for bdeg in bdeg_fractions]
        if j['n_hh_income'] > 0:
            hh_income_fractions = [j['n_hh_income_lt_10k']/j['n_hh_income'], j['n_hh_income_10k_15k']/j['n_hh_income'], j['n_hh_income_15k_20k']/j['n_hh_income'], j['n_hh_income_20k_25k']/j['n_hh_income'], j['n_hh_income_25k_30k']/j['n_hh_income'], j['n_hh_income_30k_35k']/j['n_hh_income'], j['n_hh_income_35k_40k']/j['n_hh_income'], j['n_hh_income_40k_45k']/j['n_hh_income'],
                                   j['n_hh_income_45k_50k']/j['n_hh_income'], j['n_hh_income_50k_60k']/j['n_hh_income'], j['n_hh_income_60k_75k']/j['n_hh_income'], j['n_hh_income_75k_100k']/j['n_hh_income'], j['n_hh_income_100k_125k']/j['n_hh_income'], j['n_hh_income_125k_150k']/j['n_hh_income'], j['n_hh_income_150k_200k']/j['n_hh_income'], j['n_hh_income_gt_200k']/j['n_hh_income']]
        if j['n_hh_housing_units'] > 0:
            hh_ownership_fractions = [
                j['n_hh_own']/j['n_hh_housing_units'], j['n_hh_rent']/j['n_hh_housing_units']]
        if j['n_rent_as_pct'] > 0:
            rent_as_pct_fractions = [j['n_rent_lt_10pct']/j['n_rent_as_pct'], j['n_rent_10_14.9pct']/j['n_rent_as_pct'], j['n_rent_15_19.9pct']/j['n_rent_as_pct'], j['n_rent_20_24.9pct']/j['n_rent_as_pct'], j['n_rent_25_29.9pct'] /
                                     j['n_rent_as_pct'], j['n_rent_30_34.9pct']/j['n_rent_as_pct'], j['n_rent_35_39.9pct']/j['n_rent_as_pct'], j['n_rent_40_49.9pct']/j['n_rent_as_pct'], j['n_rent_gt_50pct']/j['n_rent_as_pct'], j['n_rent_not_computed']/j['n_rent_as_pct']]
            rent_rate = j['n_rent_as_pct'] * \
                (j['n_hh_rent']/j['n_hh_housing_units'])
            # Necessary b/c own vs rent ratio can vary
            rent_as_pct_fractions = [
                rpct*rent_rate for rpct in rent_as_pct_fractions]
        if j['n_insurance'] > 0:
            insurance_fractions = [
                j['n_have_insurance']/j['n_insurance'], j['n_no_insurance']/j['n_insurance']]
        ws_characteristics = [j['number_gw'], j['number_sw'], j['ave_act_xldate'],
                              j['ave_min_xldate'], j['ave_max_xldate'], j['ave_range_xldate'],
                              j['ave_num_unique_contams'], j['max_treatment_plant_class']]
        regulator = [j['regulating']]
        area = [j['arealand'], j['areawater']]
        population = [j['population']]
        identity = [j['ws_id'], j['water_system_number'],
                    j['water_system_name']]
        # this adds 600+ columns as of most recent data pull
        ws_mean = [j[header] for header in ws_mean_headers]
        compliance = [float(j['ave_red_lean_score']), float(
            j['ave_score_red_lean_percentile'])]
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
        data_list.extend(ws_characteristics)
        data_list.extend(regulator)
        data_list.extend(area)
        data_list.extend(population)
        data_list.extend(identity)
        data_list.extend(ws_mean)
        data_list.extend(compliance)
        data_list.extend(overage)

        data_array.append(data_list)
    dataset = pd.DataFrame(data_array, columns=dataset_columns)

    dataset = dataset.replace(' ', '_', regex=True)
    print(f'dataset created in {time.perf_counter() - start} seconds')

    sublist_creation_start = time.perf_counter()
    ml_reg_gs_list_catboost = []
    ml_reg_gs_list_no_catboost = []
    for i, row in df_ml_reg_gs.iterrows():
        if row['regression'] == 'catboost':
            ml_reg_gs_list_catboost.append([list(row['independent_variables'].split(
                ', ')), row['dependent_variable'], row['params'], row['regression'], row['mean_test_score']])
        else:
            ml_reg_gs_list_no_catboost.append([list(row['independent_variables'].split(
                ', ')), row['dependent_variable'], row['params'], row['regression'], row['mean_test_score']])
    print(
        f'Created cat vs no_cat lists in {time.perf_counter() - start} seconds')
    print(
        f'Len of cat lists is {len(ml_reg_gs_list_catboost)}, len of no_cat list is {len(ml_reg_gs_list_no_catboost)}')

    active_sources = wdc.facilities_to_review()
    contam_dict = wdc.contam_info_organizer(
        len_of_source_facs=len(active_sources))

    # # print([['race', 'hh_income', 'hh_own', 'rent_as_pct', 'insurance', 'timeline_characteristics', 'area'],
    # #       'compliance_percentile', "{'fit_intercept': True} + degree=2", 'poly', -267.86256831824164])
    # print(ml_reg_gs_list_no_catboost[10328])
    # test_start = time.perf_counter()
    # test = ml_reg_scorer(
    #     dataset, ml_reg_gs_list_no_catboost[10328], contam_info_dict=contam_dict)
    # test_fin = time.perf_counter()
    # print(test)
    # print(test_fin - test_start)
    # raise ValueError

    no_cat_sublists = []
    cat_sublists = []
    sublist_size = 100

    for nc in range(0, len(ml_reg_gs_list_no_catboost), sublist_size):
        new_nc_sublist = ml_reg_gs_list_no_catboost[nc:nc+sublist_size]
        no_cat_sublists.append(new_nc_sublist)

    for cat in range(0, len(ml_reg_gs_list_catboost), sublist_size):
        new_cat_sublist = ml_reg_gs_list_catboost[cat:cat+sublist_size]
        cat_sublists.append(new_cat_sublist)

    scoring_columns = ['independent_variables', 'dependent_variable', 'params',
                       'regression', 'mean_test_score', 'r2', 'adj_r2', 'mae', 'mape', 'mse', 'rmse']
    sublists_creation_finish = time.perf_counter()
    print(
        f'Took {sublists_creation_finish - sublist_creation_start} seconds to create sublists.')
    prev_finish = sublists_creation_finish
    for sublist in no_cat_sublists:
        scoring_start = time.perf_counter()
        scoring_output = []
        with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
            results = executor.map(
                ml_reg_scorer, [dataset]*len(sublist), sublist, [contam_dict]*len(sublist))
            end_results_creation = time.perf_counter()
            print(
                f'Results creation: {end_results_creation-scoring_start}')
            for result in results:
                scoring_output.append(result)
            print(
                f'Results iteration (sublist {no_cat_sublists.index(sublist)}): {time.perf_counter()-end_results_creation}')
            print(f'Time so far: {time.perf_counter() - start} seconds')
        df_score = pd.DataFrame(scoring_output,
                                columns=scoring_columns)
        append_or_replace = ('replace')*(prev_finish == sublists_creation_finish) + \
            ('append')*(prev_finish != sublists_creation_finish)

        conn = wdc.sql_query_conn()
        df_score.to_sql('ml_regressions_scoring', conn,
                        if_exists=append_or_replace, index=False)
        conn.close()
        prev_finish = scoring_start

    for sublist in cat_sublists:
        scoring_start = time.perf_counter()
        scoring_output = []

        for cat in sublist:
            scoring_output.append(ml_reg_scorer(dataset, cat, contam_dict))
        df_score = pd.DataFrame(scoring_output,
                                columns=scoring_columns)
        append_or_replace = ('replace')*(prev_finish == sublists_creation_finish) + \
            ('append')*(prev_finish != sublists_creation_finish)

        conn = wdc.sql_query_conn()
        df_score.to_sql('ml_regressions_scoring', conn,
                        if_exists=append_or_replace, index=False)
        conn.close()
        print(
            f'sublist {cat_sublists.index(sublist)} took: {time.perf_counter() - scoring_start} seconds')
        prev_finish = scoring_start

    wdc.create_index('ml_regressions_scoring', independent_variables='ASC')
    wdc.create_index('ml_regressions_scoring', dependent_variable='ASC')
    wdc.create_index('ml_regressions_scoring', params='ASC')
    wdc.create_index('ml_regressions_scoring', mean_test_score='ASC')
    wdc.create_index('ml_regressions_scoring', regression='ASC')

    finish = time.perf_counter()
    print(f'Seconds: {finish - start}')
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

# Jae 4/8/23
# Seconds: 77903.4512753

# Jae 7/21/23
# Seconds: 80760.5827387