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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from itertools import combinations
import math


def linear_regression(X, y):
    # # # Multiple Linear Regression:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Gridsearch Cross-Validation
    regressor = LinearRegression()
    parameters = [{'fit_intercept': [True, False]}]
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
    grid_search.fit(X_train, y_train)
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    df_gridsearch['params'] = df_gridsearch['params'].astype("str")
    df_gridsearch['regression'] = 'linear'
    print("### LINEAR ")
    print(grid_search.best_estimator_.coef_)
    print("### LINEAR ")
    return df_gridsearch


def polynomial_regression(X, y):
    # # # Polynomial Regression:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Gridsearch Cross-Validation
    regressor = LinearRegression()
    degrees = [2]
    output_data = []
    for deg in degrees:
        poly_features = PolynomialFeatures(degree=deg)
        X_train_poly = poly_features.fit_transform(X_train)
        regressor = LinearRegression()
        # parameters = [{'fit_intercept': [True], 'normalize':[
        #     True, False]}, {'fit_intercept': [False]}]
        parameters = [{'fit_intercept': [True, False]}]
        grid_search = GridSearchCV(estimator=regressor,
                                   param_grid=parameters,
                                   scoring='r2',
                                   cv=10,
                                   n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
        grid_search.fit(X_train_poly, y_train)
        test_gridsearch = pd.DataFrame(grid_search.cv_results_)
        test_gridsearch['params'] = test_gridsearch['params'].astype("str")
        test_gridsearch['params'] = test_gridsearch['params'] + \
            ' + degree=' + str(deg)
        test_gridsearch['regression'] = 'poly'
        output_data.append(test_gridsearch)
    df_gridsearch = pd.concat(output_data)

    return df_gridsearch


def support_vector_regression(X, y):
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

    # Gridsearch Cross-Validation
    regressor = SVR()
    parameters = [{'kernel': ['linear'], 'C': [0.1, 1, 5]},
                  {'kernel': ['rbf', 'sigmoid'], 'gamma': [
                      0.0001, 0.01, 0.1, 1, 5, 10], 'C': [0.1, 1, 5]},
                  {'kernel': ['poly'], 'gamma': [0.0001, 0.001, 0.01],  'C': [0.1, 1, 5], 'degree': [2, 3]}]

    # {'kernel': ['linear'], 'C': [0.1, 1, 5, 10]} took 17 seconds
    # # # Keep C at 10 or less since it can take a long time
    # rbf alone was 15 seconds
    # sigmoid alone was 13 seconds
    # rbf, sigmoid was 28 seconds
    # {'kernel': ['rbf', 'sigmoid'],'gamma': [0.0001, 0.01, 0.1, 1, 5, 10], 'C': [0.1, 1, 5, 10]} took 14 seconds
    # # # Keep C at 10 or less since it can take a long time
    # poly with gamma 1 took 294 seconds.
    # {'kernel': ['poly'], 'gamma': [0.5], 'degree': [2, 3]} took 54 seconds.
    # parameters = {'kernel': ['poly'], 'gamma': [0.0001, 0.01, 0.1],  'C': [0.1, 1, 5, 10], 'degree': [2, 3]} took 26 seconds
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
    grid_search.fit(X_train, y_train.ravel())
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    df_gridsearch['params'] = df_gridsearch['params'].astype("str")
    df_gridsearch['regression'] = 'svr'
    return df_gridsearch


def decision_tree_regression(X, y):
    # # # Decision Tree:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Gridsearch Cross-Validation
    regressor = DecisionTreeRegressor(random_state=83)
    parameters = [{'criterion': ['squared_error',
                                 'friedman_mse', 'absolute_error', 'poisson'], 'max_depth': [3, 5, 10, None]}]
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
    grid_search.fit(X_train, y_train)
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    df_gridsearch['params'] = df_gridsearch['params'].astype("str")
    df_gridsearch['regression'] = 'decision_tree'
    return df_gridsearch


def random_forest_regression(X, y):
    # # # Random Forest:
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Gridsearch Cross-Validation
    regressor = RandomForestRegressor()
    # parameters = [{'n_estimators': [10, 100, 200], 'criterion': [
    #     'squared_error', 'poisson']}]
    parameters = [{'n_estimators': [10, 100, 200], 'criterion': [
        'squared_error', 'poisson'], 'max_depth': [3, 5, None]}]
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
    grid_search.fit(X_train, y_train)
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    df_gridsearch['params'] = df_gridsearch['params'].astype("str")
    df_gridsearch['regression'] = 'random_forest'
    return df_gridsearch


def catboost_regression(X, y):
    # # # Cat Boost Regression:

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)
    # to evaluate weights

    # Gridsearch Cross-Validation
    regressor = CatBoostRegressor(silent=True)
    parameters = [{'max_depth': [3, 4, 5],
                   'n_estimators':[100, 200, 300]}]
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
    grid_search.fit(X_train, y_train)
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    df_gridsearch['params'] = df_gridsearch['params'].astype("str")
    df_gridsearch['regression'] = 'catboost'

    # print(grid_search.best_estimator_)

    return df_gridsearch


def xgboost_regression(X, y):
    # # # XGBoost Regression

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83)

    # Gridsearch Cross-Validation
    regressor = XGBRegressor()
    parameters = [{'max_depth': [3, 4, 5, 6]}]
    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
    grid_search.fit(X_train, y_train)
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    df_gridsearch['params'] = df_gridsearch['params'].astype("str")
    df_gridsearch['regression'] = 'xgboost'
    return df_gridsearch


def data_and_regression_selector(data, independent_sets, dependent_variable, ws_mean_headers):
    # start_func = time.perf_counter()
    # print(independent_sets)
    # print(type(independent_sets))
    # if type(independent_sets) == tuple:
    #     independent_sets = list(independent_sets)
    # print(independent_sets)
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
                    # 'timeline_characteristics': ['ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches'],
                    'regulating': ['regulating'],
                    'area': ['arealand', 'areawater'],
                    'population': ['population'],
                    'ws_contam_means': ws_mean_headers

                    }

    # if independent_sets == 'all':
    #     independent_sets = list(ind_set_dict.keys())
    if type(independent_sets) == tuple:
        for ind in independent_sets:
            ind_variables.extend(ind_set_dict[ind])
        independent_variables = ', '.join(independent_sets)

    else:
        ind_variables.extend(ind_set_dict[independent_sets])
        independent_variables = independent_sets

    filtered_data = data[ind_variables]
    X = filtered_data.iloc[:, :].values

    dep_var_dict = {'compliance_score': -4, 'compliance_percentile': -
                    3, 'overage_rate': -2, 'overage_percentile': -1}
    y = data.iloc[:, dep_var_dict[dependent_variable]].values

    ind_var_columns_list = filtered_data.columns.to_list()

    # Encoding categorical data

    # if 'ave_target_timeline' in ind_var_columns_list:
    #     ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [
    #                            ind_var_columns_list.index('ave_target_timeline')])], remainder='passthrough')
    #     X = np.array(ct.fit_transform(X))

    if 'regulating' in ind_var_columns_list:
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [
                               ind_var_columns_list.index('regulating')])], remainder='passthrough')
        # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

    # Taking care of missing data
    print('$$$$$$$$$')
    print(X.shape)
    print(filtered_data.columns.to_list())
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:])
    # Transform method then does the replacement of all the nan with mean
    X[:] = imputer.transform(X[:])

    times = []
    df_gridsearch_results = []
    start_regs = time.perf_counter()
    df_gridsearch_results.append(linear_regression(X, y))

    linear_fin = time.perf_counter()
    times.append(linear_fin - start_regs)
    print(f'lin_reg took: {times[-1]}')

    if df_gridsearch_results[0]['mean_test_score'].max() >= 0.5:
        print(df_gridsearch_results[0])
        raise ValueError
        df_gridsearch_results.append(polynomial_regression(X, y))

        poly_fin = time.perf_counter()
        times.append(poly_fin - linear_fin)
        print(f'poly_reg took: {times[-1]}')

        df_gridsearch_results.append(support_vector_regression(X, y))

        svr_fin = time.perf_counter()
        times.append(svr_fin - poly_fin)
        print(f'svr_reg took: {times[-1]}')

        df_gridsearch_results.append(decision_tree_regression(X, y))

        dt_fin = time.perf_counter()
        times.append(dt_fin - svr_fin)
        print(f'dt_reg took: {times[-1]}')

        df_gridsearch_results.append(random_forest_regression(X, y))

        rf_fin = time.perf_counter()
        times.append(rf_fin - dt_fin)
        print(f'rf_reg took: {times[-1]}')

        df_gridsearch_results.append(catboost_regression(X, y))
        cat_fin = time.perf_counter()
        times.append(cat_fin - rf_fin)
        print(f'cat_reg took: {times[-1]}')

        df_gridsearch_results.append(xgboost_regression(X, y))
        xg_fin = time.perf_counter()
        times.append(xg_fin - cat_fin)
        print(f'xg_reg took: {times[-1]}')

    df_all_gridsearch = pd.concat(df_gridsearch_results)
    df_all_gridsearch['independent_variables'] = independent_variables
    df_all_gridsearch = df_all_gridsearch[['independent_variables', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'regression',
                                           'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                                           'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'split5_test_score', 'split6_test_score', 'split7_test_score', 'split8_test_score', 'split9_test_score']]
    # print(df_all_gridsearch['mean_test_score'].max())
    print(f'{independent_sets}: {str(times)} ({str(sum(times))} seconds)')
    return df_all_gridsearch.values.tolist()


if __name__ == '__main__':
    print("START")
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
    df_ws_contam_mean = pd.read_sql_query(
        "SELECT * from ws_sampled_and_reviewed_contam_mean", conn)
    conn.close()

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
    # df_wsp_score_census = df_wsp_score_census[['n_race', 'n_white_alone', 'n_black_alone', 'n_ai_and_an_alone', 'n_asian_alone', 'n_nh_and_opi_alone', 'n_other_alone', 'n_two_or_more_races',
    #                                            'hh_size', 'hh_1worker', 'hh_2worker', 'hh_3+worker', 'n_hh_3ppl', 'n_hh_4+ppl',
    #                                           'n_hh_type', 'n_hh_type_fam', 'n_hh_type_fam_mcf', 'n_hh_type_fam_mcf_1unit', 'n_hh_type_fam_mcf_2unit', 'n_hh_type_fam_mcf_mh_and_other', 'n_hh_type_fam_other', 'n_hh_type_fam_other_mhh_nsp', 'n_hh_type_fam_other_mhh_nsp_1unit', 'n_hh_type_fam_other_mhh_nsp_2unit', 'n_hh_type_fam_other_mhh_nsp_mh_and_other', 'n_hh_type_fam_other_fhh_nsp', 'n_hh_type_fam_other_fhh_nsp_1unit', 'n_hh_type_fam_other_fhh_nsp_2unit', 'n_hh_type_fam_other_fhh_nsp_mh_and_other', 'n_hh_type_nonfam', 'n_hh_type_nonfam_1unit', 'n_hh_type_nonfam_2unit', 'n_hh_type_nonfam_mh_and_other',
    #                                            'n_bachelors_deg', 'n_seng_compt_mat_stat_deg', 'n_seng_bio_ag_env_deg', 'n_seng_phys_sci_deg', 'n_seng_psych_deg', 'n_seng_soc_sci_deg', 'n_seng_eng_deg', 'n_seng_mds_deg', 'n_seng_rltd_deg', 'n_bus_deg', 'n_edu_deg', 'n_aho_lit_lang_deg', 'n_aho_lib_arts_and_hist_deg', 'n_aho_vis_perf_art_deg', 'n_aho_comm_deg', 'n_aho_other_deg',
    #                                            'n_hh_income', 'n_hh_income_lt_10k', 'n_hh_income_10k_15k', 'n_hh_income_15k_20k', 'n_hh_income_20k_25k', 'n_hh_income_25k_30k', 'n_hh_income_30k_35k', 'n_hh_income_35k_40k', 'n_hh_income_40k_45k', 'n_hh_income_45k_50k', 'n_hh_income_50k_60k', 'n_hh_income_60k_75k', 'n_hh_income_75k_100k', 'n_hh_income_100k_125k', 'n_hh_income_125k_150k', 'n_hh_income_150k_200k', 'n_hh_income_gt_200k',
    #                                            'n_hh_housing_units', 'n_hh_own', 'n_hh_rent',
    #                                            'n_rent_as_pct', 'n_rent_lt_10pct', 'n_rent_10_14.9pct', 'n_rent_15_19.9pct', 'n_rent_20_24.9pct', 'n_rent_25_29.9pct', 'n_rent_30_34.9pct', 'n_rent_35_39.9pct', 'n_rent_40_49.9pct', 'n_rent_gt_50pct', 'n_rent_not_computed',
    #                                            'n_insurance', 'n_have_insurance', 'n_no_insurance',
    #                                            'number_gw', 'number_sw',
    #                                            'ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches',
    #                                            'regulating',
    #                                            'arealand', 'areawater',
    #                                            'population',
    #                                            'basename', 'centlat', 'centlon', 'funcstat', 'geoid', 'geo_id', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'mtfcc', 'name', 'objectid', 'oid', 'sabl_pwsid', 'state_clas', 'county', 'proportion', 'state', 'tract', 'water_system_number',
    #                                            'water_system_name', 'ws_id', 'water_system_number', 'water_system_name', 'ave_red_lean_score', 'ave_score_red_lean_percentile', 'ave_overage_rate', 'overage_percentile']]

    df_wsp_score_census_columns = ['n_race', 'n_white_alone', 'n_black_alone', 'n_ai_and_an_alone', 'n_asian_alone', 'n_nh_and_opi_alone', 'n_other_alone', 'n_two_or_more_races',
                                   'hh_size', 'hh_1worker', 'hh_2worker', 'hh_3+worker', 'n_hh_3ppl', 'n_hh_4+ppl',
                                   'n_hh_type', 'n_hh_type_fam', 'n_hh_type_fam_mcf', 'n_hh_type_fam_mcf_1unit', 'n_hh_type_fam_mcf_2unit', 'n_hh_type_fam_mcf_mh_and_other', 'n_hh_type_fam_other', 'n_hh_type_fam_other_mhh_nsp', 'n_hh_type_fam_other_mhh_nsp_1unit', 'n_hh_type_fam_other_mhh_nsp_2unit', 'n_hh_type_fam_other_mhh_nsp_mh_and_other', 'n_hh_type_fam_other_fhh_nsp', 'n_hh_type_fam_other_fhh_nsp_1unit', 'n_hh_type_fam_other_fhh_nsp_2unit', 'n_hh_type_fam_other_fhh_nsp_mh_and_other', 'n_hh_type_nonfam', 'n_hh_type_nonfam_1unit', 'n_hh_type_nonfam_2unit', 'n_hh_type_nonfam_mh_and_other',
                                   'n_bachelors_deg', 'n_seng_compt_mat_stat_deg', 'n_seng_bio_ag_env_deg', 'n_seng_phys_sci_deg', 'n_seng_psych_deg', 'n_seng_soc_sci_deg', 'n_seng_eng_deg', 'n_seng_mds_deg', 'n_seng_rltd_deg', 'n_bus_deg', 'n_edu_deg', 'n_aho_lit_lang_deg', 'n_aho_lib_arts_and_hist_deg', 'n_aho_vis_perf_art_deg', 'n_aho_comm_deg', 'n_aho_other_deg',
                                   'n_hh_income', 'n_hh_income_lt_10k', 'n_hh_income_10k_15k', 'n_hh_income_15k_20k', 'n_hh_income_20k_25k', 'n_hh_income_25k_30k', 'n_hh_income_30k_35k', 'n_hh_income_35k_40k', 'n_hh_income_40k_45k', 'n_hh_income_45k_50k', 'n_hh_income_50k_60k', 'n_hh_income_60k_75k', 'n_hh_income_75k_100k', 'n_hh_income_100k_125k', 'n_hh_income_125k_150k', 'n_hh_income_150k_200k', 'n_hh_income_gt_200k',
                                   'n_hh_housing_units', 'n_hh_own', 'n_hh_rent',
                                   'n_rent_as_pct', 'n_rent_lt_10pct', 'n_rent_10_14.9pct', 'n_rent_15_19.9pct', 'n_rent_20_24.9pct', 'n_rent_25_29.9pct', 'n_rent_30_34.9pct', 'n_rent_35_39.9pct', 'n_rent_40_49.9pct', 'n_rent_gt_50pct', 'n_rent_not_computed',
                                   'n_insurance', 'n_have_insurance', 'n_no_insurance',
                                   'number_gw', 'number_sw',
                                   'regulating',
                                   'arealand', 'areawater',
                                               'population',
                                               'basename', 'centlat', 'centlon', 'funcstat', 'geoid', 'geo_id', 'hu100', 'intptlat', 'intptlon', 'lsadc', 'mtfcc', 'name', 'objectid', 'oid', 'sabl_pwsid', 'state_clas', 'county', 'proportion', 'state', 'tract', 'water_system_number',
                                               'water_system_name', 'ws_id', 'water_system_number', 'water_system_name', 'ave_red_lean_score', 'ave_score_red_lean_percentile', 'ave_overage_rate', 'overage_percentile']
    df_wsp_score_census_columns.extend(ws_mean_headers)
    df_wsp_score_census = df_wsp_score_census[df_wsp_score_census_columns]

    # Converting to fractions as some census questions may have varying answer rates
    # dataset_columns = ['fract_white_alone', 'fract_black_alone', 'fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races',
    #                    'fract_hh_1worker', 'fract_hh_2worker', 'fract_hh_3+worker', 'fract_hh_3ppl', 'fract_hh_4+ppl',
    #                    'fract_hh_type_fam', 'fract_hh_type_fam_mcf', 'fract_hh_type_fam_mcf_1unit', 'fract_hh_type_fam_mcf_2unit', 'fract_hh_type_fam_mcf_mh_and_other', 'fract_hh_type_fam_other', 'fract_hh_type_fam_other_mhh_nsp', 'fract_hh_type_fam_other_mhh_nsp_1unit', 'fract_hh_type_fam_other_mhh_nsp_2unit', 'fract_hh_type_fam_other_mhh_nsp_mh_and_other', 'fract_hh_type_fam_other_fhh_nsp', 'fract_hh_type_fam_other_fhh_nsp_1unit', 'fract_hh_type_fam_other_fhh_nsp_2unit', 'fract_hh_type_fam_other_fhh_nsp_mh_and_other', 'fract_hh_type_nonfam', 'fract_hh_type_nonfam_1unit', 'fract_hh_type_nonfam_2unit', 'fract_hh_type_nonfam_mh_and_other',
    #                    'fract_seng_compt_mat_stat_deg', 'fract_seng_bio_ag_env_deg', 'fract_seng_phys_sci_deg', 'fract_seng_psych_deg', 'fract_seng_soc_sci_deg', 'fract_seng_eng_deg', 'fract_seng_mds_deg', 'fract_seng_rltd_deg', 'fract_bus_deg', 'fract_edu_deg', 'fract_aho_lit_lang_deg', 'fract_aho_lib_arts_and_hist_deg', 'fract_aho_vis_perf_art_deg', 'fract_aho_comm_deg', 'fract_aho_other_deg',
    #                    'fract_hh_income_lt_10k', 'fract_hh_income_10k_15k', 'fract_hh_income_15k_20k', 'fract_hh_income_20k_25k', 'fract_hh_income_25k_30k', 'fract_hh_income_30k_35k', 'fract_hh_income_35k_40k', 'fract_hh_income_40k_45k', 'fract_hh_income_45k_50k', 'fract_hh_income_50k_60k', 'fract_hh_income_60k_75k', 'fract_hh_income_75k_100k', 'fract_hh_income_100k_125k', 'fract_hh_income_125k_150k', 'fract_hh_income_150k_200k', 'fract_hh_income_gt_200k',
    #                    'fract_hh_own', 'fract_hh_rent',
    #                    'fract_rent_lt_10pct', 'fract_rent_10_14.9pct', 'fract_rent_15_19.9pct', 'fract_rent_20_24.9pct', 'fract_rent_25_29.9pct', 'fract_rent_30_34.9pct', 'fract_rent_35_39.9pct', 'fract_rent_40_49.9pct', 'fract_rent_gt_50pct', 'fract_rent_not_computed',
    #                    'fract_have_insurance', 'fract_no_insurance',
    #                    'num_gw', 'num_sw',
    #                    'ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches',
    #                    'regulating',
    #                    'arealand', 'areawater',
    #                    'population',
    #                    'ws_id', 'water_system_number', 'water_system_name',
    #                    'ave_red_lean_score', 'ave_score_red_lean_percentile',
    #                    'ave_overage_rate', 'overage_percentile']
    dataset_columns = ['fract_white_alone', 'fract_black_alone', 'fract_ai_and_an_alone', 'fract_asian_alone', 'fract_nh_and_opi_alone', 'fract_other_alone', 'fract_two_or_more_races',
                       'fract_hh_1worker', 'fract_hh_2worker', 'fract_hh_3+worker', 'fract_hh_3ppl', 'fract_hh_4+ppl',
                       'fract_hh_type_fam', 'fract_hh_type_fam_mcf', 'fract_hh_type_fam_mcf_1unit', 'fract_hh_type_fam_mcf_2unit', 'fract_hh_type_fam_mcf_mh_and_other', 'fract_hh_type_fam_other', 'fract_hh_type_fam_other_mhh_nsp', 'fract_hh_type_fam_other_mhh_nsp_1unit', 'fract_hh_type_fam_other_mhh_nsp_2unit', 'fract_hh_type_fam_other_mhh_nsp_mh_and_other', 'fract_hh_type_fam_other_fhh_nsp', 'fract_hh_type_fam_other_fhh_nsp_1unit', 'fract_hh_type_fam_other_fhh_nsp_2unit', 'fract_hh_type_fam_other_fhh_nsp_mh_and_other', 'fract_hh_type_nonfam', 'fract_hh_type_nonfam_1unit', 'fract_hh_type_nonfam_2unit', 'fract_hh_type_nonfam_mh_and_other',
                       'fract_seng_compt_mat_stat_deg', 'fract_seng_bio_ag_env_deg', 'fract_seng_phys_sci_deg', 'fract_seng_psych_deg', 'fract_seng_soc_sci_deg', 'fract_seng_eng_deg', 'fract_seng_mds_deg', 'fract_seng_rltd_deg', 'fract_bus_deg', 'fract_edu_deg', 'fract_aho_lit_lang_deg', 'fract_aho_lib_arts_and_hist_deg', 'fract_aho_vis_perf_art_deg', 'fract_aho_comm_deg', 'fract_aho_other_deg',
                       'fract_hh_income_lt_10k', 'fract_hh_income_10k_15k', 'fract_hh_income_15k_20k', 'fract_hh_income_20k_25k', 'fract_hh_income_25k_30k', 'fract_hh_income_30k_35k', 'fract_hh_income_35k_40k', 'fract_hh_income_40k_45k', 'fract_hh_income_45k_50k', 'fract_hh_income_50k_60k', 'fract_hh_income_60k_75k', 'fract_hh_income_75k_100k', 'fract_hh_income_100k_125k', 'fract_hh_income_125k_150k', 'fract_hh_income_150k_200k', 'fract_hh_income_gt_200k',
                       'fract_hh_own', 'fract_hh_rent',
                       'fract_rent_lt_10pct', 'fract_rent_10_14.9pct', 'fract_rent_15_19.9pct', 'fract_rent_20_24.9pct', 'fract_rent_25_29.9pct', 'fract_rent_30_34.9pct', 'fract_rent_35_39.9pct', 'fract_rent_40_49.9pct', 'fract_rent_gt_50pct', 'fract_rent_not_computed',
                       'fract_have_insurance', 'fract_no_insurance',
                       'num_gw', 'num_sw',
                       'regulating',
                       'arealand', 'areawater',
                       'population',
                       'ws_id', 'water_system_number', 'water_system_name',
                       'ave_red_lean_score', 'ave_score_red_lean_percentile',
                       'ave_overage_rate', 'overage_percentile']
    dataset_columns.extend(ws_mean_headers)
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
        gw_sw = [j['number_gw'], j['number_sw']]
        # timeline_characteristics = [j['ave_target_timeline'], j['ave_method_priority_level'],
        #                             j['ave_num_time_segments'], j['ave_num_track_switches']]
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
        data_list.extend(gw_sw)
        # data_list.extend(timeline_characteristics)
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

    # variable_sets = ['regulating', 'race', 'hh_size', 'bdeg', 'hh_income', 'hh_own', 'rent_as_pct',
    #                  'insurance', 'gw_sw', 'timeline_characteristics', 'area', 'population']
    variable_sets = ['regulating', 'race', 'hh_size', 'bdeg', 'hh_income', 'hh_own', 'rent_as_pct',
                     'insurance', 'gw_sw', 'area', 'population', 'ws_contam_means']

    var_combinations = list()
    # print(var_combinations)
    for n in range(len(variable_sets) + 1):
        var_combinations += list(combinations(variable_sets, n))
    #     print(list(combinations(variable_sets, n)))
    #     print(list(tuple(list(combinations(variable_sets, n)))))

    # raise ValueError
    var_combinations.remove(())
    print(
        f'Powerset has {len(var_combinations)} values and took: {time.perf_counter()-start}')
    # print(var_combinations)
    # raise ValueError
    # Start with the heaviest regressions first to find if hyperparameters need to be cut
    var_combinations = list(reversed(var_combinations))
    var_comb_sublists = []
    sublist_size = 60
    for i in range(0, len(var_combinations), sublist_size):
        new_var_sublist = var_combinations[i:i+sublist_size]
        var_comb_sublists.append(new_var_sublist)

    for i in var_comb_sublists:
        print(len(i))

    # ('hh_size', 'bdeg', 'insurance', 'gw_sw', 'timeline_characteristics')
    # Should have r2 of 0.728174973621484 & adj_r2 of 0.719301468951892
    test = data_and_regression_selector(
        dataset, ('ws_contam_means'), 'compliance_score', ws_mean_headers)
    print(test)
    raise ValueError

    dependent_vars_to_test = [
        'compliance_score', 'compliance_percentile', 'overage_rate', 'overage_percentile']
    for dependent in dependent_vars_to_test:
        sublist_times = []
        dependent_index = dependent_vars_to_test.index(dependent)
        for sublist in var_comb_sublists:

            sublist_index = var_comb_sublists.index(sublist)
            print(sublist_index)

            # regression_columns = ['independent_variables', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'regression',
            #                       'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
            #                       'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'split5_test_score', 'split6_test_score', 'split7_test_score', 'split8_test_score', 'split9_test_score']
            # regression_outputs = []
            # regression_start = time.perf_counter()
            # with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
            #     results = executor.map(data_and_regression_selector, [
            #         dataset]*len(sublist), sublist, [dependent]*len(sublist))
            #     end_results_creation = time.perf_counter()
            #     print(
            #         f'Results creation: {end_results_creation-regression_start}')
            #     for result in results:
            #         regression_outputs.extend(result)
            #         # print(
            #         #     f'Time: {str(time.perf_counter() - end_results_creation)}, Len of result: {str(len(result))}, Len of output:{str(len(regression_outputs))}')
            #     print(
            #         f'Results iteration: {time.perf_counter()-end_results_creation}')

            regression_columns = ['independent_variables', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'regression',
                                  'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
                                  'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'split5_test_score', 'split6_test_score', 'split7_test_score', 'split8_test_score', 'split9_test_score']
            regression_outputs = []
            sublist_start = time.perf_counter()
            for var_combo in sublist:
                print(f'Trying: {var_combo}')
                regression_outputs.extend(
                    data_and_regression_selector(dataset, var_combo, dependent, ws_mean_headers))
            sublist_times.append(time.perf_counter() - sublist_start)
            print(
                f'\n\n\nSublist {sublist_index} finished in (seconds): {time.perf_counter() - sublist_start}')
            print(
                f'All sublist times for this dep variable (seconds): {sublist_times}\n\n')

            df_reg = pd.DataFrame(regression_outputs,
                                  columns=regression_columns)
            df_reg['dependent_variable'] = dependent
            append_or_replace = (
                'replace')*(sublist_index + dependent_index == 0) + ('append')*(sublist_index + dependent_index > 0)

            conn = wdc.sql_query_conn()
            df_reg.to_sql('ml_regressions_gridsearch', conn,
                          if_exists=append_or_replace, index=False)
            conn.close()

    wdc.create_index('ml_regressions_gridsearch', independent_variables='ASC')
    wdc.create_index('ml_regressions_gridsearch', params='ASC')
    wdc.create_index('ml_regressions_gridsearch', mean_test_score='ASC')
    wdc.create_index('ml_regressions_gridsearch', regression='ASC')

    finish = time.perf_counter()
    print(f'Seconds: {finish - start}')
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


# Seconds: 69684.5779637
