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
import tensorflow_addons as tfa

msle = tf.keras.losses.MeanSquaredLogarithmicError()

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

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


@jit(forceobj=True, parallel=True)
def build_model(hp):
    model = tf.keras.Sequential()
    r_square = tfa.metrics.r_square.RSquare()
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units1 = hp.Int('units1', min_value=32, max_value=200, step=1)
    hp_units2 = hp.Int('units2', min_value=32, max_value=200, step=1)
    hp_units3 = hp.Int('units3', min_value=32, max_value=200, step=1)
    hp_units4 = hp.Int('units4', min_value=32, max_value=200, step=1)
    hp_units5 = hp.Int('units5', min_value=32, max_value=200, step=1)
    # hp_units6 = hp.Int('units6', min_value=32, max_value=200, step=1)
    # hp_units7 = hp.Int('units7', min_value=32, max_value=128, step=1)
    # hp_units8 = hp.Int('units8', min_value=32, max_value=128, step=1)
    # hp_units9 = hp.Int('units9', min_value=32, max_value=128, step=1)
    # hp_units10 = hp.Int('units10', min_value=32, max_value=128, step=1)
    # hp_units11 = hp.Int('units11', min_value=32,  max_value=512, step=1)
    # hp_units12 = hp.Int('units12', min_value=32,  max_value=512, step=1)
    # hp_units13 = hp.Int('units13', min_value=32,  max_value=512, step=1)
    # hp_units14 = hp.Int('units14', min_value=32,  max_value=512, step=1)
    # hp_units15 = hp.Int('units15', min_value=32,  max_value=512, step=1)
    # hp_units16 = hp.Int('units16', min_value=32,  max_value=512, step=1)
    # hp_units17 = hp.Int('units17', min_value=32,  max_value=512, step=1)
    # hp_units18 = hp.Int('units18', min_value=32,  max_value=512, step=1)
    # hp_units19 = hp.Int('units19', min_value=32,  max_value=512, step=1)
    # hp_units20 = hp.Int('units20', min_value=32,  max_value=512, step=1)
    # hp_units21 = hp.Int('units21', min_value=32,  max_value=512, step=1)
    # hp_units22 = hp.Int('units22', min_value=32,  max_value=512, step=1)
    # hp_units23 = hp.Int('units23', min_value=32,  max_value=512, step=1)
    # hp_units24 = hp.Int('units24', min_value=32,  max_value=512, step=1)
    # hp_units25 = hp.Int('units25', min_value=32,  max_value=512, step=1)
    # hp_units26 = hp.Int('units26', min_value=32,  max_value=512, step=1)
    # hp_units27 = hp.Int('units27', min_value=32,  max_value=512, step=1)
    # hp_units28 = hp.Int('units28', min_value=32,  max_value=512, step=1)
    # hp_units29 = hp.Int('units29', min_value=32,  max_value=512, step=1)
    # hp_units30 = hp.Int('units30', min_value=32,  max_value=512, step=1)
    # hp_units31 = hp.Int('units31', min_value=32,  max_value=512, step=1)
    # hp_units32 = hp.Int('units32', min_value=32,  max_value=512, step=1)
    # hp_units33 = hp.Int('units33', min_value=32,  max_value=512, step=1)
    # hp_units34 = hp.Int('units34', min_value=32,  max_value=512, step=1)
    # hp_units35 = hp.Int('units35', min_value=32,  max_value=512, step=1)
    # hp_units36 = hp.Int('units36', min_value=32,  max_value=512, step=1)
    # hp_units37 = hp.Int('units37', min_value=32,  max_value=512, step=1)
    # hp_units38 = hp.Int('units38', min_value=32,  max_value=512, step=1)
    # hp_units39 = hp.Int('units39', min_value=32,  max_value=512, step=1)
    # hp_units40 = hp.Int('units40', min_value=32,  max_value=512, step=1)
    # hp_units41 = hp.Int('units41', min_value=32,  max_value=512, step=1)
    # hp_units42 = hp.Int('units42', min_value=32,  max_value=512, step=1)
    # hp_units43 = hp.Int('units43', min_value=32,  max_value=512, step=1)
    # hp_units44 = hp.Int('units44', min_value=32,  max_value=512, step=1)
    # hp_units45 = hp.Int('units45', min_value=32,  max_value=512, step=1)
    # hp_units46 = hp.Int('units46', min_value=32,  max_value=512, step=1)
    # hp_units47 = hp.Int('units47', min_value=32,  max_value=512, step=1)
    # hp_units48 = hp.Int('units48', min_value=32,  max_value=512, step=1)
    # hp_units49 = hp.Int('units49', min_value=32,  max_value=512, step=1)
    # hp_units50 = hp.Int('units50', min_value=32,  max_value=512, step=1)
    # hp_units51 = hp.Int('units51', min_value=32,  max_value=512, step=1)
    # hp_units52 = hp.Int('units52', min_value=32,  max_value=512, step=1)
    # hp_units53 = hp.Int('units53', min_value=32,  max_value=512, step=1)
    # hp_units54 = hp.Int('units54', min_value=32,  max_value=512, step=1)
    # hp_units55 = hp.Int('units55', min_value=32,  max_value=512, step=1)
    # hp_units56 = hp.Int('units56', min_value=32,  max_value=512, step=1)
    # hp_units57 = hp.Int('units57', min_value=32,  max_value=512, step=1)
    # hp_units58 = hp.Int('units58', min_value=32,  max_value=512, step=1)
    # hp_units59 = hp.Int('units59', min_value=32,  max_value=512, step=1)
    # hp_units60 = hp.Int('units60', min_value=32,  max_value=512, step=1)
    # hp_units61 = hp.Int('units61', min_value=32,  max_value=512, step=1)
    # hp_units62 = hp.Int('units62', min_value=32,  max_value=512, step=1)
    # hp_units63 = hp.Int('units63', min_value=32,  max_value=512, step=1)
    # hp_units64 = hp.Int('units64', min_value=32,  max_value=512, step=1)
    # hp_units65 = hp.Int('units65', min_value=32,  max_value=512, step=1)
    # hp_units66 = hp.Int('units66', min_value=32,  max_value=512, step=1)
    # hp_units67 = hp.Int('units67', min_value=32,  max_value=512, step=1)
    # hp_units68 = hp.Int('units68', min_value=32,  max_value=512, step=1)
    # hp_units69 = hp.Int('units69', min_value=32,  max_value=512, step=1)
    # hp_units70 = hp.Int('units70', min_value=32,  max_value=512, step=1)
    # hp_units71 = hp.Int('units71', min_value=32,  max_value=512, step=1)
    # hp_units72 = hp.Int('units72', min_value=32,  max_value=512, step=1)
    # hp_units73 = hp.Int('units73', min_value=32,  max_value=512, step=1)
    # hp_units74 = hp.Int('units74', min_value=32,  max_value=512, step=1)
    # hp_units75 = hp.Int('units75', min_value=32,  max_value=512, step=1)
    # hp_units76 = hp.Int('units76', min_value=32,  max_value=512, step=1)
    # hp_units77 = hp.Int('units77', min_value=32,  max_value=512, step=1)
    # hp_units78 = hp.Int('units78', min_value=32,  max_value=512, step=1)
    # hp_units79 = hp.Int('units79', min_value=32,  max_value=512, step=1)
    # hp_units80 = hp.Int('units80', min_value=32,  max_value=512, step=1)
    # hp_units81 = hp.Int('units81', min_value=32,  max_value=512, step=1)
    # hp_units82 = hp.Int('units82', min_value=32,  max_value=512, step=1)
    # hp_units83 = hp.Int('units83', min_value=32,  max_value=512, step=1)
    # hp_units84 = hp.Int('units84', min_value=32,  max_value=512, step=1)
    # hp_units85 = hp.Int('units85', min_value=32,  max_value=512, step=1)
    # hp_units86 = hp.Int('units86', min_value=32,  max_value=512, step=1)
    # hp_units87 = hp.Int('units87', min_value=32,  max_value=512, step=1)
    # hp_units88 = hp.Int('units88', min_value=32,  max_value=512, step=1)
    # hp_units89 = hp.Int('units89', min_value=32,  max_value=512, step=1)
    # hp_units90 = hp.Int('units90', min_value=32,  max_value=512, step=1)
    # hp_units91 = hp.Int('units91', min_value=32,  max_value=512, step=1)
    # hp_units92 = hp.Int('units92', min_value=32,  max_value=512, step=1)
    # hp_units93 = hp.Int('units93', min_value=32,  max_value=512, step=1)
    # hp_units94 = hp.Int('units94', min_value=32,  max_value=512, step=1)
    # hp_units95 = hp.Int('units95', min_value=32,  max_value=512, step=1)
    # hp_units96 = hp.Int('units96', min_value=32,  max_value=512, step=1)
    # hp_units97 = hp.Int('units97', min_value=32,  max_value=512, step=1)
    # hp_units98 = hp.Int('units98', min_value=32,  max_value=512, step=1)
    # hp_units99 = hp.Int('units99', min_value=32,  max_value=512, step=1)
    # hp_units100 = hp.Int('units100', min_value=32,  max_value=512, step=1)
    

    model.add(tf.keras.layers.Dense(units=hp_units1, input_dim=68, activation=hp.Choice(
                    'dense_activation_1',
                    values=["relu", "relu6", "elu", "selu", "swish"],
                    default='relu')))
    model.add(tf.keras.layers.Dense(units=hp_units2, activation=hp.Choice(
                    'dense_activation_2',
                    values=["relu", "relu6", "elu", "selu", "swish"],
                    default='relu')))
    model.add(tf.keras.layers.Dense(units=hp_units3, activation=hp.Choice(
                    'dense_activation_3',
                    values=["relu", "relu6", "elu", "selu", "swish"],
                    default='relu')))
    model.add(tf.keras.layers.Dense(units=hp_units4, activation=hp.Choice(
                    'dense_activation_4',
                    values=["relu", "relu6", "elu", "selu", "swish"],
                    default='relu')))
    model.add(tf.keras.layers.Dense(units=hp_units5, activation=hp.Choice(
                    'dense_activation_5',
                    values=["relu", "relu6", "elu", "selu", "swish"],
                    default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units6, activation=hp.Choice(
    #                 'dense_activation_6',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units7, activation=hp.Choice(
    #                 'dense_activation_7',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units8, activation=hp.Choice(
    #                 'dense_activation_8',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units9, activation=hp.Choice(
    #                 'dense_activation_9',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units10, activation=hp.Choice(
    #                 'dense_activation_10',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units11, activation=hp.Choice(
    #                 'dense_activation_11',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units12, activation=hp.Choice(
    #                 'dense_activation_12',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units13,  activation=hp.Choice(
    #                 'dense_activation_13',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units14,  activation=hp.Choice(
    #                 'dense_activation_14',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units15,  activation=hp.Choice(
    #                 'dense_activation_15',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units16,  activation=hp.Choice(
    #                 'dense_activation_16',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units17,  activation=hp.Choice(
    #                 'dense_activation_17',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units18,  activation=hp.Choice(
    #                 'dense_activation_18',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units19,  activation=hp.Choice(
    #                 'dense_activation_19',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units20,  activation=hp.Choice(
    #                 'dense_activation_20',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units21,  activation=hp.Choice(
    #                 'dense_activation_21',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units22,  activation=hp.Choice(
    #                 'dense_activation_22',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units23,  activation=hp.Choice(
    #                 'dense_activation_23',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units24,  activation=hp.Choice(
    #                 'dense_activation_24',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units25,  activation=hp.Choice(
    #                 'dense_activation_25',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units26,  activation=hp.Choice(
    #                 'dense_activation_26',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units27,  activation=hp.Choice(
    #                 'dense_activation_27',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units28,  activation=hp.Choice(
    #                 'dense_activation_28',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units29,  activation=hp.Choice(
    #                 'dense_activation_29',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units30,  activation=hp.Choice(
    #                 'dense_activation_30',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units31,  activation=hp.Choice(
    #                 'dense_activation_31',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units32,  activation=hp.Choice(
    #                 'dense_activation_32',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units33,  activation=hp.Choice(
    #                 'dense_activation_33',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units34,  activation=hp.Choice(
    #                 'dense_activation_34',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units35,  activation=hp.Choice(
    #                 'dense_activation_35',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units36, activation=hp.Choice(
    #                 'dense_activation_36',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units37, activation=hp.Choice(
    #                 'dense_activation_37',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units38,  activation=hp.Choice(
    #                 'dense_activation_38',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units39,  activation=hp.Choice(
    #                 'dense_activation_39',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units40,  activation=hp.Choice(
    #                 'dense_activation_40',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units41,  activation=hp.Choice(
    #                 'dense_activation_41',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units42,  activation=hp.Choice(
    #                 'dense_activation_42',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units43,  activation=hp.Choice(
    #                 'dense_activation_43',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units44,  activation=hp.Choice(
    #                 'dense_activation_44',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units45,  activation=hp.Choice(
    #                 'dense_activation_45',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units46,  activation=hp.Choice(
    #                 'dense_activation_46',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units47,  activation=hp.Choice(
    #                 'dense_activation_47',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units48,  activation=hp.Choice(
    #                 'dense_activation_48',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units49,  activation=hp.Choice(
    #                 'dense_activation_49',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units50,  activation=hp.Choice(
    #                 'dense_activation_50',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units51,  activation=hp.Choice(
    #                 'dense_activation_51',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units52,  activation=hp.Choice(
    #                 'dense_activation_52',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units53,  activation=hp.Choice(
    #                 'dense_activation_53',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units54,  activation=hp.Choice(
    #                 'dense_activation_54',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units55,  activation=hp.Choice(
    #                 'dense_activation_55',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units56,  activation=hp.Choice(
    #                 'dense_activation_56',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units57,  activation=hp.Choice(
    #                 'dense_activation_57',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units58,  activation=hp.Choice(
    #                 'dense_activation_58',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units59,  activation=hp.Choice(
    #                 'dense_activation_59',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units60,  activation=hp.Choice(
    #                 'dense_activation_60',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units61, activation=hp.Choice(
    #                 'dense_activation_61',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units62, activation=hp.Choice(
    #                 'dense_activation_62',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units63, activation=hp.Choice(
    #                 'dense_activation_63',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units64, activation=hp.Choice(
    #                 'dense_activation_64',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units65, activation=hp.Choice(
    #                 'dense_activation_65',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units66, activation=hp.Choice(
    #                 'dense_activation_66',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units67, activation=hp.Choice(
    #                 'dense_activation_67',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units68, activation=hp.Choice(
    #                 'dense_activation_68',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units69, activation=hp.Choice(
    #                 'dense_activation_69',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units70, activation=hp.Choice(
    #                 'dense_activation_70',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu'))) 
    # model.add(tf.keras.layers.Dense(units=hp_units71, activation=hp.Choice(
    #                 'dense_activation_71',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units72, activation=hp.Choice(
    #                 'dense_activation_72',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units73, activation=hp.Choice(
    #                 'dense_activation_73',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units74, activation=hp.Choice(
    #                 'dense_activation_74',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units75, activation=hp.Choice(
    #                 'dense_activation_75',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units76, activation=hp.Choice(
    #                 'dense_activation_76',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units77, activation=hp.Choice(
    #                 'dense_activation_77',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units78, activation=hp.Choice(
    #                 'dense_activation_78',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units79, activation=hp.Choice(
    #                 'dense_activation_79',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units80, activation=hp.Choice(
    #                 'dense_activation_80',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu'))) 
    # model.add(tf.keras.layers.Dense(units=hp_units81, activation=hp.Choice(
    #                 'dense_activation_81',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units82, activation=hp.Choice(
    #                 'dense_activation_82',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units83, activation=hp.Choice(
    #                 'dense_activation_83',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units84, activation=hp.Choice(
    #                 'dense_activation_84',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units85, activation=hp.Choice(
    #                 'dense_activation_85',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units86, activation=hp.Choice(
    #                 'dense_activation_86',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units87, activation=hp.Choice(
    #                 'dense_activation_87',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units88, activation=hp.Choice(
    #                 'dense_activation_88',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units89, activation=hp.Choice(
    #                 'dense_activation_89',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units90, activation=hp.Choice(
    #                 'dense_activation_90',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu'))) 
    # model.add(tf.keras.layers.Dense(units=hp_units91, activation=hp.Choice(
    #                 'dense_activation_91',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units92, activation=hp.Choice(
    #                 'dense_activation_92',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units93, activation=hp.Choice(
    #                 'dense_activation_93',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units94, activation=hp.Choice(
    #                 'dense_activation_94',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units95, activation=hp.Choice(
    #                 'dense_activation_95',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units96, activation=hp.Choice(
    #                 'dense_activation_96',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units97, activation=hp.Choice(
    #                 'dense_activation_97',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units98, activation=hp.Choice(
    #                 'dense_activation_98',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))           
    # model.add(tf.keras.layers.Dense(units=hp_units99, activation=hp.Choice(
    #                 'dense_activation_99',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu')))
    # model.add(tf.keras.layers.Dense(units=hp_units100, activation=hp.Choice(
    #                 'dense_activation_100',
    #                 values=["relu", "relu6", "elu", "selu", "swish"],
    #                 default='relu'))) 


    model.add(tf.keras.layers.Dense(1))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        # optimizer="adam",
        loss="mean_squared_error",
        metrics=[r_square, "mean_squared_error"]
    )

    return model

# HyperBand algorithm from keras tuner
# tuner = kt.BayesianOptimization(
#     build_model,
#     objective= kt.Objective('val_r_square', direction="max"),
#     max_trials=100,
#     seed = 42,
#     directory='keras_tuner_dir_ann_for_reg_enforce_test_28',
#     project_name='keras_tuner_test_run_28'
# )

tuner = kt.Hyperband(
    build_model,
    objective= kt.Objective("val_r_square", direction="max"),
    max_epochs=100,
    seed = 42,
    directory='keras_tuner_dir_ann_for_reg_enforce_test_50',
    project_name='keras_tuner_test_run_50'
)


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


    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

    best_model = tuner.get_best_models()[0]
    best_model.build(X_train.shape)
    print("SUMMARY")
    print("###############################")

    best_model.summary()
    print("###############################")

    best_model.fit(
        X_train, 
        y_train,
        epochs=34,
        batch_size=1
    )
    y_pred = best_model.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1),
          y_test.reshape(len(y_test), 1)), 1))
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(y_test, y_pred)
    print(metric.result().numpy())
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1-(1-r2)*(len(y)-1)/(len(y)-1-X.shape[1])
    print(f"r2 / adj r2: {r2} / {adj_r2}")
    # r2_list.append((r2, adj_r2))
    # msle(y_test, best_model.predict(X_train)).numpy()
