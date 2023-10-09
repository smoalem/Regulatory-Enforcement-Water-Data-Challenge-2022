from ipaddress import AddressValueError
from multiprocessing.sharedctypes import Value
from pickletools import read_decimalnl_long
from threading import currentThread
from weakref import ref
import pandas as pd
import numpy as np
import time
import concurrent.futures
import cProfile
import pstats
import io
from pstats import SortKey
from math import sqrt
import regex as re
import wdc_lib
from pstats import SortKey
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from itertools import combinations
from scipy.stats import expon # import exponential function from scipy stats
import warnings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    return df_gridsearch

def logarithmic_regression(X, y):
    # # # Multiple Linear Regression:
    # Splitting the dataset into the Training set and Test set
    # y_log = np.log(y + 1)
    y_log = np.log(y + 0.01) 
    # modify y log 
    # y_log = expon.pdf(X, scale=2) + 1 * 100

    # raise ValueError
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.3, random_state=83)

    # Gridsearch Cross-Validation
    regressor = LinearRegression()
    # Define the hyperparameter grid to search
    param_grid = {
        'fit_intercept': [True, False],  # Whether to fit the intercept or not
        'normalize': [True, False]       # Whether to normalize the input features or not
    }
    # param_grid = {}
    # from sklearn.model_selection import RandomizedSearchCV
    # grid_search = RandomizedSearchCV(regressor,
    #                            param_grid,
    #                    scoring='r2', cv=7, n_jobs=-1, n_iter=1000)

    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=param_grid,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1)  # n_jobs = -1 means it will use all CPU cores
    grid_search.fit(X_train, y_train)
    df_gridsearch = pd.DataFrame(grid_search.cv_results_)
    df_gridsearch['params'] = df_gridsearch['params'].astype("str")
    df_gridsearch['regression'] = 'log'

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
    print([r2, adj_r2,mae,mape,mse,rmse])
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
    parameters = [{'kernel': ['linear'], 'C': [0.01, 0.05]},
                  {'kernel': ['rbf', 'sigmoid'], 'gamma': [
                      0.0001, 0.01, 0.1, 1, 5, 10], 'C': [0.01, 0.05]},
                  {'kernel': ['poly'], 'gamma': [0.0001, 0.001, 0.005],  'C': [0.01, 0.05], 'degree': [2, 3]}]
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

def color_ratio_reg_contam(reg_contam):
    reg = reg_contam[0]
    contam = reg_contam[1]

    color_dictionary = {"RED": 0, "YELLOW": 0, "GREEN": 0, "BLACK": 0, "TBD": 0}
    # total_days = 0
    # green_days = 0
    # red_days = 0
    # yellow_days = 0

    query = f"""SELECT * FROM user_effective_timeline WHERE regulation= "{reg}" AND contam_id={contam}"""
    conn = wdc_lib.sql_query_conn()
    df_reg_contam = pd.read_sql_query(query, conn)
    conn.close()

    # print(df_reg_contam)

    for i, j in df_reg_contam.iterrows():
        color_dictionary[j["color"]] += j["end_date"] - j["start_date"]

    # print(color_dictionary)

    green_red_yellow_total_days = (
        color_dictionary["GREEN"] + color_dictionary["RED"] + color_dictionary["YELLOW"]
    )

    if green_red_yellow_total_days > 0:
        green_fraction = 0
        red_fraction = 0
        yellow_fraction = 0
        if color_dictionary["GREEN"] > 0:
            green_fraction = color_dictionary["GREEN"] / green_red_yellow_total_days
        if color_dictionary["RED"] > 0:
            red_fraction = color_dictionary["RED"] / green_red_yellow_total_days
        if color_dictionary["YELLOW"] > 0:
            yellow_fraction = color_dictionary["YELLOW"] / green_red_yellow_total_days
        return [reg, contam, green_fraction, red_fraction, yellow_fraction]
    else:
        return []


# test_time = time.perf_counter()
# print(color_ratio_reg_contam(["CA Title 22 subsec 64445", 100]))
# print(time.perf_counter() - test_time)


if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    # start = time.perf_counter()

    # reg_query = "SELECT DISTINCT regulation FROM user_effective_timeline"
    # contam_query = "SELECT DISTINCT contam_id FROM user_effective_timeline"
    # conn = wdc_lib.sql_query_conn()
    # reg_list = pd.read_sql_query(reg_query, conn)["regulation"].values.tolist()
    # contam_list = pd.read_sql_query(contam_query, conn)["contam_id"].values.tolist()
    # conn.close()

    # reg_list.remove("NA")
    # reg_list.remove("General practice")

    # print(reg_list)
    # print(contam_list)
    # reg_contam_list = []

    # for r in reg_list:
    #     for c in contam_list:
    #         reg_contam_list.append([r, c])

    # color_ratio_columns = [
    #     "regulation",
    #     "contam_id",
    #     "green_fraction",
    #     "red_fraction",
    #     "yellow_fraction",
    # ]
    # color_ratio_data = []
    # with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
    #     results = executor.map(color_ratio_reg_contam, reg_contam_list)
    #     end_results_creation = time.perf_counter()
    #     print(f"Results creation: {end_results_creation-start}")
    #     for result in results:
    #         if len(result) > 0:
    #             color_ratio_data.append(result)

        

    #     color_ratio_df = pd.DataFrame(color_ratio_data, columns=color_ratio_columns)

    #     conn = wdc_lib.sql_query_conn()
    #     color_ratio_df.to_sql(
    #         "color_ratio_data", conn, if_exists="replace", index=False
    #     )
    #     conn.close()

    reg_complexity_dict = { 
        "CA Title 22 subsec 64445.1(c)(7)": (1, 1/365),
        "CA Title 22 subsec 64445.1(b)(1)": (1, 1),
        "CA Title 22 subsec 64445.1(b)(3)": (1, 1),
        "CA Title 22 subsec 64445.1(c)(4)": (2, 0.5),
        "CA Title 22 subsec 64445.1(b)(2)": (2, 1),
        "CA Title 22 subsec 64445.1(c)(5)(A)": (6, 0.5),
        "CA Title 22 subsec 64445.1(c)(5)(C)": (5, 0.75),
        "CA Title 22 subsec 64445.1(c)(5)(B)": (4, 1),
        "CA Title 22 subsec 64445": (4, 3)
    }
    conn = wdc_lib.sql_query_conn()
    color_ratio_list = pd.read_sql("SELECT * FROM color_ratio_data", conn).values.tolist()
    contam_info_df = pd.read_sql("SELECT * FROM contam_info", conn)
    # print(color_ratio_list)
    # raise ValueError
    conn.close()
    color_with_complexity_contam_list = []
    for color in color_ratio_list:
        min_samples = reg_complexity_dict[color[0]][0]
        years_reviewed = reg_complexity_dict[color[0]][1]
        color.insert(1, min_samples)
        color.insert(2, years_reviewed)
        contaminant = contam_info_df[ contam_info_df["id"] == color[3] ][["reg_xldate", "contam_group","method", "unit", "dlr", "mcl", "min_xldate", "max_xldate", "unique_sampled_and_reviewed_facilities"]].values.tolist()
        if contaminant[0][3] != "PG/L":
            contaminant[0][4] *= 1_000_000
            contaminant[0][5] *= 1_000_000
        del contaminant[0][3]
        # print(color)
        # print(contaminant[0])
        color_w_complexity_contam = color[:4] + contaminant[0] + color[4:]
        color_with_complexity_contam_list.append(color_w_complexity_contam)
        color_w_complexity_contam.insert(0, min_samples * years_reviewed)

        # print(len(color_w_complexity_contam))
        # if len(color_w_complexity_contam) == 16:
        #     print(color_w_complexity_contam)
        # regulation, samples, 


    reg_contam_df = pd.DataFrame(color_with_complexity_contam_list, columns=["Complexity", "regulation", "min_reg_samples", "reg_time_span", "contam_id", "reg_xldate", "contam_group","method", "dlr", "mcl", "min_xldate", "max_xldate", "unique_sampled_and_reviewed_facilities", "green", "red", "yellow"])
    convert_dict = {"contam_id": str, "reg_xldate": np.int64, "min_xldate": np.int64, "max_xldate": np.int64 }
    reg_contam_df = reg_contam_df.astype(convert_dict)

    
    # variable_sets_dict = {'regulating': ["regulation", "min_reg_samples", "reg_time_span"] ,'contam':["contam_id", "reg_xldate", "contam_group","method", "dlr", "mcl", "min_xldate", "max_xldate", "unique_sampled_and_reviewed_facilities"] }

    # variable_sets = ['regulating', 'contam']

    # var_combinations = list()
    # for n in range(len(variable_sets) + 1):
    #     var_combinations += list(combinations(variable_sets, n))

    # var_combinations.remove(())
 
    # print(var_combinations)
    # raise ValueError

    # X_indices = [[]]




    # X_vars = [(0, 4, [1]), (4,-3, [0,2,3]), (0, -3, [1,4,6,7])]
    df_gridsearch_results = []
    # SUCCESS?
    # X = reg_contam_df.iloc[:, var[0]:var[1]].values
    # X = reg_contam_df.iloc[:, [0,1,4]].values # complexity, reg, contam id - 14  decision_tree  5.912e-01
    X = reg_contam_df.iloc[:, [0,1]].values # complexity, reg - .log - 0.7276 
    # X = reg_contam_df.iloc[:, [0]].values # complexity- log - 0.64 
    X = reg_contam_df.iloc[:, [1]].values # reg - log .7276 same as 0, 1
    X = reg_contam_df.iloc[:, [1, 9]].values # reg, mcl - log .7265
    X = reg_contam_df.iloc[:, [1, 8]].values # reg, dlr - log               False           False        7.283e-01
    X = reg_contam_df.iloc[:, [1, 8, 9]].values # reg, dlr mcl -  log               False           False        7.275e-01
    X = reg_contam_df.iloc[:, [1, 6]].values # regulation and contam group - log - .7241
    X = reg_contam_df.iloc[:, [1, 5]].values # regulation and reg xladate - log - .7136
    X = reg_contam_df.iloc[:, [0, 1, 5, 6, 8, 9]].values # complexity, reg, reg_xldate, contam group, dlr, mcl - log  w/ fit intcpt True        7.287e-01
    y = reg_contam_df.iloc[:, -3].values 
    print(X)
    print(len(X[0]))
    print(y)
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(
            sparse=False), [1, 3])], remainder='passthrough')    

    # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(
    #         sparse=False), [0])], remainder='passthrough')    
    X = np.array(ct.fit_transform(X))
    # print("TRANSFORMED")
    # print(X)
    # print(len(X[0]))
    # Taking care of missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:])
    # Transform method then does the replacement of all the nan with mean
    X[:] = imputer.transform(X[:])

    times = []
    start_regs = time.perf_counter()


    df_gridsearch_results.append(linear_regression(X, y))

    linear_fin = time.perf_counter()
    times.append(linear_fin - start_regs)
    # print(reg_contam_df)
    # print(reg_contam_df.dtypes)


    df_gridsearch_results.append(logarithmic_regression(X, y))
    log_fin = time.perf_counter()
    times.append(log_fin - linear_fin)
    print(f'log_reg took: {times[-1]}')
    
    df_gridsearch_results.append(polynomial_regression(X, y))

    poly_fin = time.perf_counter()
    times.append(poly_fin - log_fin)
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
    print("CONCAT")

    with pd.option_context('display.max_rows', None,
                    'display.max_columns', None,
                    'display.precision', 3,
                    ):
        print(df_all_gridsearch[['regression', 'param_fit_intercept', 'param_normalize', "mean_test_score"]].to_string())
    # print(df_all_gridsearch.columns.tolist())
    print("Total time:")
    print(time.perf_counter() - start)
  

