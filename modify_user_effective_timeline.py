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

pr = cProfile.Profile()
pr.enable()
start = time.perf_counter()

uef_query = "SELECT * FROM user_effective_timeline"
contam_query = "SELECT * FROM contam_info"
fac_df = wdc_lib.facilities_to_review()
conn = wdc_lib.sql_query_conn()
uef_df = pd.read_sql_query(uef_query, conn)
contam_df = pd.read_sql_query(contam_query, conn)
conn.close()
uef_contam_df = pd.merge(uef_df, contam_df, how='left', left_on=['contam_id'], right_on=['id'])
uef_contam_fac_df = pd.merge(uef_contam_df, fac_df, how='left', left_on=['fac_id'], right_on=['id'])

conn = wdc_lib.sql_query_conn()
uef_contam_fac_df.to_sql(
    "user_effective_timeline_w_contam_fac", conn, if_exists="replace", index=False
)
conn.close()

print(time.perf_counter() - start)
print(uef_contam_fac_df)