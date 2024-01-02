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
from scipy.stats import expon  # import exponential function from scipy stats

pr = cProfile.Profile()
pr.enable()
start = time.perf_counter()

uef_query = "SELECT * FROM user_effective_timeline_w_contam_fac"
contam_query = "SELECT * FROM contam_info"
fac_df = wdc_lib.facilities_to_review()


conn = wdc_lib.sql_query_conn()
# uef_df = pd.read_sql_query(uef_query, conn)


uef_contam_df = pd.read_sql_query(uef_query, conn)


contam_df = pd.read_sql_query(contam_query, conn)
conn.close()


# uef_contam_df = pd.merge(
#     uef_df, contam_df, how="left", left_on=["contam_id"], right_on=["id"]
# )
# uef_contam_fac_df = pd.merge(
#     uef_contam_df, fac_df, how="left", left_on=["fac_id"], right_on=["id"]
# )


def ax_or_red_older(ax, red):
    ax = int(ax)
    red = int(red)
    return "red" * (ax >= red) + "ax" * (ax < red)


def pop_above_below_3300(pserved):
    pserved = int(pserved)
    if pserved > 3300:
        return "greater"
    else:
        return "lesser"


def gw_sw_determiner(fac_type, pswt):
    fac_water_check = wdc_lib.water_system_type(fac_type)
    if fac_water_check == "pswt":
        return pswt
    else:
        return fac_water_check


# df['new_stuff'] = df.apply(lambda row: calculate_c(row['column1'], row['column2']), axis=1)


# uef_contam_df["pop_above_or_below_3300"] = uef_contam_df["pserved"].apply(
#     pop_above_below_3300
# )
uef_contam_df["pop_above_or_below_3300"] = uef_contam_df.apply(
    lambda row: pop_above_below_3300(row["pserved"]), axis=1
)
print(f"done with pop in {time.perf_counter() - start}")
print(uef_contam_df)

# uef_contam_df["is_ax_or_red_older"] = uef_contam_df[
#     "reg_xldate", "activity_xldate"
# ].apply(ax_or_red_older)
uef_contam_df["is_ax_or_red_older"] = uef_contam_df.apply(
    lambda row: ax_or_red_older(row["activity_xldate"], row["reg_xldate"]), axis=1
)
print(f"done with ax red in {time.perf_counter() - start}")
print(uef_contam_df)

# uef_contam_df["gw_or_sw"] = uef_contam_df[
#     "facility_type", "primary_source_water_type"
# ].apply(gw_sw_determiner)
uef_contam_df["gw_or_sw"] = uef_contam_df.apply(
    lambda row: gw_sw_determiner(
        row["facility_type"], row["primary_source_water_type"]
    ),
    axis=1,
)
print(f"done with gw sw in {time.perf_counter() - start}")
print(uef_contam_df)


conn = wdc_lib.sql_query_conn()
uef_contam_df.to_sql(
    "user_effective_timeline_w_contam_fac", conn, if_exists="replace", index=False
)
conn.close()

print(time.perf_counter() - start)
print(uef_contam_df)
