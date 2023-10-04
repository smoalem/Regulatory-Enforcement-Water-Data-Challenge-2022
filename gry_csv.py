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

conn = wdc.sql_query_conn()
contam_id = pd.read_sql_query(
    "SELECT id from contam_info", conn)
facilities_id = pd.read_sql_query(
    "SELECT id from facilities", conn)
contam_id_list = contam_id.iloc[99:, 0].to_list()
facilities_id_list = facilities_id.iloc[11:, 0].to_list()
contam_fac_id_comb = [(c, f) for c in contam_id_list for f in facilities_id_list]

by_cycle_period_effective = {"cycle": {"GREEN": 0, "YELLOW": 0, "RED": 0}, "effective": {"GREEN": 0, "YELLOW": 0, "RED": 0}, "cycle": {"GREEN": 0, "YELLOW": 0, "RED": 0}}
cycle_tl = pd.read_sql_query(
    "SELECT * from user_cycle_timeline", conn)
period_tl = pd.read_sql_query(
    "SELECT * from user_period_timeline", conn)
effective_tl = pd.read_sql_query(
    "SELECT * from user_effective_timeline", conn)
conn.close()
print("Finished Loading")

# for tl in [effective_tl, period_tl, cycle_tl]:
#     red_time = []
#     yellow_time = []
#     green_time = []
#     counter = 0
#     previous_id = "0-0"
#     for i, j in tl.iterrows():
#         current_id = str(j["fac_id"]) + "-" + str(j["contam_id"])
#         if current_id != previous_id:
#             previous_id = current_id
#             counter += 1
#             if j["color"] == "GREEN":
#                 green_time.append(j["end_date"] - j["start_date"])
#             elif j["color"] == "YELLOW":
#                 yellow_time.append(j["end_date"] - j["start_date"])
#             elif j["color"] == "RED":
#                 red_time.append(j["end_date"] - j["start_date"]) 
        
#     print(sum(red_time))
#     print(sum(yellow_time))
#     print(sum(green_time))
#     print(counter)


tl_counter = 0
for tl in [effective_tl, period_tl, cycle_tl]:
    tl_counter += 1
    green_reg_list ={ "CA Title 22 subsec 64445": [],
    "CA Title 22 subsec 64445.1(b)(1)": [],
    "CA Title 22 subsec 64445.1(b)(2)": [],
    "CA Title 22 subsec 64445.1(b)(3)": [],
    "CA Title 22 subsec 64445.1(c)(4)": [],
    "CA Title 22 subsec 64445.1(c)(5)(A)": [],
    "CA Title 22 subsec 64445.1(c)(5)(B)": [],
    "CA Title 22 subsec 64445.1(c)(5)(C)": [],
    "CA Title 22 subsec 64445.1(c)(7)": []}

    yellow_reg_list ={ "CA Title 22 subsec 64445": [],
    "CA Title 22 subsec 64445.1(b)(1)": [],
    "CA Title 22 subsec 64445.1(b)(2)": [],
    "CA Title 22 subsec 64445.1(b)(3)": [],
    "CA Title 22 subsec 64445.1(c)(4)": [],
    "CA Title 22 subsec 64445.1(c)(5)(A)": [],
    "CA Title 22 subsec 64445.1(c)(5)(B)": [],
    "CA Title 22 subsec 64445.1(c)(5)(C)": [],
    "CA Title 22 subsec 64445.1(c)(7)": []}

    red_reg_list ={ "CA Title 22 subsec 64445": [],
    "CA Title 22 subsec 64445.1(b)(1)": [],
    "CA Title 22 subsec 64445.1(b)(2)": [],
    "CA Title 22 subsec 64445.1(b)(3)": [],
    "CA Title 22 subsec 64445.1(c)(4)": [],
    "CA Title 22 subsec 64445.1(c)(5)(A)": [],
    "CA Title 22 subsec 64445.1(c)(5)(B)": [],
    "CA Title 22 subsec 64445.1(c)(5)(C)": [],
    "CA Title 22 subsec 64445.1(c)(7)": []}
    for i, j in tl.iterrows():
        if "CA" in j["regulation"]:
            if j["color"] == "GREEN":
                green_reg_list[j["regulation"]].append(j["end_date"] - j["start_date"])
            elif j["color"] == "YELLOW":
                yellow_reg_list[j["regulation"]].append(j["end_date"] - j["start_date"])
            elif j["color"] == "RED":
                red_reg_list[j["regulation"]].append(j["end_date"] - j["start_date"])
    print(tl_counter)
    print("GREEN")
    for k, v in green_reg_list.items():
        print(k, ": ", sum(green_reg_list[k]))
    print("__________")
    print("YELLOW")
    for k, v in yellow_reg_list.items():
        print(k, ": ", sum(yellow_reg_list[k]))    
    print("__________")
    print("RED")
    for k, v in red_reg_list.items():
        print(k, ": ", sum(red_reg_list[k]))    
    print("__________")