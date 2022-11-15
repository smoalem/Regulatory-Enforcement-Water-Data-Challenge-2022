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

# build ANN by Friday

# two approaches:
# use ideal variables seen in wdc - regressions - random forest results
# use dimensionality reduction - feed in all variables

# multiple indpdt var - increase in processing time
# can see how long each epoch lasts and decide gpu acceleration 
# November 15th - to make initial prototype of ANN - decide on next steps in terms of boosting efficiency and accuracy

# Dec 1st - A1 - A10 in vapyr and the cesus steps to refresh the data and feed into neural network