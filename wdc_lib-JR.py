#import bs4 as bs
#import urllib.request

from datetime import date, timedelta
import datetime as dt
import calendar as cal
from lib2to3.pgen2.pgen import DFAState
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import cycle
import getpass
import os.path
import time
import winsound
import regex as re
import inspect
from pygit2 import Repository


def sql_query_conn(dbname='', basepath=''):
    # C:\Users\sarma\Dropbox\Water Data Challenge 2022\wdc_2022_test.db
    # C:\Users\sarma\Dropbox\Water Data Challenge 2022\wdc_2022_active.db
    # C:\Users\sarma\OneDrive\Documents\GitHub\Regulatory-Enforcement-Water-Data-Challenge-2022
    print(getpass.getuser())
    print('test')
    db_path = ''
    if len(basepath) == 0 and len(dbname) == 0:
        if getpass.getuser() == 'sarma':
            repo_head_name = Repository(
                'C:/Users/sarma/OneDrive/Documents/GitHub/Regulatory-Enforcement-Water-Data-Challenge-2022').head.name
            if repo_head_name == 'refs/heads/Development':
                db_path = r'C:\Users\sarma\Dropbox\Water Data Challenge 2022\wdc_2022_test.db'
            elif repo_head_name == 'refs/heads/Production':
                db_path = r'C:\Users\sarma\Dropbox\Water Data Challenge 2022\wdc_2022_active.db'
            else:
                print('New branch must be registered')
                raise OSError
        elif getpass.getuser() == 'hoonje92':
            repo_head_name = Repository(
                'C:/Users/jheaf/Desktop/Regulatory-Enforcement-Water-Data-Challenge-2022').head.name
            if repo_head_name == 'refs/heads/Development':
                db_path = r'C:\Users\jheaf\Dropbox\Water Data Challenge 2022\wdc_2022_test.db'
            elif repo_head_name == 'refs/heads/Production':
                db_path = r'C:\Users\jheaf\Dropbox\Water Data Challenge 2022\wdc_2022_active.db'
            else:
                print('New branch must be registered')
                raise OSError
        elif getpass.getuser() == 'Jessa Rego':  # update
            repo_head_name = Repository(
                'C:/Users/Jessa Rego/Documents/Regulatory-Enforcement-Water-Data-Challenge-2022').head.name  # update
            print(repo_head_name)
            if repo_head_name == 'refs/heads/Development':
                db_path = r'C:\Users\Jessa Rego\Dropbox\Water Data Challenge 2022\wdc_2022_test.db'  # update
            elif repo_head_name == 'refs/heads/main':
                db_path = r'C:\Users\Jessa Rego\Dropbox\Water Data Challenge 2022\wdc_2022_active.db'  # update
            else:
                print('New branch must be registered')
                raise OSError
        else:
            print("New user must be registered")
            raise OSError
    else:
        BASE_DIR = basepath
        db_path = os.path.join(BASE_DIR, dbname)
    conn = sqlite3.connect(db_path) #vs conn in line 72
    print(repo_head_name)
    return conn


conn = sql_query_conn() # vs conn in line 67
df_contam = pd.read_sql_query('SELECT * from contam_info', conn) #pd=pandas
df_facilities = pd.read_sql_query('SELECT * from facilities', conn)
conn.close()
print(df_contam)
print(df_facilities)
print(df_facilities.columns.values)
print(df_facilities['facility_type'].tolist())

def water_system_type(smplpttype):
    water_system = {"GW": ("WL", "SP", "SS", "CC", "CS",
                           "OT"), "SW": ("IN", "RS", "IG")}
    water_type = ''
    for k, it in water_system.items():
        if smplpttype in it:
            water_type = k #known?
        else:
            pass
    if water_type == '':
        water_type = 'pswt'
    return water_type
# print(water_system_type(('IN')))
# print(water_system_type(('IG')))
# print(water_system_type(('CC')))

for f in df_facilities['facility_type'].tolist():
    print(water_system_type(f))

def grab_water_results(contam_id="", fac_id=""):
    conn = sql_query_conn()
    query_base = f'''SELECT id, fac_id, contam_id, result_date, result_xldate, int_res FROM All_Results'''
    if fac_id == "":
        return 'Do not use this function for a total data pull'
    query_sec = ''' WHERE fac_id = ''' + fac_id + (''' AND contam_id = ''' + contam_id) * (
        len(contam_id) > 0) + " AND fac_id IS NOT NULL AND contam_id IS NOT NULL"
    query = query_base + query_sec
    df = pd.read_sql_query(query, conn) # first definition of df
    if len(df.index) == 0:
        df = pd.DataFrame({'id': [], 'fac_id': [], 'contam_id': [
        ], 'result_xldate': [], 'int_res': []}) # alternate definition of df, on the condition that len=0
    conn.close()
    df_final = df[(df['int_res'] != 'NO DATA') & (df['int_res'] != 'NO URL')]
    df_final['int_res'] = df_final['int_res'].astype(float)
    return df_final


# test_start = time.perf_counter()
# for fac in range(1, 10):
#     for s in range(1, 676):
#         grab_water_results(contam_id=str(s), fac_id='''"'''+str(fac)+'''"''')
#     print(time.perf_counter() - test_start)
# print(time.perf_counter() - test_start)
# raise ValueError
# print(grab_water_results(contam_id=str(101), fac_id='''"'''+str(130)+'''"'''))


def create_index(table_name, **index_names):
    # for each index in index names
    try:
        i = 0
        sub_str = ''
        # replace with lambda
        key_name = []
        for index in index_names:
            key_name.append(index)
            if i > 0 and i < len(index_names):
                sub_str = sub_str + ', '
            sub_str = sub_str + str(index) + ' ' + \
                index_names[index].upper() + ' '
            i += 1
        keys = '_'.join(key_name)
        ix_name = f'ix_{table_name.lower()}_{keys}'

        conn = sql_query_conn()
        sqliteCursor = conn.cursor()
        drop_query = f'''
            DROP INDEX IF EXISTS "{ix_name}";'''
        create_query = f'''
            CREATE INDEX IF NOT EXISTS "{ix_name}" ON "{table_name}" (
            {sub_str}); 
        '''
        sqliteCursor.execute(drop_query)
        sqliteCursor.execute(create_query)

        print(f"Index for {table_name} on key(s) {keys} created.")

    except ValueError:
        print("Error happened")
df_contam.to_excel(r'C:\Users\Jessa Rego\Documents\Regulatory-Enforcement-Water-Data-Challenge-2022\Vapyr-df_contam.xlsx')
