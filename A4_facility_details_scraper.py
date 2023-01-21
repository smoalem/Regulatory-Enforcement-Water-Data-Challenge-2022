from calendar import month
import concurrent.futures
import re
import time
import bs4 as bs
import pandas as pd
from bs4 import BeautifulSoup
import requests
import datetime as dt
import vapyr_date_library as vdl
from pypasser import reCaptchaV3


def list_of_urls():
    conn = vdl.sql_query_conn()
    df_facs = pd.read_sql_query("SELECT * from facilities", conn)
    conn.close()
    all_urls = df_facs['facility_hyperlink'].to_list()
    return all_urls


def fac_details_get(url):
    html = requests.get(url)

    html.encoding = 'utf-8'
    soup = bs.BeautifulSoup(html.text, 'html.parser')
    table = soup.table
    try:
        table_rows = table.find_all('tr')
    except:
        try:
            reCaptchaV3("https://www.google.com/recaptcha/api2/anchor?ar=1&k=6Ld38BkUAAAAAPATwit3FXvga1PI6iVTb6zgXw62&co=aHR0cHM6Ly9zZHdpcy53YXRlcmJvYXJkcy5jYS5nb3Y6NDQz&hl=en&v=pn3ro1xnhf4yB8qmnrhh9iD2&size=normal&cb=58rdj81p8zib")
            table_rows = table.find_all('tr')
        except:
            print(url)
            raise ValueError

    prev_row = ''
    sampling_point_type = 'Investigate'
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text.strip() for i in td]
        if len(row) == 4 and row[0] == 'Activity Reason\r\n      Text :':
            activity_reason_text = row[1]
            activity_status_date = row[3]
        else:
            pass
        if prev_row == ['Sampling\r\n        Point', 'Location', 'Type'] and row[2] != 'End Month' and row[2] != '':
            sampling_point_type = row[2]
            break
        if len(row) == 3 and row == ['Sampling\r\n        Point', 'Location', 'Type']:
            prev_row = ['Sampling\r\n        Point', 'Location', 'Type']

    try:
        day = int(activity_status_date[3:5])
        month = int(activity_status_date[:2])
        year = int(activity_status_date[-4:])
        activity_xldate = vdl.excel_date(dt.datetime(year, month, day))
    except:  # in cases where activity_status_date is blank
        activity_xldate = 99999

    fac_details = [activity_reason_text, activity_status_date,
                   activity_xldate, sampling_point_type, url]
    return fac_details


# fac_urls = list_of_urls()
# start = time.perf_counter()
# print(fac_urls)
# print(fac_urls[0])
# fac_stuff = fac_details_get(fac_urls[0])
# finish = time.perf_counter()
# print(fac_stuff)
# print(finish-start)
# raise ValueError


def results_hyperlinks(facility_and_wsp_info):
    test_columns = ['id_x', 'ws_id', 'activity_status_date', 'activity_xldate', 'state_asgn_id_number', 'facility_name', 'facility_type', 'sample_point_type', 'activity_status', 'activity_reason_text', 'classification',
                    'facility_hyperlink', 'id_y', 'water_system_number', 'water_system_name', 'ddw_name', 'ptype', 'pserved', 'ccount', 'service_area_code', 'type', 'primary_source_water_type', 'url', 'ws_link_suffix']
    print(facility_and_wsp_info)
    facility_and_wsp_info_df = pd.DataFrame(
        [facility_and_wsp_info], columns=test_columns)
    new_prefix = 'https://sdwis.waterboards.ca.gov/PDWW/JSP/WSamplingResultsByStoret.jsp?SystemNumber='
    new_suffix = '&Analyte=&ChemicalName=&begin_date=&end_date=&mDWW='
    old_prefix = 'https://sdwis.waterboards.ca.gov/PDWW/JSP/SamplingResultsByStoret.jsp?SystemNumber='
    old_suffix = '&Storet=&ChemicalName=&begin_date=&end_date='

    ws_num = facility_and_wsp_info_df['water_system_number'].iloc[0]
    ws_num_no_ca = ws_num[2:]
    ws_link_suffix = facility_and_wsp_info_df['ws_link_suffix'].iloc[0]
    tinwsys_number = re.search(
        r"tinwsys_is_number=(.*?)&tinwsys_st_code=", ws_link_suffix).group(1)
    ws_name = facility_and_wsp_info_df['water_system_name'].iloc[0]
    ws_name_with_pluses = ws_name.replace(' ', '+')
    fac_hyperlink = facility_and_wsp_info_df['facility_hyperlink'].iloc[0]
    wsf_is_number = fac_hyperlink[fac_hyperlink.index(
        '&tinwsf_is_number=') + 18:fac_hyperlink.index('&tinwsf_st_code')]
    fac_name = facility_and_wsp_info_df['facility_name'].iloc[0]
    fac_name_with_pluses = fac_name.replace(' ', '+')
    state_id_number = facility_and_wsp_info_df['state_asgn_id_number'].iloc[0]

    old_hyperlink = (old_prefix + ws_num_no_ca + '&SamplingPointID=' +
                     state_id_number + '&SamplingPointName=' + fac_name_with_pluses + old_suffix).replace('#', '')
    new_hyperlink = (new_prefix + ws_num_no_ca + '&tinwsys_is_number=' + tinwsys_number + '&FacilityID=' + state_id_number + '&WSFNumber=' + wsf_is_number +
                     '&SamplingPointID=' + state_id_number + '&SystemName=' + ws_name_with_pluses + '&SamplingPointName=' + fac_name_with_pluses + new_suffix).replace('#', '')
    facility_and_wsp_info_df = facility_and_wsp_info_df.assign(
        old_results_hyperlink=[old_hyperlink])
    facility_and_wsp_info_df = facility_and_wsp_info_df.assign(
        new_results_hyperlink=[new_hyperlink])
    facility_and_wsp_info_df = facility_and_wsp_info_df.loc[:, ['id_x', 'ws_id', 'activity_status_date', 'activity_xldate', 'state_asgn_id_number', 'facility_name', 'facility_type',
                                                                'sample_point_type', 'activity_status', 'activity_reason_text', 'classification', 'facility_hyperlink', 'old_results_hyperlink', 'new_results_hyperlink']]
    return facility_and_wsp_info_df.values.tolist()


if __name__ == '__main__':
    start = time.perf_counter()
    SFPU = list_of_urls()
    fac_details_columns = ['activity_reason_text', 'activity_status_date',
                           'activity_xldate', 'sample_point_type', 'facility_hyperlink']
    fac_details_list = []

    all_facilities_sublists = []
    afs_sublist_number = 0
    afs_sublist_size = 1000

    for afs in range(0, len(SFPU), afs_sublist_size):
        afs_sublist_number += 1
        new_afs_sublist = SFPU[afs: afs+afs_sublist_size]
        all_facilities_sublists.append([afs_sublist_number, new_afs_sublist])

    for sublist in all_facilities_sublists:
        sublist_start = time.perf_counter()
        for s in sublist[1]:
            s_start = time.perf_counter()
            fac_details_list.append(fac_details_get(s))
            print(
                f'Link {sublist[1].index(s) + 1} of {len(sublist[1])} took {time.perf_counter() - s_start} seconds')

        print(f'Sublist {sublist[0]} of {afs_sublist_number} took {time.perf_counter() - sublist_start} seconds, total time so far is {time.perf_counter() - start} seconds.')

    fac_details_df = pd.DataFrame(
        fac_details_list, columns=fac_details_columns)
    conn = vdl.sql_query_conn()
    fac_original_df = pd.read_sql_query(
        "SELECT * from facilities", conn)
    conn.close()
    all_facility_df = pd.merge(fac_original_df.reset_index(drop=True), fac_details_df.reset_index(
        drop=True), left_on='facility_hyperlink', right_on='facility_hyperlink', how='right')

    final_facility_df = all_facility_df[['id', 'ws_id', 'activity_status_date', 'activity_xldate', 'state_asgn_id_number', 'facility_name',
                                         'facility_type', 'sample_point_type', 'activity_status', 'activity_reason_text', 'classification', 'facility_hyperlink']]

    conn = vdl.sql_query_conn()
    final_facility_df.to_sql(
        'facilities', conn, if_exists='replace', index=False)
    conn.close()
    print(
        f'Time so far is {time.perf_counter() - start} seconds. Sublist {sublist[0]} of {afs_sublist_number} completed.')
    vdl.create_index('facilities', id='ASC')

    conn = vdl.sql_query_conn()
    all_facilities_dataframe = pd.read_sql_query(
        "SELECT * from facilities", conn)
    all_water_system_primary_dataframe = pd.read_sql_query(
        "SELECT * from water_system_primary", conn)
    conn.close()

    # Addition of results hyperlinks
    conn = vdl.sql_query_conn()
    all_facilities_dataframe = pd.read_sql_query(
        "SELECT * from facilities", conn)
    all_water_system_primary_dataframe = pd.read_sql_query(
        "SELECT * from water_system_primary", conn)
    conn.close()
    all_facs_and_wsp = pd.DataFrame.merge(
        all_facilities_dataframe, all_water_system_primary_dataframe, left_on='ws_id', right_on='id', how='left').values.tolist()

    final_facility_columns = ['id', 'ws_id', 'activity_status_date', 'activity_xldate', 'state_asgn_id_number', 'facility_name', 'facility_type',
                              'sample_point_type', 'activity_status', 'activity_reason_text', 'classification', 'facility_hyperlink', 'old_results_hyperlink', 'new_results_hyperlink']

    final_facility_list_of_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(results_hyperlinks, all_facs_and_wsp)
        for result in results:
            final_facility_list_of_list.extend(result)

        final_facility_df_with_results_hyperlinks = pd.DataFrame(
            final_facility_list_of_list, columns=final_facility_columns)

        conn = vdl.sql_query_conn()
        final_facility_df_with_results_hyperlinks.to_sql(
            'facilities', conn, if_exists='replace', index=False)
        conn.close()
    finish = time.perf_counter()
    print(finish - start)

# Jae 12/13/2022
# 16535.070647099987
