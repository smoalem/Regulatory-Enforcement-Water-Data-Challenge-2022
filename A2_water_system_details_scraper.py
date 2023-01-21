import concurrent.futures
import re
import time
import bs4 as bs
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

import requests
import vapyr_date_library as vdl


def list_of_urls():
    conn = vdl.sql_query_conn()
    df = pd.read_sql_query("SELECT * from water_system_primary", conn)
    conn.close()
    df["Hyperlink"] = 'https://sdwis.waterboards.ca.gov/PDWW/JSP/WaterSystemDetail.jsp?' + \
        df["ws_link_suffix"]
    all_urls = df['Hyperlink'].to_list()
    return all_urls


def res_get(url):
    html = requests.get(url)
    html.encoding = 'utf-8'
    soup = bs.BeautifulSoup(html.text, 'html.parser')
    # Details Box
    try:
        if len(soup.select("#AutoNumber6 > tr:nth-child(4) > td:nth-child(2)")[0].get_text()) > 0:
            WS_Status = soup.select(
                "#AutoNumber6 > tr:nth-child(4) > td:nth-child(2)")[0].get_text()
        else:
            WS_Status = 'No WS_Status data'
    except:
        WS_Status = 'Investigate'
    try:
        if len(soup.select("#AutoNumber6 > tr:nth-child(4) > td:nth-child(4)")[0].get_text()) > 0:
            WS_Act_Date = soup.select(
                "#AutoNumber6 > tr:nth-child(4) > td:nth-child(4)")[0].get_text()
        else:
            WS_Act_Date = 'No WS_Act_Date data'
    except:
        WS_Act_Date = 'Investigate'

    # WS Contacts Box
    try:
        if len(soup.find(title="Google Map").text.strip()) > 0:
            WS_Address = soup.find(title="Google Map").text.strip()
        else:
            WS_Address = 'No smonth data'
    except:
        WS_Address = 'Investigate'
    try:
        if len(soup.select(".AC > td:nth-child(2)")[0].get_text()) > 0:
            WS_Phone = soup.select(".AC > td:nth-child(2)")[0].get_text()
        else:
            WS_Phone = 'No WS_Phone data'
    except:
        WS_Phone = 'Investigate'

    # DDW Info Box
    try:

        ddw_table_row = str(list(soup.findAll('table')[6])[3])
        ddw_list = ddw_table_row.split('\n')
        ddw_name = ddw_list[1][94:-16]
        # regex looking for ###-###-#### pattern
        ddw_phone = re.compile(
            r'\d\d\d-\d\d\d-\d\d\d\d').search(ddw_table_row).group()
        ddw_email = ddw_list[11][46:-5]
        ddw_address_raw = (ddw_list[14:-2])
        ddw_address = ''
        for a in ddw_address_raw:
            address_element = a.lstrip().strip('\r')
            ddw_address += address_element
    except:
        ddw_name = 'Investigate'
        ddw_phone = 'Investigate'
        ddw_email = 'Investigate'
        ddw_address = 'Investigate'

    # Operating Periods/Population Box
    try:
        if len(soup.find(headers="smonth").text.strip()) > 0:
            smonth = soup.find(headers="smonth").text.strip()
        else:
            smonth = 'No smonth data'
    except:
        smonth = 'Investigate'
    try:
        if len(soup.find(headers="sday").text.strip()) > 0:
            sday = soup.find(headers="sday").text.strip()
        else:
            sday = 'No sday data'
    except:
        sday = 'Investigate'
    try:
        if len(soup.find(headers="emonth").text.strip()) > 0:
            emonth = soup.find(headers="emonth").text.strip()
        else:
            emonth = 'No emonth data'
    except:
        emonth = 'Investigate'
    try:
        if len(soup.find(headers="eday").text.strip()) > 0:
            eday = soup.find(headers="eday").text.strip()
        else:
            eday = 'No eday data'
    except:
        eday = 'Investigate'
    try:
        if len(soup.find(headers="ptype").text.strip()) > 0:
            ptype = soup.find(headers="ptype").text.strip()
        else:
            ptype = 'No ptype data'
    except:
        ptype = 'Investigate'
    try:
        if len(soup.find(headers="pserved").text.strip()) > 0:
            pserved = soup.find(headers="pserved").text.strip()
        else:
            pserved = 'No pserved data'
    except:
        pserved = 'Investigate'

    # Service Connections Box
    try:
        if len(soup.find(headers="type").text.strip()) > 0:
            ctype = soup.find(headers="type").text.strip()
        else:
            ctype = 'No ctype data'
    except:
        ctype = 'Investigate'
    try:
        if len(soup.find(headers="count").text.strip()) > 0:
            ccount = soup.find(headers="count").text.strip()
        else:
            ccount = 'No ccount data'
    except:
        ccount = 'Investigate'
    try:
        if len(soup.find(headers="mtype").text.strip()) > 0:
            mtype = soup.find(headers="mtype").text.strip()
        else:
            mtype = 'No mtype data'
    except:
        mtype = 'Investigate'
    try:
        if len(soup.find(headers="msize").text.strip()) > 0:
            msize = soup.find(headers="msize").text.strip()
        else:
            msize = 'No msize data'
    except:
        msize = 'Investigate'
    try:
        if len(soup.find(headers="code").text.strip()) > 0:
            Service_Area_Code = soup.find(headers="code").text.strip()
        else:
            Service_Area_Code = 'No Service_Area_Code data'
    except:
        Service_Area_Code = 'Investigate'
    try:
        if len(soup.find(headers="sname").text.strip()) > 0:
            Service_Area_Name = soup.find(headers="sname").text.strip()
        else:
            Service_Area_Name = 'No Service_Area_Name data'
    except:
        Service_Area_Name = 'Investigate'

    # Water Purchases Box
    try:
        if len(soup.find(headers="name", width="10%").text.strip()) > 0:
            PW_Sys_Num = soup.find(headers="name", width="10%").text.strip()
        else:
            PW_Sys_Num = 'No PW_Sys_Num data'
    except:
        PW_Sys_Num = 'Investigate'

    raw_output = [WS_Status, WS_Act_Date, WS_Address, WS_Phone, ddw_name, ddw_phone, ddw_email, ddw_address, smonth,
                  sday, emonth, eday, ptype, pserved, ctype, ccount, mtype, msize, Service_Area_Code, Service_Area_Name, PW_Sys_Num]
    finished_result = [item.translate(str.maketrans(
        '', '', '''\n\t\r"\xa0"''')).strip() for item in raw_output]
    return finished_result


# print(res_get('https://sdwis.waterboards.ca.gov/PDWW/JSP/WaterSystemDetail.jsp?tinwsys_is_number=1735&tinwsys_st_code=CA&wsnumber=CA1510003'))
# print(res_get(list_of_urls()[0]))
# raise ValueError
# # print(list_of_urls()[0])
# print('https://sdwis.waterboards.ca.gov/PDWW/JSP/WaterSystemDetail.jsp?tinwsys_is_number=1735&tinwsys_st_code=CA&wsnumber=CA1510003')
# print(list_of_urls().index(
#     'https://sdwis.waterboards.ca.gov/PDWW/JSP/WaterSystemDetail.jsp?tinwsys_is_number=1735&tinwsys_st_code=CA&wsnumber=CA1510003'))


if __name__ == '__main__':
    start = time.perf_counter()
    Water_Sys_Details_urls = list_of_urls()
    WSDU = Water_Sys_Details_urls
    water_system_details = []
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(res_get, WSDU)
        end_create = time.perf_counter()
        for result in results:
            water_system_details.append(result)
        end_iterate = time.perf_counter()
    for i in range(len(water_system_details)):
        water_system_details[i].append(WSDU[i])
    process_for_df = np.array(water_system_details).T
    column_names = ['ws_status', 'ws_act_date', 'ws_address', 'ws_phone', 'ddw_name', 'ddw_phone', 'ddw_email', 'ddw_address', 'start_month', 'start_day',
                    'end_month', 'end_day', 'ptype', 'pserved', 'ctype', 'ccount', 'mtype', 'msize', 'service_area_code', 'service_area_name', 'pw_sys_num', 'url']
    water_sys_detail_df = pd.DataFrame(process_for_df, column_names).T

    conn = vdl.sql_query_conn()
    water_sys_original_df = pd.read_sql_query(
        "SELECT * from water_system_primary", conn)
    conn.close()
    fin_water_sys_df = pd.concat([water_sys_original_df.reset_index(
        drop=True), water_sys_detail_df.reset_index(drop=True)], axis=1)
    water_system_primary = fin_water_sys_df[['water_system_number', 'water_system_name', 'ddw_name', 'ptype',
                                             'pserved', 'ccount', 'service_area_code', 'distribution_system_class', 'max_treatment_plant_class', 'type', 'primary_source_water_type', 'url', 'ws_link_suffix']]
    water_system_secondary = fin_water_sys_df[['ws_act_date', 'ws_address', 'ws_phone', 'ws_status', 'principal_county_served', 'service_area_code', 'service_area_name',
                                               'pw_sys_num', 'start_month', 'start_day', 'end_month', 'end_day', 'ctype', 'mtype', 'msize', 'ddw_phone', 'ddw_email', 'ddw_address']]
    water_system_primary.insert(
        0, 'id', range(1, len(water_system_primary) + 1))
    water_system_secondary['ws_id'] = water_system_primary['id']

    conn = vdl.sql_query_conn()
    water_system_primary.to_sql(
        'water_system_primary', conn, if_exists='replace', index=False)
    water_system_secondary.to_sql(
        'water_system_secondary', conn, if_exists='replace', index=False)
    conn.close()

    print(fin_water_sys_df)
    print(water_system_primary)
    print(water_system_secondary)
    finish = time.perf_counter()
    print(f'Results Creation: {end_create - start}')
    print(f'Results Iteration: {end_iterate - end_create}')
    print(f'Seconds: {finish - start}')

# Jae Test 12/10/22
# Results Creation: 2.236424000002444
# Results Iteration: 238.0695601000334
# Seconds: 241.77644300000975
