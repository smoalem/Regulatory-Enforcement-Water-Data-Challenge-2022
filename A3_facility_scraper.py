import concurrent.futures
import re
import time
import bs4 as bs
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import vapyr_date_library as vdl
from datetime import datetime
from pypasser import reCaptchaV3


def list_of_urls_wsids():
    conn = vdl.sql_query_conn()
    wsp_df = pd.read_sql_query("SELECT * from water_system_primary", conn)
    conn.close()
    wsp_df["hyperlink"] = 'https://sdwis.waterboards.ca.gov/PDWW/JSP/WaterSystemFacilities.jsp?' + \
        wsp_df["ws_link_suffix"]
    wsp_df["hyperlink"] = wsp_df["hyperlink"].str.replace(
        r'\&wsnumber=CA.......', '')
    all_urls = wsp_df['hyperlink'].to_list()
    all_wsids = wsp_df['id'].to_list()
    wsid_url_list_of_list = []
    for url in all_urls:
        index = all_urls.index(url)
        ws_id = all_wsids[index]
        wsid_url_list_of_list.append([url, ws_id])
    return wsid_url_list_of_list


def facilities_url_scraper(url_wsid):
    url = url_wsid[0]
    wsid = url_wsid[1]
    source = requests.get(url)
    source.encoding = 'utf-8'
    soup = bs.BeautifulSoup(source.text, 'html.parser')
    table = soup.table
    # try:
    try:
        table_rows = table.find_all('tr')

    except:
        try:
            reCaptchaV3("https://www.google.com/recaptcha/api2/anchor?ar=1&k=6Ld38BkUAAAAAPATwit3FXvga1PI6iVTb6zgXw62&co=aHR0cHM6Ly9zZHdpcy53YXRlcmJvYXJkcy5jYS5nb3Y6NDQz&hl=en&v=pn3ro1xnhf4yB8qmnrhh9iD2&size=normal&cb=58rdj81p8zib")
            table_rows = table.find_all('tr')
        except:
            print(
                f'table is: {table} and url_wsid is: {url_wsid} and soup is {soup}')
            print(source)
            raise ValueError

    hlinks = []
    facilities_list_of_list = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text.strip() for i in td]
        for link in tr.find_all('a', attrs={'href': re.compile("^WaterSystemFacility.jsp?")}):
            hlinks.append(
                'https://sdwis.waterboards.ca.gov/PDWW/JSP/WaterSystemFacility.jsp?' + link.get('href').strip()[24:])
        if(len(row) == 5):
            facilities_list_of_list.append(row)
    slen = len(hlinks) // 2
    sliced = np.array(hlinks[slen:])
    sliced = hlinks[slen:]
    facilities_list_of_list = facilities_list_of_list[1:]
    final_fac_list_of_list = []
    for f in facilities_list_of_list:
        list_index = facilities_list_of_list.index(f)
        f.append(sliced[list_index])
        f.insert(0, wsid)
        final_fac_list_of_list.append(f)
    return final_fac_list_of_list


# test_columns = ['ws_id', 'state_asgn_id_number', 'facility_name',
#                 'facility_type', 'classification', 'activity_status', 'facility_hyperlink']
# urls_list = list_of_urls_wsids()
# print(urls_list)
# # raise ValueError
# test = urls_list[693]
# print(test)
# test_data = facilities_url_scraper(test)
# print(test_data)
# test_df = pd.DataFrame(test_data, columns=test_columns)
# print(test_df)
# raise ValueError


if __name__ == '__main__':
    start = time.perf_counter()
    df_columns = ['ws_id', 'state_asgn_id_number', 'facility_name',
                  'facility_type', 'classification', 'activity_status', 'facility_hyperlink']
    FPUs = list_of_urls_wsids()
    fac_df_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(facilities_url_scraper, FPUs)
        results_creation = time.perf_counter()
        for result in results:
            fac_df_list.extend(result)
        results_iteration = time.perf_counter()
        facilities_df = pd.DataFrame(fac_df_list, columns=df_columns)
        facilities_df.insert(0, 'id', range(1, len(facilities_df) + 1))

        conn = vdl.sql_query_conn()
        facilities_df.to_sql('facilities', conn,
                             if_exists='replace', index=False)
        conn.close()

    finish = time.perf_counter()
    print(f'Results Creation: {results_creation - start}')
    print(f'Results Iteration: {results_iteration - results_creation}')
    print(f'Seconds: {finish - start}')


# Jae test 12/11/2022
# Results Creation: 2.3574605000321753
# Results Iteration: 173.6447102999664
# Seconds: 177.33737920003477
