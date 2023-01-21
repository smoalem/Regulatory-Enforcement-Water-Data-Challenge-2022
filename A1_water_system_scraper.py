import bs4 as bs
import urllib.request
import numpy as np
import pandas as pd
import time
import re
import vapyr_date_library as vdl

if __name__ == '__main__':
    start = time.perf_counter()
    source = urllib.request.urlopen(
        'https://sdwis.waterboards.ca.gov/PDWW/JSP/SearchDispatch?number=&name=&county=&WaterSystemType=All&WaterSystemStatus=A&SourceWaterType=All&action=Search+For+Water+Systems')
    soup = bs.BeautifulSoup(source, 'html.parser')
    water_system_array = []
    table = soup.table
    table_rows = table.find_all('tr')
    hlinks = []
    hlink_extract = time.perf_counter()
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text.strip() for i in td]
        for link in tr.find_all('a', attrs={'href': re.compile("^WaterSystemDetail.jsp?")}):
            hlinks.append(link.get('href').strip()[22:])
        if(len(row) == 7):
            water_system_array.append(row)
    hlink_slice = time.perf_counter()
    slen = len(hlinks) // 2
    sliced = hlinks[:slen]
    water_np_arr = np.array(water_system_array).T
    appended = np.vstack([water_np_arr, sliced])
    columns = ['water_system_number', 'water_system_name', 'type', 'distribution_system_class',
               'max_treatment_plant_class', 'principal_county_served', 'primary_source_water_type', 'ws_link_suffix']
    df = pd.DataFrame(appended, columns).T
    df.insert(0, 'id', range(1, len(df) + 1))

    conn = vdl.sql_query_conn()
    df.to_sql('water_system_primary', conn,
              if_exists='replace', index=False)
    conn.close()
    finish = time.perf_counter()

    print(f'Extract: {hlink_extract - start}')
    print(f'Iterate & Slice: {hlink_slice - hlink_extract}')
    print(f'Write to SQL: {finish - hlink_extract}')

# Jae test 12/10/22
# Extract: 59.41056589997606
# Iterate & Slice: 2.3835855000070296
# Write to SQL: 2.4297240000450984
