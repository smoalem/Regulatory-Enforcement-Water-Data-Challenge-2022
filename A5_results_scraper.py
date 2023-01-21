import concurrent.futures
import time
from unittest import result
import bs4 as bs
import pandas as pd
import requests
import vapyr_date_library as vdl
import datetime as dt
import cProfile
import pstats
import io
from pstats import SortKey
from pypasser import reCaptchaV3


def pgl_unit_modifier(raw_unit):
    # takes unit of the raw data and creates multiplier for converting to pg/l
    raw_unit = raw_unit.lower()
    pgl_unit = 'pg/l'
    unit_dict = {'mg/l': 1000, 'ug/l': 1000000, 'ng/l': 1000000000, 'pg/l': 1000000000000,
                 'c': 'const', 'mfl': 'const', 'ntu': 'const', 'pci/l': 'const', 'ton': 'const',
                 'units': 'const', 'umho/cm': 'const', '0': 'const', '': 'const', 'lang': 'const',
                 'ph': 'const', 'aggr': 'const'}
    if raw_unit == pgl_unit or unit_dict[raw_unit] == 'const':
        multiplier = 1
    else:
        multiplier = unit_dict[pgl_unit]/unit_dict[raw_unit]
    return multiplier


def contam_unit_modifier(contam_unit):
    # takes unit that was in pg/l form and creates multiplier for converting to unit used by contam
    contam_unit = contam_unit.lower()
    unit_dict = {'mg/l': 1000, 'ug/l': 1000000, 'ng/l': 1000000000, 'pg/l': 1000000000000,
                 'c': 'const', 'mfl': 'const', 'ntu': 'const', 'pci/l': 'const', 'ton': 'const',
                 'units': 'const', 'umho/cm': 'const', '0': 'const', '': 'const', 'lang': 'const',
                 'ph': 'const', 'aggr': 'const'}
    if unit_dict[contam_unit] == 'const':
        multiplier = 1
    else:
        multiplier = unit_dict[contam_unit]/unit_dict['pg/l']
    return multiplier


def contam_id_match_key(contaminant_row_list):
    # Creates tuple with each contam id being matched to some combination of storet_no, analyte_no, mcl, dlr, trig, etc
    # This ultimately creates two outputs that go into facility_results function:
    # old_match for matching to storet_no combinations on old link
    # new_match for matching to analyte_no combinations on new link
    unit = contaminant_row_list[9]
    multiplier = pgl_unit_modifier(unit)
    contam_id = contaminant_row_list[0]
    storet_no = contaminant_row_list[1]
    analyte_no = contaminant_row_list[2]
    old_res_match = contaminant_row_list[3].lower()
    new_res_match = contaminant_row_list[4].lower()
    mcl = contaminant_row_list[6] * multiplier
    dlr = contaminant_row_list[7] * multiplier
    trig = contaminant_row_list[8] * multiplier
    old_match = ''
    new_match = ''
    insert_dictionary = {'storet_no': storet_no, 'analyte_no': analyte_no, 'mcl': mcl, 'dlr': dlr, 'trig': trig,
                         'dlofanalyte_no': analyte_no, 'rlofanalyte_no': analyte_no, 'ceofanalyte_no': analyte_no, 'dlrisblank': 0}
    # Example: contam_id 9 has old_res_link of storet_no, mcl, dlr, trig with storet_no == 00440, mcl == 0.0, dlr == 0.0, and trig == 0.0
    # This function would generate a string of: 'storet_no==00440|mcl==0.0|dlr==0.0|trig==0.0'
    if 'removed' not in old_res_match and 'delete' not in old_res_match:
        old_match_list = []

        for old in old_res_match.lower().split(','):
            old = old.replace(' ', '')
            if '(' in old:
                pass
            else:
                old_match_list.append(
                    str(old) + '==' + str(insert_dictionary[old]))
        old_match = ('|'.join(old_match_list), contam_id)
    else:
        pass
    if 'removed' not in new_res_match and 'delete' not in new_res_match:
        new_match_list = []
        for new in new_res_match.lower().split(','):
            new = new.replace(' ', '')
            if '(' in new:
                pass
            else:
                new_match_list.append(
                    str(new) + '==' + str(insert_dictionary[new]))
        new_match = ('|'.join(new_match_list), contam_id)
    else:
        pass
    return [old_match, new_match]


def raw_value_standardizer(value):
    # Replaces blank value with 0.0 float
    if value == '':
        value = 0.0
    return float(value)


def old_res_match(old_result_row, contam_match_dict, contam_units_dict):
    # Individual result rows are fed into this
    # result rows matched to a contam_id
    # will ultimately have results returned in whatever unit is used by that id on contam_info table
    storet_no = old_result_row[0]
    unit = old_result_row[8]
    pgl_multiplier = pgl_unit_modifier(unit)
    mcl = raw_value_standardizer(old_result_row[5]) * pgl_multiplier
    dlr = raw_value_standardizer(old_result_row[6]) * pgl_multiplier
    trig = raw_value_standardizer(old_result_row[7]) * pgl_multiplier
    xmod_less_than = old_result_row[3]
    raw_result = raw_value_standardizer(old_result_row[4]) * pgl_multiplier
    int_res = raw_result
    if raw_result < dlr or xmod_less_than == '<':
        int_res = 0.0

    all_test = []
    all_test.append(
        f'storet_no=={str(storet_no)}|mcl=={str(mcl)}|dlr=={str(dlr)}|trig=={str(trig)}')
    all_test.append(
        f'storet_no=={str(storet_no)}|mcl=={str(mcl)}|dlr=={str(dlr)}')
    all_test.append(
        f'storet_no=={str(storet_no)}|dlr=={str(dlr)}|trig=={str(trig)}')
    all_test.append(
        f'storet_no=={str(storet_no)}|mcl=={str(mcl)}')
    all_test.append(
        f'storet_no=={str(storet_no)}')
    contam_id = ''
    for test in all_test:  # iterate through every combination of all_test to find one that matches a key in the contam_match dictionary
        if test in contam_match_dict.keys():
            contam_id = contam_match_dict[test]
            break
    result_year = int(old_result_row[2][:4])
    result_month = int(old_result_row[2][5:7])
    result_day = int(old_result_row[2][8:])
    result_xldate = vdl.excel_date(dt.datetime(
        result_year, result_month, result_day))
    result_datetime = vdl.xl_to_date_print(result_xldate)
    try:
        contam_unit = contam_units_dict[contam_id]
    except:
        print('old_res_match exception')
        print(contam_id)
        print(old_result_row)
        raise ValueError
    contam_unit_multiplier = contam_unit_modifier(contam_unit)
    fin_raw_result = raw_result * contam_unit_multiplier
    fin_int_res = int_res * contam_unit_multiplier
    finished_old_result_row = [
        contam_id, result_datetime, result_xldate, xmod_less_than, fin_raw_result, fin_int_res]
    return finished_old_result_row


def old_res_get(facility_url_contam):
    # Scrapes old link and returns formatted results
    contam_dict = facility_url_contam[2]
    url = facility_url_contam[1]
    facility_id = facility_url_contam[0]
    contam_units_dict = facility_url_contam[3]
    results_array = []

    try:
        html = requests.get(url)
        html.encoding = 'utf-8'
        soup = bs.BeautifulSoup(html.text, 'html.parser')
        table = soup.table
        table_rows = table.find_all('tr')

        for tr in table_rows:
            td = tr.find_all('td')
            row = [i.text.strip() for i in td]

            if len(row) == 9:
                results_array.append(row)
            else:
                pass
        # If webpage exists but facility has no data, then it creates a row in the results table for it where it indicates that it had no data
        if len(results_array) == 0:
            results_array.append(['NO DATA']*9)
        else:
            pass
    except:
        try:
            reCaptchaV3("https://www.google.com/recaptcha/api2/anchor?ar=1&k=6Ld38BkUAAAAAPATwit3FXvga1PI6iVTb6zgXw62&co=aHR0cHM6Ly9zZHdpcy53YXRlcmJvYXJkcy5jYS5nb3Y6NDQz&hl=en&v=pn3ro1xnhf4yB8qmnrhh9iD2&size=normal&cb=58rdj81p8zib")
        except:
            results_array.append(['NO URL']*9)
    final_output = []
    for i in results_array:  # at this point we are adding in the contam_id, reorganizing the columns in the desired order, and transforming units if necessary
        xmod = i[3]
        if xmod == 'NO DATA' or xmod == 'NO URL':
            raw_result = xmod
            int_res = xmod
            result_date = xmod
            result_xldate = xmod
            contam_id = xmod
            lab_method_info = xmod
            final_output = [[facility_id, contam_id, result_date,
                            result_xldate, xmod, raw_result, int_res, lab_method_info]]
        else:
            try:
                row_with_contam_id = old_res_match(
                    i, contam_dict, contam_units_dict)
            except:
                print("########")
                print('old_res_get error')
                print(facility_url_contam)
                print(i)
                print("#$#$#$#$#$$")
                raise ValueError
            row_with_contam_id.insert(0, facility_id)
            row_with_contam_id.append(
                'Lab SID: Null, Lab: Null, ELAP: Null, Method: Null')
            final_output.append(row_with_contam_id)
    return final_output


def new_res_match(new_result_row, contam_match_dict, contam_units_dict, rad_type):
    # Individual result rows are fed into this
    # result rows matched to a contam_id
    # will ultimately have results returned in whatever unit is used by that id on contam_info table
    analyte_no = new_result_row[0]
    unit = new_result_row[9]
    pgl_multiplier = pgl_unit_modifier(unit)
    reporting_level = raw_value_standardizer(
        new_result_row[5]) * pgl_multiplier
    counting_error = raw_value_standardizer(new_result_row[6]) * pgl_multiplier
    mcl = raw_value_standardizer(new_result_row[7]) * pgl_multiplier
    dlr = raw_value_standardizer(new_result_row[8]) * pgl_multiplier
    xmod_less_than = new_result_row[4]
    raw_result = raw_value_standardizer(new_result_row[3]) * pgl_multiplier
    int_res = raw_result
    if raw_result < dlr or xmod_less_than == '<':
        int_res = 0.0

    all_test = []
    analyte_rad_append = ''
    if rad_type != 'non':
        analyte_rad_append = rad_type + 'of'
    all_test.append(
        f'{analyte_rad_append}analyte_no=={str(analyte_no)}|mcl=={str(mcl)}|dlr=={str(dlr)}')
    all_test.append(
        f'{analyte_rad_append}analyte_no=={str(analyte_no)}|mcl=={str(mcl)}|dlrisblank=={str(0)}')
    all_test.append(
        f'{analyte_rad_append}analyte_no=={str(analyte_no)}|dlr=={str(dlr)}')
    all_test.append(
        f'{analyte_rad_append}analyte_no=={str(analyte_no)}|mcl=={str(mcl)}')
    all_test.append(
        f'{analyte_rad_append}analyte_no=={str(analyte_no)}')
    contam_id = ''
    passed_test = ''
    for test in all_test:
        if test in contam_match_dict.keys():
            passed_test = test
            contam_id = contam_match_dict[test]
            break
    result_year = int(new_result_row[2][6:])
    result_month = int(new_result_row[2][:2])
    result_day = int(new_result_row[2][3:5])
    result_xldate = vdl.excel_date(dt.datetime(
        result_year, result_month, result_day))
    result_datetime = vdl.xl_to_date_print(result_xldate)
    try:
        contam_unit = contam_units_dict[contam_id]
    except:
        print('new exception')
        print(rad_type)
        print(rad_type != 'non')
        print(contam_id)
        print(passed_test)
        print(new_result_row)
        print(all_test)
        raise ValueError
    if rad_type != 'non':
        if rad_type == 'dl':
            raw_result = raw_result
        elif rad_type == 'rl':
            raw_result = reporting_level
        elif rad_type == 'ce':
            raw_result = counting_error
        else:
            print('Improper rad_type entered')
            raise ValueError
    contam_unit_multiplier = contam_unit_modifier(contam_unit)
    fin_raw_result = raw_result * contam_unit_multiplier
    fin_int_res = int_res * contam_unit_multiplier
    lab_method_info = f'Lab SID: {new_result_row[10]}, Lab: {new_result_row[11]}, ELAP: {new_result_row[12]}, Method: {new_result_row[13]}'
    finished_new_result_row = [
        contam_id, result_datetime, result_xldate, xmod_less_than, fin_raw_result, fin_int_res, lab_method_info]
    return finished_new_result_row


def new_res_get(facility_url_contam):
    # Scrapes new link and returns formatted results
    facility_id = facility_url_contam[0]
    url = facility_url_contam[1]
    contam_dict = facility_url_contam[2]
    contam_units_dict = facility_url_contam[3]
    results_array = []
    try:
        html = requests.get(url)
        html.encoding = 'utf-8'
        soup = bs.BeautifulSoup(html.text, 'html.parser')
        table = soup.table
        table_rows = table.find_all('tr')
        counter = 0
        for tr in table_rows:
            td = tr.find_all('td')
            row = [i.text.strip() for i in td]

            if len(row) == 14:
                results_array.append(row)
            else:
                pass
        # If webpage exists but facility has no data, then it creates a row in the results table for it where it indicates that it had no data
        if len(results_array) == 0:
            results_array.append(['NO DATA']*14)
        else:
            pass
    except:
        try:
            reCaptchaV3("https://www.google.com/recaptcha/api2/anchor?ar=1&k=6Ld38BkUAAAAAPATwit3FXvga1PI6iVTb6zgXw62&co=aHR0cHM6Ly9zZHdpcy53YXRlcmJvYXJkcy5jYS5nb3Y6NDQz&hl=en&v=pn3ro1xnhf4yB8qmnrhh9iD2&size=normal&cb=58rdj81p8zib")
        except:
            results_array.append(['NO URL']*14)
    final_output = []
    for i in results_array:  # at this point we are adding in the contam_id, reorganizing the columns in the desired order, and transforming units if necessary
        xmod = i[4]
        if xmod == 'NO DATA' or xmod == 'NO URL':
            raw_result = xmod
            int_res = xmod
            result_date = xmod
            result_xldate = xmod
            contam_id = xmod
            lab_method_info = xmod
            final_output = [[facility_id, contam_id, result_date,
                             result_xldate, xmod, raw_result, int_res, lab_method_info]]
        else:
            try:
                if i[9].lower() != 'pci/l':
                    row_with_contam_id = new_res_match(
                        i, contam_dict, contam_units_dict, 'non')
                    row_with_contam_id.insert(0, facility_id)
                    final_output.append(row_with_contam_id)
                else:
                    # dl - detection level
                    # rl - reporting level
                    # ce - counting error
                    for rad in ['dl', 'rl', 'ce']:
                        try:
                            row_with_contam_id = new_res_match(
                                i, contam_dict, contam_units_dict, rad)
                            row_with_contam_id.insert(0, facility_id)
                            final_output.append(row_with_contam_id)
                        except:
                            print(f'no match with rad_type: {rad}')
            except:
                print("########")
                print('new_res_get error')
                print(facility_url_contam)
                print(i)
                print("#$#$#$#$#$$")
                raise ValueError
    return final_output


def facility_results(facility_id_old_link_new_link, old_match, new_match, contam_units_dict):
    # This function takes results from old and new links and then marries them together
    # If an identical result exists on both old and new, default is to go with new as it is the only one that would include values for lab_method_info.
    fac_id = facility_id_old_link_new_link[0]
    ws_id = facility_id_old_link_new_link[1]
    old_link = facility_id_old_link_new_link[2]
    new_link = facility_id_old_link_new_link[3]
    old_res_list_of_list = old_res_get(
        [fac_id, old_link, old_match, contam_units_dict])
    new_res_list_of_list = new_res_get(
        [fac_id, new_link, new_match, contam_units_dict])
    old_res_df = pd.DataFrame(old_res_list_of_list, columns=[
                              'fac_id', 'contam_id', 'result_date', 'result_xldate', 'xmod_less_than', 'raw_result', 'int_res', 'lab_method_info'])
    new_res_df = pd.DataFrame(new_res_list_of_list, columns=[
                              'fac_id', 'contam_id', 'result_date', 'result_xldate', 'xmod_less_than', 'raw_result', 'int_res', 'lab_method_info'])

    old_res_df['contam_id_xldate_int_res'] = old_res_df['contam_id'].astype(str) + \
        '-' + \
        old_res_df['result_xldate'].astype(
            str) + '-' + old_res_df['int_res'].astype(str)
    new_res_df['contam_id_xldate_int_res'] = new_res_df['contam_id'].astype(str) + \
        '-' + \
        new_res_df['result_xldate'].astype(
            str) + '-' + new_res_df['int_res'].astype(str)

    old_res_df['contam_id_xldate_int_res_and_count'] = old_res_df['contam_id_xldate_int_res'] + '-' + old_res_df.groupby('contam_id_xldate_int_res')[
        'contam_id_xldate_int_res'].transform('count').astype(str)
    new_res_df['contam_id_xldate_int_res_and_count'] = new_res_df['contam_id_xldate_int_res'] + '-' + new_res_df.groupby('contam_id_xldate_int_res')[
        'contam_id_xldate_int_res'].transform('count').astype(str)

    old_lab_info = old_res_df['lab_method_info'].loc[0]
    new_lab_info = new_res_df['lab_method_info'].loc[0]

    if old_lab_info == 'NO DATA' or old_lab_info == 'NO URL':
        old_lab_info = 'NO'
    if new_lab_info == 'NO DATA' or new_lab_info == 'NO URL':
        new_lab_info = 'NO'

    # Only do merge of both if neither has a 'NO DATA' or 'NO URL' result
    if 'NO' != old_lab_info and 'NO' != new_lab_info:
        combined_res_df = pd.concat([new_res_df, old_res_df]).sort_values(by=["lab_method_info", "result_xldate", "int_res"], ascending=[
            True, True, False]).drop_duplicates(subset=["contam_id_xldate_int_res_and_count"], keep="first").reset_index(drop=True)
    # If both have no results, then just go with new
    elif 'NO' == old_lab_info and 'NO' == new_lab_info:
        combined_res_df = new_res_df
    # if new has actual results and old doesn't, go with new
    elif 'NO' != new_lab_info and 'NO' == old_lab_info:
        combined_res_df = new_res_df
    # if old has actual results and new doesn't, go with old
    elif 'NO' != old_lab_info and 'NO' == new_lab_info:
        combined_res_df = old_res_df

    combined_res_df.drop(
        ['contam_id_xldate_int_res', 'contam_id_xldate_int_res_and_count'], axis=1, inplace=True)
    combined_res_df['ws_id'] = ws_id
    combined_res_df = combined_res_df[['fac_id', 'ws_id', 'contam_id', 'result_date',
                                      'result_xldate', 'xmod_less_than', 'raw_result', 'int_res', 'lab_method_info']]

    return combined_res_df.values.tolist()


# # # ##### test #####
# conn = vdl.sql_query_conn()
# contam_df = pd.read_sql_query(
#     "SELECT * from contam_info", conn)
# facilities_list = pd.read_sql_query(
#     "SELECT * from facilities", conn).values.tolist()
# conn.close()
# test_fac = [facilities_list[38077][0], facilities_list[38077][1],
#             facilities_list[38077][12], facilities_list[38077][13]]  # fac_id, old link, and new link

# contam_match_old_res_dictionary = {}
# # matches results from new link with correct contaminant id
# contam_match_new_res_dictionary = {}
# # Simply for making results for particular id be in common unit
# contam_id_unit_dict = {}
# for i, j in contam_df.iterrows():
#     contam_id_unit_dict[j['id']] = j['unit']
#     # creates string that is combination of storet_no, analyte_no, mcl, dlr, trig, etc that connects to particular contam id (old and new links)
#     contams_to_add = contam_id_match_key(j.values.tolist())
#     if contams_to_add[0] != '':  # if old contam match is not ''
#         contam_match_old_res_dictionary[contams_to_add[0]
#                                         [0]] = contams_to_add[0][1]
#     if contams_to_add[1] != '':  # if new contam match is not ''
#         contam_match_new_res_dictionary[contams_to_add[1]
#                                         [0]] = contams_to_add[1][1]


def facility_results_counter(fac):
    df_results = vdl.grab_water_results(fac_id='''"'''+str(fac[0])+'''"''')
    fac.append(df_results['result_xldate'].min())
    fac.append(df_results['result_xldate'].max())
    fac.append(len(df_results.contam_id.unique()))
    return fac

# conn = vdl.sql_query_conn()
# facilities_df = pd.read_sql_query(
#     "SELECT * from facilities", conn)
# conn.close()
# facilities_list = facilities_df.values.tolist()
# test_fac = facilities_list[14412]
# start_func = time.perf_counter()
# print(facility_results_counter(test_fac))
# fin_func = time.perf_counter()
# print(f'Total time: {fin_func-start_func}')


def contam_results_counter(contaminant_info, sources):
    query_base = f'''SELECT id, fac_id, contam_id, result_date, result_xldate, int_res FROM All_Results'''
    query_sec = ''' WHERE contam_id = ''' + str(contaminant_info[0])
    query = query_base + query_sec
    conn = vdl.sql_query_conn()
    df_results = pd.read_sql_query(query, conn)
    conn.close()
    unique_facilities = df_results.fac_id.unique()
    unique_facilities_to_review = list(
        set(unique_facilities).intersection(sources))
    contaminant_info.append(df_results['result_xldate'].min())
    contaminant_info.append(df_results['result_xldate'].max())
    contaminant_info.append(len(unique_facilities))
    contaminant_info.append(len(unique_facilities_to_review))
    return contaminant_info


# conn = vdl.sql_query_conn()
# contam_df = pd.read_sql_query(
#     "SELECT * from contam_info", conn)
# conn.close()
# source_facs = vdl.facilities_to_review()['id'].values.tolist()
# test_contam = contam_df.values.tolist()[166]
# print(test_contam)
# start_func = time.perf_counter()
# results_count = contam_results_counter(test_contam, sources=source_facs)
# fin_func = time.perf_counter()
# print(results_count)
# print(f'Total time: {fin_func-start_func}')
# raise ValueError


def water_system_contam_mean(ws_id, contam_dict, facs_to_review):
    df_results = vdl.grab_water_system_results(
        ws_id='''"'''+str(ws_id)+'''"''')
    df_source_results = df_results[df_results['fac_id'].isin(facs_to_review)]

    ws_sampled_data_list = [ws_id]
    ws_sampled_and_reviewed_and_has_mcl_data_list = [ws_id]
    ws_tol_list = [ws_id]
    ws_sampled_reviewed_has_mcl_and_ninety_percent_list = [ws_id]

    sampled_contams = contam_dict['sampled']
    sampled_and_reviewed_and_has_mcl_contams = contam_dict['sampled_and_reviewed_and_has_mcl']
    tol_contams = contam_dict['tol']
    sampled_reviewed_has_mcl_and_ninety_percent_contams = contam_dict[
        'sampled_reviewed_has_mcl_and_ninety_percent']

    id_mcl_dict = contam_dict['id_mcl_dict']
    ws_contaminants = df_results.contam_id.unique().astype(int)

    if len(df_source_results) == 0:
        ws_sampled_data_list.extend([None]*len(sampled_contams))
        ws_sampled_and_reviewed_and_has_mcl_data_list.extend(
            [None]*len(sampled_and_reviewed_and_has_mcl_contams))
        ws_tol_list.extend([None]*len(tol_contams))
        ws_sampled_reviewed_has_mcl_and_ninety_percent_list.extend(
            [None]*len(sampled_reviewed_has_mcl_and_ninety_percent_contams))
        return [ws_sampled_data_list, ws_sampled_and_reviewed_and_has_mcl_data_list, ws_tol_list, ws_sampled_reviewed_has_mcl_and_ninety_percent_list]
    else:
        for sampled in sampled_contams:
            if int(sampled) in ws_contaminants:
                contam_mean = df_source_results[df_source_results['contam_id'] == str(
                    sampled)]['int_res'].astype(float).mean()
                ws_sampled_data_list.append(contam_mean)
            else:
                ws_sampled_data_list.append(None)
        for samp_rev in sampled_and_reviewed_and_has_mcl_contams:
            if int(samp_rev) in ws_contaminants:
                contam_mean = df_source_results[df_source_results['contam_id'] == str(
                    samp_rev)]['int_res'].astype(float).mean()
                contam_mean_mcl_quotient = contam_mean/id_mcl_dict[samp_rev]
                ws_sampled_and_reviewed_and_has_mcl_data_list.append(
                    contam_mean_mcl_quotient)
            else:
                ws_sampled_and_reviewed_and_has_mcl_data_list.append(None)
        for tol in tol_contams:
            if int(tol) in ws_contaminants:
                contam_mean = df_source_results[df_source_results['contam_id'] == str(
                    tol)]['int_res'].astype(float).mean()
                contam_mean_mcl_quotient = contam_mean/id_mcl_dict[tol]
                ws_tol_list.append(
                    contam_mean_mcl_quotient)
            else:
                ws_tol_list.append(None)
        for srmn in sampled_reviewed_has_mcl_and_ninety_percent_contams:
            if int(srmn) in ws_contaminants:
                contam_mean = df_source_results[df_source_results['contam_id'] == str(
                    srmn)]['int_res'].astype(float).mean()
                contam_mean_mcl_quotient = contam_mean/id_mcl_dict[srmn]
                ws_sampled_reviewed_has_mcl_and_ninety_percent_list.append(
                    contam_mean_mcl_quotient)
            else:
                ws_sampled_reviewed_has_mcl_and_ninety_percent_list.append(
                    None)
    return [ws_sampled_data_list, ws_sampled_and_reviewed_and_has_mcl_data_list, ws_tol_list, ws_sampled_reviewed_has_mcl_and_ninety_percent_list]


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()

    conn = vdl.sql_query_conn()
    contam_df = pd.read_sql_query(
        "SELECT * from contam_info", conn)
    facilities_list = pd.read_sql_query(
        "SELECT * from facilities", conn).values.tolist()
    conn.close()

    facilities_and_links = []

    for f in facilities_list:
        facilities_and_links.append([f[0], f[1], f[12], f[13]])

    all_fac_link_sublists = []
    afl_sublist_number = 0
    afl_sublist_size = 50

    for afl in range(0, len(facilities_and_links), afl_sublist_size):
        afl_sublist_number += 1
        new_afl_sublist = facilities_and_links[afl:afl+afl_sublist_size]
        all_fac_link_sublists.append([afl_sublist_number, new_afl_sublist])

    contam_match_old_res_dictionary = {}
    # matches results from new link with correct contaminant id
    contam_match_new_res_dictionary = {}
    # Simply for making results for particular id be in common unit
    contam_id_unit_dict = {}
    for i, j in contam_df.iterrows():
        contam_id_unit_dict[j['id']] = j['unit']
        # creates string that is combination of storet_no, analyte_no, mcl, dlr, trig, etc that connects to particular contam id (old and new links)
        contams_to_add = contam_id_match_key(j.values.tolist())
        if contams_to_add[0] != '':  # if old contam match is not ''
            contam_match_old_res_dictionary[contams_to_add[0]
                                            [0]] = contams_to_add[0][1]
        if contams_to_add[1] != '':  # if new contam match is not ''
            contam_match_new_res_dictionary[contams_to_add[1]
                                            [0]] = contams_to_add[1][1]
    prev_max_id = 1
    prev_start = start
    sql_table_name = 'all_results'

    for fac_link_sublist in all_fac_link_sublists:
        loop_number = fac_link_sublist[0]
        sublist_start = time.perf_counter()
        fac_links = fac_link_sublist[1]

        results_columns = ['fac_id', 'ws_id', 'contam_id', 'result_date', 'result_xldate',
                           'xmod_less_than', 'raw_result', 'int_res', 'lab_method_info']
        sublist_df = pd.DataFrame()
        sublist_results = []

        # if loop_number > 656:
        for fac in fac_links:
            f_start = time.perf_counter()
            sublist_results.extend(facility_results(
                fac, contam_match_old_res_dictionary, contam_match_new_res_dictionary, contam_id_unit_dict))
            print(
                f'Facility {fac_links.index(fac) + 1} of {len(fac_links)} took {time.perf_counter() - f_start}, total time so far is {time.perf_counter() - start} seconds.')

        sublist_df = pd.DataFrame(
            sublist_results, columns=results_columns)
        sublist_df.insert(0, 'id', range(
            prev_max_id, prev_max_id + len(sublist_df)))
        prev_max_id += len(sublist_df)
        sublist_df[["contam_id"]] = sublist_df[[
            "contam_id"]].astype(str)
        append_or_replace = 'replace' * \
            (prev_start == start) + 'append'*(prev_start != start)
        conn = vdl.sql_query_conn()
        sublist_df.to_sql(sql_table_name, conn,
                          if_exists=append_or_replace, index=False)
        conn.close()

        prev_start = sublist_start
        sublist_end = time.perf_counter()
        print(
            f'Loop {loop_number} of {afl_sublist_number}')
        print(f'Sublist Time: {sublist_end - sublist_start}')
        print(
            f'Average Time Per Facility: {(sublist_end - sublist_start)/(len(fac_links))}')
    vdl.create_index(sql_table_name, fac_id='ASC',
                     contam_id='ASC', result_xldate='ASC', int_res='DESC')
    vdl.create_index(sql_table_name, ws_id='ASC',
                     contam_id='ASC', result_xldate='ASC', int_res='DESC')
    vdl.create_index(sql_table_name, contam_id='ASC',
                     result_xldate='ASC', int_res='DESC')

    start_results_counting = time.perf_counter()
    conn = vdl.sql_query_conn()
    facilities_df = pd.read_sql_query(
        "SELECT * from facilities", conn)
    contam_df = pd.read_sql_query(
        "SELECT * from contam_info", conn)
    ws_df = pd.read_sql_query(
        "SELECT * from water_system_primary", conn)
    conn.close()
    if len(facilities_df.columns.to_list()) == 17:
        facilities_df.drop(['min_xldate', 'max_xldate',
                           'num_unique_contams'], axis=1, inplace=True)

    facilities_list = facilities_df.values.tolist()
    if len(contam_df.columns.to_list()) == 20:
        contam_df.drop(['min_xldate', 'max_xldate',
                       'unique_sampled_facilities', 'unique_sampled_and_reviewed_facilities'], axis=1, inplace=True)

    contam_list = contam_df.values.tolist()
    ws_id_list = ws_df['id'].values.tolist()
    processed_facilities_list = []
    processed_contaminants_list = []
    ws_sampled_contam_mean = []
    ws_sampled_reviewed_and_has_mcl_contam_mean = []
    ws_tol_contam_mean = []
    ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean = []
    fac_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing

        # Adding 'min_xldate', 'max_xldate', 'num_unique_contams' to facilities table
        results = executor.map(facility_results_counter, facilities_list)
        for result in results:
            processed_facilities_list.append(result)
        processed_facilities_columns = ['id', 'ws_id', 'activity_status_date', 'activity_xldate', 'state_asgn_id_number', 'facility_name', 'facility_type', 'sample_point_type',
                                        'activity_status', 'activity_reason_text', 'classification', 'facility_hyperlink', 'old_results_hyperlink', 'new_results_hyperlink', 'min_xldate', 'max_xldate', 'num_unique_contams']
        processed_facilities_df = pd.DataFrame(
            processed_facilities_list, columns=processed_facilities_columns)
        conn = vdl.sql_query_conn()
        processed_facilities_df.to_sql(
            'facilities', conn, if_exists='replace', index=False)
        conn.close()
        fac_finish = time.perf_counter()
        print(
            f'Total time to finish facility results counter: {fac_finish -fac_start}')

        # Adding 'min_xldate', 'max_xldate', 'num_unique_facilities' to contam_info table
        source_facs = vdl.facilities_to_review()['id'].values.tolist()
        results = executor.map(contam_results_counter, contam_list, [
            source_facs]*len(contam_list))
        for result in results:
            processed_contaminants_list.append(result)

        processed_contaminants_columns = ['id', 'storetno', 'analyteno', 'old_res_link', 'new_res_link', 'contam_name', 'mcl', 'dlr', 'trig', 'unit',
                                          'regulatory_effective_date', 'reg_xldate', 'contam_group', 'notes', 'method', 'library_group', 'min_xldate', 'max_xldate', 'unique_sampled_facilities', 'unique_sampled_and_reviewed_facilities']
        processed_contaminants_df = pd.DataFrame(
            processed_contaminants_list, columns=processed_contaminants_columns)
        conn = vdl.sql_query_conn()
        processed_contaminants_df.to_sql(
            'contam_info', conn, if_exists='replace', index=False)
        conn.close()
        contam_finish = time.perf_counter()
        print(
            f'Total time to finish contam results counter: {contam_finish -fac_finish}')

        # Creating new table with mean value for each contaminant at each water system (~675 columns due to one column for each contaminant)
        contam_mean_start = time.perf_counter()
        active_source_facilities = vdl.facilities_to_review()[
            'id'].values.tolist()
        contam_info_dict = vdl.contam_info_organizer(
            len_of_source_facs=len(active_source_facilities))

        results = executor.map(water_system_contam_mean, ws_id_list, [
            contam_info_dict]*len(ws_id_list), [active_source_facilities]*len(ws_id_list))
        for result in results:
            ws_sampled_contam_mean.append(result[0])
            ws_sampled_reviewed_and_has_mcl_contam_mean.append(result[1])
            ws_tol_contam_mean.append(result[2])
            ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean.append(
                result[3])

        ws_sampled_contam_mean_columns = ['ws_id']
        ws_sampled_contam_mean_columns.extend(contam_info_dict['sampled'])
        ws_sampled_reviewed_and_has_mcl_contam_mean_columns = ['ws_id']
        ws_sampled_reviewed_and_has_mcl_contam_mean_columns.extend(
            contam_info_dict['sampled_and_reviewed_and_has_mcl'])
        ws_tol_contam_mean_columns = ['ws_id']
        ws_tol_contam_mean_columns.extend(contam_info_dict['tol'])
        ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean_columns = [
            'ws_id']
        ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean_columns.extend(
            contam_info_dict['sampled_reviewed_has_mcl_and_ninety_percent'])

        ws_sampled_contam_mean_df = pd.DataFrame(
            ws_sampled_contam_mean, columns=ws_sampled_contam_mean_columns)
        ws_sampled_reviewed_has_mcl_contam_mean_df = pd.DataFrame(
            ws_sampled_reviewed_and_has_mcl_contam_mean, columns=ws_sampled_reviewed_and_has_mcl_contam_mean_columns)
        ws_tol_contam_mean_df = pd.DataFrame(
            ws_tol_contam_mean, columns=ws_tol_contam_mean_columns)
        ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean_df = pd.DataFrame(
            ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean, columns=ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean_columns)

        conn = vdl.sql_query_conn()
        ws_sampled_contam_mean_df.to_sql('ws_contam_mean_sampled', conn,
                                         if_exists='replace', index=False)
        ws_sampled_reviewed_has_mcl_contam_mean_df.to_sql('ws_contam_mean_sampled_reviewed_has_mcl', conn,
                                                          if_exists='replace', index=False)
        ws_tol_contam_mean_df.to_sql(
            'ws_contam_mean_tol', conn, if_exists='replace', index=False)
        ws_sampled_reviewed_has_mcl_and_ninety_percent_contam_mean_df.to_sql(
            'ws_contam_mean_sampled_reviewed_has_mcl_and_ninety_percent', conn, if_exists='replace', index=False)
        conn.close()
        contam_mean_finish = time.perf_counter()
        print(
            f'Took {contam_mean_finish - contam_mean_start} seconds to generate all ws contam means')

    finish_results_counting = time.perf_counter()
    finish = time.perf_counter()

    print(
        f'Time to count results: {finish_results_counting - start_results_counting}')
    print(
        f'Average time per facility: {(fac_finish - fac_start)/(len(facilities_list))}')

    print(
        f'Average time per contaminant: {(finish - fac_finish)/(len(contam_list))}')

    print(
        f'Total time for scraping and counting everything (seconds): {finish - start}')

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
