
import vapyr_date_library as vdl
import pandas as pd
import time
import cProfile
import pstats
import io
from pstats import SortKey
import datetime as dt
import concurrent.futures


def potential_standby_calendar_year_check(fac_tup):
    fac_id = str(fac_tup[0])
    fac_asd = int(fac_tup[1])
    today_year = dt.datetime.now().year
    today_xl = vdl.excel_date(dt.datetime.today())
    standby_text = 'Does not seem to be mechanically operational for this time range'
    active_text = 'Seems to be mechanically operational for this time range'
    df_all_results = vdl.grab_water_results(contam_id='', fac_id=fac_id)
    df_all_results['result_xldate'] = df_all_results['result_xldate'].astype(
        int)
    df_all_results = df_all_results.sort_values(by='result_xldate', ascending=True) \
        .drop_duplicates(subset=['result_xldate'], keep='first') \
        .reset_index(drop=True)
    start_year = vdl.yearly(fac_asd)

    df_potential_start = vdl.digit_get(vdl.date_code_format(fac_asd)[:5])[0]
    if len(df_all_results) > 0:
        if df_all_results.loc[0]['result_xldate'] < df_potential_start:
            df_potential_start = vdl.digit_get(vdl.date_code_format(
                df_all_results.loc[0]['result_xldate'])[:5])[0]
            start_year = vdl.yearly(df_all_results.loc[0]['result_xldate'])

    df_potential = pd.DataFrame([[fac_id, df_potential_start, '', '']], columns=[
                                'fac_id', 'start_date', 'end_date', 'potential_standby_determination'])
    if len(df_all_results) > 0:
        first_result = df_all_results.loc[0]['result_xldate']
        first_result_year = vdl.yearly(first_result)
        prev_year = first_result_year
        if first_result_year == start_year:
            df_potential.loc[[df_potential.index.max()], ['end_date', 'potential_standby_determination']] = [
                vdl.digit_get('Y'+str(first_result_year))[1], active_text]
        else:
            df_potential.loc[[df_potential.index.max()], ['end_date', 'potential_standby_determination']] = [
                vdl.digit_get('Y'+str(first_result_year-1))[1], standby_text]
            df_potential.loc[len(df_potential)] = [fac_id, vdl.digit_get(
                'Y'+str(first_result_year))[0], vdl.digit_get('Y'+str(first_result_year))[1], active_text]

        for i, j in df_all_results.iterrows():
            current_year = vdl.yearly(int(j[4]))
            current_year_digits = vdl.digit_get('Y'+str(current_year))
            if current_year == prev_year:
                pass
            elif current_year == prev_year + 1:
                df_potential.loc[[df_potential.index.max()], ['end_date']] = [
                    current_year_digits[1]]
            elif current_year > prev_year + 1:
                df_potential.loc[len(df_potential)] = [fac_id, vdl.digit_get(
                    'Y'+str(prev_year+1))[0], current_year_digits[0]-1, standby_text]
                df_potential.loc[len(df_potential)] = [
                    fac_id, current_year_digits[0], current_year_digits[1], active_text]
            prev_year = current_year

        if today_year > prev_year + 1:
            df_potential.loc[len(df_potential)] = [fac_id, vdl.digit_get(
                'Y'+str(prev_year+1))[0], today_xl, standby_text]
        else:
            df_potential.loc[[df_potential.index.max()], ['end_date']] = [
                today_xl]
    else:
        df_potential.loc[[df_potential.index.max()], [
            'end_date', 'potential_standby_determination']] = [today_xl, standby_text]
    start_dates_of_timelines = [
        ('effective', 732), ('cycle', 40544), ('period', 42736)]
    all_potential_output = []

    for timelines in start_dates_of_timelines:
        timeline_name = timelines[0]
        timeline_start = timelines[1]

        df_trial = df_potential.copy()
        df_user = df_potential.copy()

        df_trial.drop(df_trial[df_trial['start_date']
                      > 43830].index, inplace=True)
        df_trial.loc[df_trial["end_date"] > 43830, "end_date"] = 43830

        df_trial.drop(df_trial[df_trial['end_date'] <
                      timeline_start].index, inplace=True)
        df_trial.loc[df_trial["start_date"] <
                     timeline_start, "start_date"] = timeline_start
        df_user.drop(df_user[df_user['end_date'] <
                     timeline_start].index, inplace=True)
        df_user.loc[df_user["start_date"] <
                    timeline_start, "start_date"] = timeline_start

        df_trial['query'] = 'trial_' + timeline_name
        df_user['query'] = 'user_' + timeline_name
        all_potential_output.extend(df_trial.values.tolist())
        all_potential_output.extend(df_user.values.tolist())
    return all_potential_output


# sample_ftup = (8074, 44820, 'WL', '138995', 'SWP')
# psd_df = pd.DataFrame(potential_standby_calendar_year_check(sample_ftup), columns=[
#                       'fac_id', 'start_date', 'end_date', 'potential_standby_determination', 'query'])
# print(psd_df)
# raise ValueError


def priority_finder(track):
    tol_priority = [('H'), ('A', 'D', 'F', 'G', 'J', 'X',
                            'Y', 'Z'), ('E', 'I'), ('B'), ('C')]
    for level in tol_priority:
        if track in level:
            return tol_priority.index(level) + 1


def method_review(facility_and_method_and_library, timeline):
    facility_id = facility_and_method_and_library[0]
    method = str(facility_and_method_and_library[1])
    library = str(facility_and_method_and_library[2])
    tl_string = f'user_{timeline}_timeline'

    contam_query = f'''SELECT * from 'contam_info' WHERE method = "{method}" AND library_group= "{library}"'''
    timeline_query = f"SELECT fac_id, contam_id, start_date, end_date, overage, determination, color, qhr_value, qhr_quarter, track FROM {tl_string} WHERE fac_id={facility_id}"

    conn = vdl.sql_query_conn()
    contam_id_pd = pd.read_sql_query(contam_query, conn)
    timeline_tables = pd.read_sql_query(timeline_query, conn)
    conn.close()

    contam_id = contam_id_pd['id'].tolist()
    method_variables = {"fac_id": [], "method": [], "priority_contam_id": [], "track": [], "end_date": [
    ], "determination": [], "color": [], "priority_level": [], "requirement": [], "qhr_quarter": []}

    tol_requirement = {1: 'Monthly sampling', 2: 'Quarterly sampling',
                       3: 'Annual sampling on quarter(s) of highest result', 4: 'Annual sampling', 5: 'Triannual sampling'}

    for id in contam_id:
        subtable = timeline_tables.loc[timeline_tables["contam_id"] == id]
        if len(subtable):
            fac_id = facility_id
            last_track = subtable.loc[subtable.index.max(), 'track']
            priority = priority_finder(last_track)
            end_date = subtable.loc[subtable.index.max(), 'end_date']
            determination = subtable.loc[subtable.index.max(), 'determination']
            color = subtable.loc[subtable.index.max(), 'color']
            qhr_quarter = subtable.loc[subtable.index.max(), 'qhr_quarter']
            method_variables["fac_id"].append(fac_id)
            method_variables["method"].append(method)
            method_variables["priority_contam_id"].append(id)
            method_variables["track"].append(last_track)
            method_variables["end_date"].append(end_date)
            method_variables["determination"].append(determination)
            method_variables["color"].append(color)
            method_variables["priority_level"].append(priority)
            method_variables["requirement"].append(
                tol_requirement.get(priority))
            method_variables["qhr_quarter"].append(qhr_quarter)

    priority_df = pd.DataFrame(method_variables)
    priority_df = priority_df.sort_values(
        ['priority_level', 'end_date', 'priority_contam_id'], ignore_index=True)
    priority_output = priority_df.iloc[0].tolist()

    timeline_tables = timeline_tables[timeline_tables['contam_id'].isin(
        contam_id)]
    date_tuple_list = []
    start_dates = timeline_tables['start_date'].tolist()
    end_dates = timeline_tables['end_date'].tolist()
    all_dates = start_dates + end_dates
    all_dates_no_dup = list(set(all_dates))
    all_dates_no_dup = sorted(all_dates_no_dup)
    for date in all_dates_no_dup:
        if all_dates_no_dup.index(date) == len(all_dates_no_dup) - 1:
            pass
        else:
            next_date = all_dates_no_dup[all_dates_no_dup.index(date)+1]
            mid_date = (date + next_date)/2
            test_tuple = (date, next_date, mid_date)
            if test_tuple not in date_tuple_list:
                date_tuple_list.append(test_tuple)

    ts_overage_counter = 0
    adj_overage_counter = 0
    method_review_output = []
    for tup in date_tuple_list:
        initial_date = tup[0]
        final_date = tup[1]
        middle_date = tup[2]
        color = ''
        ts_overage = ''
        adj_overage = 0
        if final_date == all_dates_no_dup[-1]:
            final_date += 1

        tuple_start_time_segments = timeline_tables[(initial_date <= timeline_tables['start_date']) & (
            timeline_tables['start_date'] < final_date)]
        tuple_end_time_segments = timeline_tables[(initial_date <= timeline_tables['end_date']) & (
            timeline_tables['end_date'] < final_date)]
        tuple_overspan_time_segments = timeline_tables[(initial_date > timeline_tables['start_date']) & (
            timeline_tables['end_date'] > final_date)]  # Covers time segments that start before and end after the tuple
        tuple_all_time_segments = pd.concat(
            [tuple_start_time_segments, tuple_end_time_segments, tuple_overspan_time_segments]).drop_duplicates()

        if '' in set(tuple_all_time_segments['overage']):
            ts_overage = '0'

            if 'RED' in set(tuple_all_time_segments['color']):
                color = 'RED'
            elif 'YELLOW' in set(tuple_all_time_segments['color']):
                color = 'YELLOW'
            elif 'TBD' in set(tuple_all_time_segments['color']):
                color = 'TBD'
            elif 'TBD' in set(tuple_all_time_segments['color']):
                color = 'TBD'
            else:
                color = 'GREEN'
        elif 'TBD' in set(tuple_all_time_segments['color']):
            color = 'TBD'
        else:
            color = 'GREEN'

        if '0' in set(tuple_all_time_segments['overage']):
            ts_overage = '0'

        if ts_overage != '0' and (color == 'TBD' or color == 'GREEN'):
            ts_overage = str(tuple_all_time_segments['overage'].min())
            ts_overage_rows = tuple_all_time_segments[tuple_all_time_segments['overage'] == ts_overage]
            ts_overage_rows.drop(
                ts_overage_rows[ts_overage_rows['end_date'] == initial_date].index, inplace=True)

            if len(ts_overage_rows) == 0:
                ts_overage_rows.at[124, 'overage_dud_check'] = False
                ts_overage = '0'
                adj_overage = float(0)
            else:
                ts_overage = int(tuple_all_time_segments['overage'].min())
                ts_overage_rows['time_span'] = ts_overage_rows['end_date'] - \
                    ts_overage_rows['start_date']
                tuple_days = int(final_date - initial_date)
                ts_overage_days = int(
                    max(ts_overage_rows['time_span'].min(), 1))
                adj_overage = ts_overage * (tuple_days/ts_overage_days)

        if ts_overage == '0':
            adj_overage = 0
        adj_overage_counter += adj_overage
        method_review_output.append(
            [facility_id, method, initial_date, final_date, middle_date, color, adj_overage])
        ts_overage_counter += int(ts_overage)

    return [priority_output, method_review_output]


def review_append(dict_review, review_list_of_list):
    for rev in review_list_of_list:
        dict_review['fac_id'].append(rev[0])
        dict_review['method'].append(rev[1])
        dict_review['initial_date'].append(rev[2])
        dict_review['final_date'].append(rev[3])
        dict_review['middle_date'].append(rev[4])
        dict_review['color'].append(rev[5])
        dict_review['adj_overage'].append(rev[6])
    return dict_review


def priority_append(dict_priority, priority_list):
    dict_priority['fac_id'].append(priority_list[0])
    dict_priority['method'].append(priority_list[1])
    dict_priority['priority_contam_id'].append(priority_list[2])
    dict_priority['track'].append(priority_list[3])
    dict_priority['end_date'].append(priority_list[4])
    dict_priority['determination'].append(priority_list[5])
    dict_priority['color'].append(priority_list[6])
    dict_priority['priority_level'].append(priority_list[7])
    dict_priority['requirement'].append(priority_list[8])
    dict_priority['qhr_quarter'].append(priority_list[9])


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()

    conn = vdl.sql_query_conn()
    df_facilities = pd.read_sql_query(
        "SELECT fac.id, fac.activity_xldate, fac.sample_point_type, fac.activity_status, fac.facility_type, wsp.pserved, primary_source_water_type from facilities fac LEFT JOIN water_system_primary wsp ON fac.ws_id = wsp.id", conn)
    conn.close()

    # For column headers, remove all capitalizations and replace spaces with underscores so it plays well with pandas
    df_facilities.columns = df_facilities.columns.str.strip().str.lower(
    ).str.replace(' ', '_').str.replace('(', '').str.replace(')', '')  # future warning
    # Filter for facilities that are active and raw water:
    df_act_raw_sr_facs = df_facilities[(df_facilities["activity_status"] == 'A') & (df_facilities["sample_point_type"] == 'RW') | (
        df_facilities["sample_point_type"] == 'SR') & (df_facilities["facility_type"] != 'DS')]
    # Cut the dataframe down to only the columns needed for review algo:
    df_act_raw_sr_facs = df_act_raw_sr_facs[[
        "id", "activity_xldate", "facility_type", "pserved", "primary_source_water_type"]]
    fac_tups = [tuple(x) for x in df_act_raw_sr_facs.to_numpy()]
    potential_standby_columns = [
        'fac_id', 'start_date', 'end_date', 'potential_standby_determination', 'query']
    potential_timer = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(potential_standby_calendar_year_check, fac_tups)
        potential_standby_df = pd.DataFrame()
        potential_standby_list_of_list = []
        for result in results:
            potential_standby_list_of_list.extend(result)
    potential_standby_df = pd.DataFrame(
        potential_standby_list_of_list, columns=potential_standby_columns)
    conn = vdl.sql_query_conn()
    potential_standby_df.to_sql(
        'potential_mech_downage_timeline', conn, if_exists='replace', index=False)
    conn.close()
    vdl.create_index('potential_mech_downage_timeline',
                     fac_id='ASC', start_date='ASC')
    vdl.create_index('potential_mech_downage_timeline',
                     query='ASC')
    print(time.perf_counter()-potential_timer)

    # Begin method review:
    conn = vdl.sql_query_conn()
    df_timeline = pd.read_sql_query(
        f"SELECT * from user_effective_timeline", conn)
    df_contam = pd.read_sql_query(f"SELECT * from contam_info", conn)
    conn.close()
    # list of all unique facilities
    fac_list = df_timeline['fac_id'].unique().tolist()
    df_timeline = pd.DataFrame()  # this is to free up memory

    libraries_ready = ['tol']
    timelines = ['effective', 'cycle', 'period']
    method_lib_combos = []
    for i, j in df_contam.iterrows():
        if j['library_group'] in libraries_ready:
            test_method_lib_combo = [j['method'], j['library_group']]
            if test_method_lib_combo not in method_lib_combos and j['method'] != None:
                method_lib_combos.append(test_method_lib_combo)

    fac_method_lib_combos = []
    new_fml_combo = []

    for f in fac_list:
        for fm in method_lib_combos:
            new_fml_combo = [f] + fm
            fac_method_lib_combos.append(new_fml_combo)

    all_fac_method_lib_sublists = []
    sublist_number = 0
    for i in range(0, len(fac_method_lib_combos), 10000):
        sublist_number += 1
        new_sublist = fac_method_lib_combos[i:i+10000]
        all_fac_method_lib_sublists.append([sublist_number, new_sublist])

    method_start = time.perf_counter()
    prev_method_start = method_start
    for t in timelines:
        for sublist in all_fac_method_lib_sublists:
            sublist_start = time.perf_counter()
            dict_priority = {"fac_id": [], "method": [], "priority_contam_id": [], "track": [], "end_date": [
            ], "determination": [], "color": [], "priority_level": [], "requirement": [], "qhr_quarter": []}
            dict_review = {"fac_id": [], "method": [], "initial_date": [
            ], "final_date": [], "middle_date": [], "color": [], "adj_overage": []}
            timeline_list = [t]*len(sublist[1])
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(
                    method_review, sublist[1], timeline_list)
                end_results_creation = time.perf_counter()
                print(
                    f'Results creation: {end_results_creation-sublist_start}')
                for result in results:
                    priority_append(dict_priority, result[0])
                    review_append(dict_review, result[1])
                df_priority = pd.DataFrame.from_dict(dict_priority)
                df_priority['timeline'] = t
                df_review = pd.DataFrame.from_dict(dict_review)
                df_review['timeline'] = t
                end_results_iteration = time.perf_counter()
                print(
                    f'Results iteration: {end_results_iteration-end_results_creation}')
                start_write_to_sql = time.perf_counter()

                append_or_replace = 'replace' * \
                    (prev_method_start == method_start) + \
                    'append'*(prev_method_start != method_start)
                conn = vdl.sql_query_conn()
                df_review.to_sql('method_historical_review',
                                 conn, if_exists=append_or_replace, index=False)
                df_priority.to_sql('method_priority_schedule',
                                   conn, if_exists=append_or_replace, index=False)
                conn.close()
                df_review = pd.DataFrame()
                df_priority = pd.DataFrame()
                end_write_to_sql = time.perf_counter()
                prev_method_start = end_write_to_sql
                print(
                    f'Finished loop {sublist[0]} of {len(all_fac_method_lib_sublists)} in {t}')
                print(f'Time so far: {end_write_to_sql - start}')

    vdl.create_index('method_priority_schedule', fac_id='ASC',
                     method='ASC', priority_contam_id='ASC', timeline='ASC')
    vdl.create_index('method_historical_review', fac_id='ASC', method='ASC',
                     initial_date='ASC', final_date='ASC', timeline='ASC')

    conn = vdl.sql_query_conn()
    df_overage_check = pd.read_sql_query(
        f"SELECT * from method_historical_review", conn)
    conn.close()
    timelines_to_check = ['effective', 'cycle', 'period']
    for t in timelines_to_check:
        df_timeline_overage = df_overage_check[df_overage_check['timeline'] == t]
        total_of_all_adj_overages = df_timeline_overage['adj_overage'].sum()
        print(
            f'The total of all the adjusted overages in {t} is: {total_of_all_adj_overages}')

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

# Jae Test 12.18.22L
# 1318.4384543999913 seconds
