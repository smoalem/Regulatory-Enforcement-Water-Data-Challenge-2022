
from ipaddress import AddressValueError
from multiprocessing.sharedctypes import Value
from pickletools import read_decimalnl_long
from threading import currentThread
from weakref import ref
import pandas as pd
import numpy as np
import time
import concurrent.futures
import vapyr_date_library as vdl
import cProfile
import pstats
import io
from pstats import SortKey
import math
import regex as re


def color_timespan_and_sample_frequency_modifier(df):
    raw_days_list = []
    adj_days_list = []
    if len(df) == 0:
        return [0, 0]
    for i, j in df.iterrows():
        sf_det_dict = {'sameday': 0.9, 'daily': 0.81, 'sameweek': 0.729, 'weekly': 0.6561, 'samemonth': 0.5905, 'monthly': 0.5314, 'samequarter': 0.4783,
                       'quarterly': 0.4305, 'sameyear': 0.3874, 'annual': 0.3487, 'triannual': 0.3138, '9yearschedule': 0.2824, '>9yearfrequency': 0.2542}
        sf_text = j['sample_frequency']
        color = j['color']
        days = j['end_date'] - j['start_date'] + 1
        years = days/365
        if sf_text == 'No samples taken' or color == 'GREEN':
            raw_days_list.append(days)
            adj_days_list.append(days)
        elif sf_text == 'Only one sample received in this date range':
            sf_multiplier = 0.22878
            sf_coefficient = 1 - sf_multiplier * min(1, (1/years))
            raw_days_list.append(days)
            adj_days_list.append(days*sf_coefficient)
        else:
            # remove all spaces and only look at text to the right of the colon
            sf_num_samples = int(sf_text[:sf_text.find(' sample')])
            sf_list = re.sub(' ', '', sf_text[sf_text.find(':')+1:]).split(',')
            for sf in sf_list:
                sf_percent = float(sf[:sf.find('%')])
                sf_multiplier = sf_det_dict[sf[sf.find('%')+1:]]
                sf_coefficient = 1 - (sf_percent/100) * \
                    (sf_multiplier) * min(1, sf_num_samples/years)
                raw_days_list.append(days)
                adj_days_list.append(days*sf_coefficient)
        return [sum(raw_days_list), sum(adj_days_list)]


def color_score(df):
    if (df['color'] == 'TBD').all():
        return ['TBD', 'TBD', 'TBD', 'TBD', 'TBD', 'TBD']
    elif (df['color'] == 'BLACK').all():
        return ['PMD', 'PMD', 'PMD', 'PMD', 'PMD', 'PMD']
    elif len(df[df['color'] == 'BLACK']) + len(df[df['color'] == 'TBD']) == len(df):
        return ['PMD', 'PMD', 'PMD', 'PMD', 'PMD', 'PMD']
    else:
        df_red = df[df['color'] == "RED"]
        df_yellow = df[df['color'] == "YELLOW"]
        df_green = df[df['color'] == "GREEN"]
        red_scores = color_timespan_and_sample_frequency_modifier(df_red)
        yellow_scores = color_timespan_and_sample_frequency_modifier(df_yellow)
        green_scores = color_timespan_and_sample_frequency_modifier(df_green)
        red_raw_days = red_scores[0]
        red_adj_days = red_scores[1]
        yellow_raw_days = yellow_scores[0]
        yellow_adj_days = yellow_scores[1]
        green_raw_days = green_scores[0]

        # TBD periods of time are not part of total calculation
        total_days = red_raw_days + yellow_raw_days + green_raw_days
        red_lean_score = ((red_adj_days * 1) +
                          (yellow_adj_days * 0.5))/total_days
        yellow_lean_score = ((red_adj_days * 0.5) +
                             (yellow_adj_days * 1.0)) / total_days
        green_score = green_raw_days/total_days
        return [total_days, (red_raw_days, red_adj_days), (yellow_raw_days, yellow_adj_days), red_lean_score, yellow_lean_score, green_score]


def time_seg_track_switch_counter(ref_timeline):
    num_time_segments = len(ref_timeline)
    previous_track = ref_timeline['track'].iloc[0]
    num_track_switches = 0
    for i, j in ref_timeline.iterrows():
        current_track = j['track']
        if current_track != previous_track:
            num_track_switches += 1
            previous_track = current_track
    return [num_time_segments, num_track_switches]


def average_facility_method_priority_level(fac_timeline):
    query = f'''SELECT * from method_priority_schedule WHERE timeline = {str("'"+fac_timeline[1]+"'")} AND fac_id = {fac_timeline[0]}'''
    conn = vdl.sql_query_conn()
    df_method_priority = pd.read_sql_query(query, conn)
    conn.close()
    average_priority = df_method_priority['priority_level'].mean()
    return average_priority


# print(average_facility_method_priority_level([27950, 'period']))
# raise ValueError


def fac_contam_score(combo):
    facility = str(combo[0])
    contam_id = str(combo[1])
    reference_timelines = ['effective', 'cycle', 'period']
    dict_percent_green = {}
    dict_compliance_offset = {}
    dict_first_row_color = {}
    dict_last_row_color = {}
    dict_time_seg_and_track_switch = {}
    try:
        for ref in reference_timelines:
            reference_timeline = 'user_' + ref + '_timeline'
            query = f'''SELECT * from {reference_timeline} WHERE contam_id = {contam_id} AND fac_id = {facility}'''
            conn = vdl.sql_query_conn()
            df_reference_timeline = pd.read_sql_query(query, conn)
            conn.close()
            dict_first_row_color[ref] = df_reference_timeline['color'].iloc[0]
            dict_last_row_color[ref] = df_reference_timeline['color'].iloc[len(
                df_reference_timeline)-1]
            compliance_offset_scores = color_score(df_reference_timeline)
            dict_percent_green[ref] = compliance_offset_scores[5]
            dict_compliance_offset[ref] = compliance_offset_scores[1:5]
            dict_time_seg_and_track_switch[ref] = time_seg_track_switch_counter(
                df_reference_timeline)
        if list(dict_percent_green.values()).count('TBD') == 3:
            return [facility, contam_id, 'TBD', 'TBD', 'TBD', 'TBD', 'TBD']
        elif list(dict_percent_green.values()).count(0.0) + list(dict_percent_green.values()).count('PMD') == 3:
            return [facility, contam_id, 'PMD', 'PMD', 'PMD', 'PMD', 'PMD']
        else:
            dict_percent_green = {
                key: 0.0 if val == 'TBD' else val for key, val in dict_percent_green.items()}
            dict_percent_green = {
                key: 0.0 if val == 'PMD' else val for key, val in dict_percent_green.items()}

        effective_first_green = dict_first_row_color['effective'] == 'GREEN'
        effective_last_green = dict_last_row_color['effective'] == 'GREEN'
        effective_most_green = max(dict_percent_green.values(
        )) == dict_percent_green['effective'] and dict_percent_green['effective'] > 0
        maxed_out_cycle_score = dict_compliance_offset['cycle'][2] == 0.0

        cycle_first_green = dict_first_row_color['effective'] == 'GREEN'
        cycle_most_green = max(dict_percent_green.values(
        )) == dict_percent_green['cycle'] and dict_percent_green['cycle'] > 0
        maxed_out_period_score = dict_compliance_offset['period'][2] == 0.0

        conn = vdl.sql_query_conn()
        df_potential_mech_downage = pd.read_sql_query(
            f'''SELECT * from potential_mech_downage_timeline WHERE fac_id = {facility}''', conn)
        conn.close()

        df_period_last_ts = df_potential_mech_downage[df_potential_mech_downage['query'] == 'user_period'].tail(
            1)

        no_samples_since_2021 = (df_period_last_ts['potential_standby_determination'] == 'Does not seem to be mechanically operational for this time range').all(
        ) and (df_period_last_ts['start_date'] <= 44197).all()

        if no_samples_since_2021:
            return [facility, contam_id, 'PMD', 'PMD', 'PMD', 'PMD', 'PMD']
        elif effective_first_green or effective_last_green or effective_most_green or maxed_out_cycle_score:
            if effective_first_green or effective_last_green or effective_most_green:
                baseline = 900
                max_drop = 299
                target_timeline = 'effective'
            else:
                baseline = 700
                max_drop = 99
                target_timeline = 'cycle'
            red_lean_coefficient = dict_compliance_offset['effective'][2]
            yellow_lean_coefficient = dict_compliance_offset['effective'][3]
        elif cycle_first_green or cycle_most_green or maxed_out_period_score:
            if cycle_first_green or cycle_most_green:
                baseline = 600
                max_drop = 299
                target_timeline = 'cycle'
            else:
                baseline = 400
                max_drop = 99
                target_timeline = 'period'
            red_lean_coefficient = dict_compliance_offset['cycle'][2]
            yellow_lean_coefficient = dict_compliance_offset['cycle'][3]
        else:
            red_lean_coefficient = dict_compliance_offset['period'][2]
            yellow_lean_coefficient = dict_compliance_offset['period'][3]
            baseline = 300
            max_drop = 299
            target_timeline = 'period'

        compliance_red_score = baseline - max_drop * red_lean_coefficient
        compliance_yellow_score = baseline - max_drop * yellow_lean_coefficient
        # num_time_segment = dict_time_seg_and_track_switch[target_timeline][0]
        num_time_segment = dict_time_seg_and_track_switch['effective'][0] + \
            dict_time_seg_and_track_switch['cycle'][0] + \
            dict_time_seg_and_track_switch['period'][0]
        num_track_switch = dict_time_seg_and_track_switch['effective'][1] + \
            dict_time_seg_and_track_switch['cycle'][1] + \
            dict_time_seg_and_track_switch['period'][1]
        compliance_output = [
            facility, contam_id, compliance_red_score, compliance_yellow_score, target_timeline, num_time_segment, num_track_switch]
        return compliance_output

    except:
        print(combo)
        raise ValueError


# test_start = time.perf_counter()
# print(fac_contam_score((10730, 418)))

# print(f'Time to process test (seconds): {time.perf_counter() - test_start}')
# print(time.perf_counter() - test_start)
# raise ValueError


def fac_contam_score_percentile(compliance_tuple, red_lean_list, yellow_lean_list):
    fac_id = compliance_tuple[0]
    contam_id = compliance_tuple[2]
    red_lean_score = compliance_tuple[3]
    yellow_lean_score = compliance_tuple[4]
    target_timeline = compliance_tuple[5]
    num_time_segments = compliance_tuple[6]
    num_track_switches = compliance_tuple[7]
    if red_lean_score == 'TBD' or red_lean_score == 'PMD':
        red_percentile = red_lean_score
        yellow_percentile = red_lean_score
    else:
        red_calc_abs_comp = (np.abs(red_lean_list) < float(red_lean_score))
        yellow_calc_abs_comp = (np.abs(yellow_lean_list)
                                < float(red_lean_score))
        red_len_list = float(len(red_lean_list))
        yellow_len_list = float(len(yellow_lean_list))
        red_calc_sub = red_calc_abs_comp / red_len_list
        yellow_calc_sub = yellow_calc_abs_comp / yellow_len_list
        red_percentile = math.floor(sum(red_calc_sub)*100)
        yellow_percentile = math.floor(sum(yellow_calc_sub)*100)
    return [fac_id, contam_id, red_lean_score, yellow_lean_score, red_percentile, yellow_percentile, target_timeline, num_time_segments, num_track_switches]


# conn = vdl.sql_query_conn()
# df_compliance = pd.read_sql_query(
#     f"SELECT * from score_and_percentile_fac_contam where contam_id = {str(400)}", conn)
# conn.close()
# df_comparison = df_compliance[(df_compliance['red_lean_score'] != 'TBD') & (
#     df_compliance['red_lean_score'] != 'PMD')]
# red_lean_list = np.array(
#     df_comparison['red_lean_score'].astype(float).to_list())
# yellow_lean_list = np.array(
#     df_comparison['yellow_lean_score'].astype(float).to_list())
# print(fac_contam_score_percentile((8176, 400, 101, 464.5733024691358, 371.2511574074074, 'cycle', 11, 0),
#                                   red_lean_list, yellow_lean_list))
# raise ValueError


def ave_fac_score(fac):
    facility = str(fac)
    conn = vdl.sql_query_conn()
    df_compliance_percentiles = pd.read_sql_query(
        f"SELECT * from score_and_percentile_fac_contam WHERE fac_id= {facility}", conn)
    conn.close()

    red_lean_score_list = df_compliance_percentiles[
        (df_compliance_percentiles['red_lean_score'] != 'TBD') & (df_compliance_percentiles['red_lean_score'] != 'PMD')]['red_lean_score'].tolist()
    red_lean_score_list = [float(i) for i in red_lean_score_list]

    yellow_lean_score_list = df_compliance_percentiles[
        (df_compliance_percentiles['yellow_lean_score'] != 'TBD') & (df_compliance_percentiles['yellow_lean_score'] != 'PMD')]['yellow_lean_score'].tolist()
    yellow_lean_score_list = [float(i) for i in yellow_lean_score_list]

    if len(red_lean_score_list) == 0:
        if len(df_compliance_percentiles[df_compliance_percentiles['red_lean_score'] == 'TBD']['red_lean_score']) > 0:
            average_percentiles = [facility, 'TBD',
                                   'TBD', 'TBD', 'TBD', 'TBD', 'TBD']
        else:
            average_percentiles = [facility, 'PMD',
                                   'PMD', 'PMD', 'PMD', 'PMD', 'PMD']
    else:
        ave_red_lean_score = round(sum(
            red_lean_score_list)/len(red_lean_score_list), 2)
        ave_yellow_lean_score = round(sum(
            yellow_lean_score_list)/len(yellow_lean_score_list), 2)
        ave_target_timeline = df_compliance_percentiles[(df_compliance_percentiles['red_lean_score'] != 'TBD') & (
            df_compliance_percentiles['red_lean_score'] != 'PMD')]['target_timeline'].mode()[0]
        ave_method_priority_level = average_facility_method_priority_level(
            [fac, ave_target_timeline])
        ave_num_time_segments = df_compliance_percentiles[(df_compliance_percentiles['red_lean_score'] != 'TBD') & (
            df_compliance_percentiles['red_lean_score'] != 'PMD')]['num_time_segments'].astype(int).mean()
        ave_num_track_switches = df_compliance_percentiles[(df_compliance_percentiles['red_lean_score'] != 'TBD') & (
            df_compliance_percentiles['red_lean_score'] != 'PMD')]['num_track_switches'].astype(int).mean()
        average_percentiles = [
            facility, ave_red_lean_score, ave_yellow_lean_score, ave_target_timeline, ave_method_priority_level, ave_num_time_segments, ave_num_track_switches]
    return average_percentiles


# print(ave_fac_score(27950))
# raise ValueError


def percentile_of_ave_fac_score(compliance_tuple, red_lean_list, yellow_lean_list):
    fac_id = compliance_tuple[0]
    ws_id = compliance_tuple[1]
    ave_red_lean_score = compliance_tuple[2]
    ave_yellow_lean_score = compliance_tuple[3]
    ave_target_timeline = compliance_tuple[4]
    ave_method_priority_level = compliance_tuple[5]
    ave_num_time_segments = compliance_tuple[6]
    ave_num_track_switches = compliance_tuple[7]
    if ave_red_lean_score == 'TBD' or ave_red_lean_score == 'PMD':
        ave_red_percentile = ave_red_lean_score
        ave_yellow_percentile = ave_red_lean_score
    else:
        red_calc_abs_comp = (np.abs(red_lean_list) < float(ave_red_lean_score))
        yellow_calc_abs_comp = (np.abs(yellow_lean_list)
                                < float(ave_red_lean_score))
        red_len_list = float(len(red_lean_list))
        yellow_len_list = float(len(yellow_lean_list))
        red_calc_sub = red_calc_abs_comp / red_len_list
        yellow_calc_sub = yellow_calc_abs_comp / yellow_len_list
        ave_red_percentile = math.floor(sum(red_calc_sub)*100)
        ave_yellow_percentile = math.floor(sum(yellow_calc_sub)*100)
    return [fac_id, ws_id, ave_red_lean_score, ave_yellow_lean_score, ave_red_percentile, ave_yellow_percentile, ave_target_timeline, ave_method_priority_level, ave_num_time_segments, ave_num_track_switches]


def facility_overage_counter(fac_compliance):
    fac_id = int(fac_compliance[0])
    ws_id = int(fac_compliance[1])
    ave_red_lean_score = fac_compliance[2]
    if ave_red_lean_score == 'PMD' or ave_red_lean_score == 'TBD':
        return [fac_id, ws_id, ave_red_lean_score, ave_red_lean_score]
    df_fac_results = vdl.grab_water_results(fac_id='''"'''+str(fac_id)+'''"''')

    conn = vdl.sql_query_conn()
    df_fac_method_historical = pd.read_sql_query(
        f"SELECT * from 'method_historical_review' WHERE fac_id = {fac_id} and timeline='effective'", conn)
    conn.close()
    try:
        unique_sample_dates = float(
            len(pd.unique(df_fac_results['result_xldate'])))
        overage_total = float(df_fac_method_historical['adj_overage'].sum())
        overage_rate = overage_total/unique_sample_dates
        # print([fac_id, ws_id, overage_total, overage_rate])
    except:
        print(fac_compliance)
        raise ValueError

    return [int(fac_id), int(ws_id), str(overage_total), str(overage_rate)]


# start = time.perf_counter()
# facility_overage_counter((22399, 4409))
# print(time.perf_counter() - start)
# raise ValueError

def facility_overage_percentile(fac_overage_list, all_facs_overage_rate_list):
    fac_id = fac_overage_list[0]
    ws_id = fac_overage_list[1]
    fac_overage_total = fac_overage_list[2]
    fac_overage_rate = fac_overage_list[3]
    if fac_overage_total == 'PMD' or fac_overage_total == 'TBD':
        return [fac_id, ws_id, fac_overage_total, fac_overage_total, fac_overage_total]
    try:
        overage_calc_abs_comp = (np.abs(all_facs_overage_rate_list)
                                 < float(fac_overage_rate))
        overage_len_list = float(len(all_facs_overage_rate_list))
        overage_calc_sub = overage_calc_abs_comp / overage_len_list
        fac_overage_percentile = (math.floor(
            sum(overage_calc_sub)*100)-99) * (-1)
    except:
        print(fac_overage_list)
        print(fac_overage_rate)
        print(type(fac_overage_rate))
        print(type(all_facs_overage_rate_list[0]))
        raise ValueError

    return [fac_id, ws_id, fac_overage_total, fac_overage_rate, fac_overage_percentile]


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()
    prev_start = start
    conn = vdl.sql_query_conn()
    # Just for getting the unique facid/contamid combos
    df_timeline = pd.read_sql_query(
        f"SELECT fac_id, contam_id from 'user_period_timeline'", conn)
    conn.close()
    # list of all unique facilities
    fac_list = df_timeline['fac_id'].unique().tolist()
    contam_list = df_timeline['contam_id'].unique(
    ).tolist()  # list of all unique contam_id
    df_timeline = pd.DataFrame()  # this is to free up memory
    all_combos_list = [[f, c] for f in fac_list for c in contam_list]
    write_table_name = "score_and_percentile_fac_contam"
    all_combos_sublists = []
    for i in range(0, len(all_combos_list), 100000):
        all_combos_sublists.append(all_combos_list[i:i+100000])

    print('Ready to start')
    timeline_start = time.perf_counter()
    df_compliance = pd.DataFrame()
    conn = vdl.sql_query_conn()
    df_facilities = pd.read_sql_query(
        f"SELECT * from facilities", conn)[['id', 'ws_id']]
    conn.close()
    df_facilities.rename(columns={'id': 'fac_id'}, inplace=True)

    print(f'Number of sublists: {len(all_combos_sublists)}')
    for sublist in all_combos_sublists:
        print(f'Starting loop {all_combos_sublists.index(sublist)+1}')
        sublist_start = time.perf_counter()
        dict_compliance = {'fac_id': [], 'contam_id': [],
                           'red_lean_score': [], 'yellow_lean_score': [], 'target_timeline': [], 'num_time_segments': [], 'num_track_switches': []}
        with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
            results = executor.map(fac_contam_score, sublist)
            end_results_creation = time.perf_counter()
            print(f'Results creation: {end_results_creation-sublist_start}')
            for result in results:
                dict_compliance['fac_id'].append(result[0])
                dict_compliance['contam_id'].append(result[1])
                dict_compliance['red_lean_score'].append(result[2])
                dict_compliance['yellow_lean_score'].append(result[3])
                dict_compliance['target_timeline'].append(result[4])
                dict_compliance['num_time_segments'].append(result[5])
                dict_compliance['num_track_switches'].append(result[6])
            df_compliance = pd.DataFrame.from_dict(dict_compliance)
            end_results_iteration = time.perf_counter()
            print(
                f'Results iteration: {end_results_iteration-end_results_creation}')

            start_write_to_sql = time.perf_counter()
            df_compliance = df_compliance.astype(
                {'fac_id': int, 'contam_id': int, 'red_lean_score': str, 'yellow_lean_score': str, 'target_timeline': str, 'num_time_segments': str, 'num_track_switches': str})
            df_compliance = pd.merge(
                df_facilities, df_compliance, on="fac_id", how="inner")
            append_or_replace = 'replace' * \
                (prev_start == start) + 'append'*(prev_start != start)
            conn = vdl.sql_query_conn()
            df_compliance.to_sql(write_table_name, conn,
                                 if_exists=append_or_replace, index=False)
            conn.close()
            df_compliance = pd.DataFrame()
            end_write_to_sql = time.perf_counter()
            prev_start = end_write_to_sql
            print(
                f'Time so far in score determination: {end_write_to_sql - start}')

    compliance_scores_percentiles_list = []
    compliance_scores_percentiles_columns = [
        'fac_id', 'contam_id', 'red_lean_score', 'yellow_lean_score', 'red_lean_percentile', 'yellow_lean_percentile', 'target_timeline', 'num_time_segments', 'num_track_switches']
    for contam in contam_list:
        contam_start = time.perf_counter()
        conn = vdl.sql_query_conn()
        df_compliance = pd.read_sql_query(
            f"SELECT * from score_and_percentile_fac_contam where contam_id = {str(contam)}", conn)
        conn.close()
        compliance_tups = [tuple(x) for x in df_compliance.to_numpy()]
        df_comparison = df_compliance[(df_compliance['red_lean_score'] != 'TBD') & (
            df_compliance['red_lean_score'] != 'PMD')]
        red_lean_list = np.array(
            df_comparison['red_lean_score'].astype(float).to_list())
        red_lean_list_of_lists = [red_lean_list]*len(compliance_tups)
        yellow_lean_list = np.array(
            df_comparison['yellow_lean_score'].astype(float).to_list())
        yellow_lean_list_of_lists = [yellow_lean_list]*len(compliance_tups)

        with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
            results = executor.map(fac_contam_score_percentile, compliance_tups,
                                   red_lean_list_of_lists, yellow_lean_list_of_lists)
            end_results_creation = time.perf_counter()
            print(f'Results creation: {end_results_creation-contam_start}')
            for result in results:
                compliance_scores_percentiles_list.append(result)
            end_results_iteration = time.perf_counter()
            print(
                f'Results iteration: {end_results_iteration-end_results_creation}')
            print(
                f'Time so far (fac_id/contam_id percentile loop {contam_list.index(contam) + 1}): {end_results_iteration - start}')

    df_compliance_scores_percentiles = pd.DataFrame(
        compliance_scores_percentiles_list, columns=compliance_scores_percentiles_columns)
    start_write_to_sql = time.perf_counter()
    df_compliance_scores_percentiles = df_compliance_scores_percentiles.astype(
        {'fac_id': int, 'contam_id': int, 'red_lean_score': str, 'yellow_lean_score': str, 'red_lean_percentile': str, 'yellow_lean_percentile': str, 'target_timeline': str, 'num_time_segments': str, 'num_track_switches': str})
    df_compliance_scores_percentiles = pd.merge(
        df_facilities, df_compliance_scores_percentiles, on="fac_id", how="inner")
    conn = vdl.sql_query_conn()
    df_compliance_scores_percentiles.to_sql(
        'score_and_percentile_fac_contam', conn, if_exists='replace', index=False)
    conn.close()
    end_write_to_sql = time.perf_counter()
    vdl.create_index('score_and_percentile_fac_contam',
                     fac_id='ASC', ws_id='ASC', contam_id='ASC')
    fac_id_contam_id_finish = time.perf_counter()
    print(
        f'Total time to finish fac-contam percentile loop {contam_list.index(contam) + 1} of {len(contam_list)}: {fac_id_contam_id_finish - start}')

    # Begin generating average of facility scores in order to determine the overall percentile for the facility
    start_average_facs = time.perf_counter()
    conn = vdl.sql_query_conn()
    df_fac_percentiles = pd.read_sql_query(
        f"SELECT * from score_and_percentile_fac_contam", conn)
    conn.close()
    fac_list = df_fac_percentiles['fac_id'].unique(
    ).tolist()  # list of all unique facilities
    dict_percentile_averages = {'fac_id': [], 'ave_red_lean_score': [
    ], 'ave_yellow_lean_score': [], 'ave_target_timeline': [], 'ave_method_priority_level': [], 'ave_num_time_segments': [], 'ave_num_track_switches': []}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(ave_fac_score, fac_list)
        end_results_creation = time.perf_counter()
        print(
            f'Results creation for average facility scores: {end_results_creation-start_average_facs}')
        for result in results:
            dict_percentile_averages['fac_id'].append(result[0])
            dict_percentile_averages['ave_red_lean_score'].append(
                result[1])
            dict_percentile_averages['ave_yellow_lean_score'].append(
                result[2])
            dict_percentile_averages['ave_target_timeline'].append(result[3])
            dict_percentile_averages['ave_method_priority_level'].append(
                result[4])
            dict_percentile_averages['ave_num_time_segments'].append(result[5])
            dict_percentile_averages['ave_num_track_switches'].append(
                result[6])
        df_percentile_averages = pd.DataFrame(dict_percentile_averages)
        end_results_iteration = time.perf_counter()
        print(
            f'Results iteration for average facility scores: {end_results_iteration-end_results_creation}')
    df_percentile_averages = pd.DataFrame(dict_percentile_averages)
    df_percentile_averages = df_percentile_averages.astype(
        {'fac_id': int, 'ave_red_lean_score': str, 'ave_yellow_lean_score': str, 'ave_target_timeline': str, 'ave_method_priority_level': str, 'ave_num_time_segments': str, 'ave_num_track_switches': str})
    df_percentile_averages = pd.merge(
        df_facilities, df_percentile_averages, on="fac_id", how="inner")
    conn = vdl.sql_query_conn()
    df_percentile_averages.to_sql(
        'score_and_percentile_ave_fac', conn, if_exists='replace', index=False)
    conn.close()
    ave_fac_score_finish = time.perf_counter()
    print(
        f'Total time to finish average facility scores: {ave_fac_score_finish - start_average_facs}')

    conn = vdl.sql_query_conn()
    df_compliance = pd.read_sql_query(
        f"SELECT * from score_and_percentile_ave_fac", conn)
    conn.close()
    compliance_tups = [tuple(x) for x in df_compliance.to_numpy()]
    df_comparison = df_compliance[(df_compliance['ave_red_lean_score'] != 'TBD') & (
        df_compliance['ave_red_lean_score'] != 'PMD')]
    red_lean_list = np.array(
        df_comparison['ave_red_lean_score'].astype(float).to_list())
    red_lean_list_of_lists = [red_lean_list]*len(compliance_tups)
    yellow_lean_list = np.array(
        df_comparison['ave_yellow_lean_score'].astype(float).to_list())
    yellow_lean_list_of_lists = [yellow_lean_list]*len(compliance_tups)
    compliance_scores_percentiles_list = []
    compliance_scores_percentiles_columns = [
        'fac_id', 'ws_id', 'ave_red_lean_score', 'ave_yellow_lean_score', 'ave_score_red_lean_percentile', 'ave_score_yellow_lean_percentile', 'ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches']
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(percentile_of_ave_fac_score, compliance_tups,
                               red_lean_list_of_lists, yellow_lean_list_of_lists)
        end_results_creation = time.perf_counter()
        print(f'Results creation: {end_results_creation-ave_fac_score_finish}')
        for result in results:
            compliance_scores_percentiles_list.append(result)
        end_results_iteration = time.perf_counter()
        print(
            f'Results iteration: {end_results_iteration-end_results_creation}')
        print(
            f'Time so far fac percentile: {end_results_iteration - start}')

    df_ave_fac_scores_percentiles = pd.DataFrame(
        compliance_scores_percentiles_list, columns=compliance_scores_percentiles_columns)
    df_ave_fac_scores_percentiles = df_ave_fac_scores_percentiles.astype(
        {'fac_id': int, 'ws_id': int, 'ave_red_lean_score': str, 'ave_yellow_lean_score': str, 'ave_score_red_lean_percentile': str, 'ave_score_yellow_lean_percentile': str, 'ave_target_timeline': str, 'ave_method_priority_level': str, 'ave_num_time_segments': str, 'ave_num_track_switches': str})

    conn = vdl.sql_query_conn()
    df_ave_fac_scores_percentiles.to_sql(
        'score_and_percentile_ave_fac', conn, if_exists='replace', index=False)
    conn.close()
    vdl.create_index('score_and_percentile_ave_fac', fac_id='ASC', ws_id='ASC')
    ave_fac_percentile_finish = time.perf_counter()
    print(
        f'Total time to finish average facility percentile: {ave_fac_percentile_finish - ave_fac_score_finish}')

    # facility overage calculation:
    conn = vdl.sql_query_conn()
    fac_ws_list = pd.read_sql_query(
        f"SELECT * from 'score_and_percentile_ave_fac'", conn).values.tolist()
    conn.close()

    fac_overage_tally = []
    start_overage_tally = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(facility_overage_counter, fac_ws_list)
        end_results_creation = time.perf_counter()
        for result in results:
            fac_overage_tally.append(result)

    df_fac_overage = pd.DataFrame(fac_overage_tally, columns=[
        'fac_id', 'ws_id', 'overage_total', 'overage_rate'])

    df_fac_overage[["fac_id", "ws_id"]] = df_fac_overage[[
        "fac_id", "ws_id"]].astype(str).astype(int)
    df_fac_overage[["overage_total", "overage_rate"]] = df_fac_overage[[
        "overage_total", "overage_rate"]].astype(str)
    conn = vdl.sql_query_conn()
    df_fac_overage.to_sql('overage_count_and_percentile_fac',
                          conn, if_exists='replace', index=False)
    conn.close()
    finish_overage_tally = time.perf_counter()
    print(df_fac_overage)
    print(finish_overage_tally - start_overage_tally)

    facility_overage_rates_list = df_fac_overage['overage_rate'].values.tolist(
    )
    filtered_values = ['PMD', 'TBD']
    filtered_fac_overage_rates_list = []

    for fac in facility_overage_rates_list:
        if fac not in filtered_values:
            filtered_fac_overage_rates_list.append(float(fac))

    print(len(facility_overage_rates_list))
    print(len(filtered_fac_overage_rates_list))

    fac_overage_list = df_fac_overage.values.tolist()
    fac_overage_percentile_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(facility_overage_percentile, fac_overage_list, [
            filtered_fac_overage_rates_list]*len(fac_overage_list))
        end_results_creation = time.perf_counter()
        for result in results:
            fac_overage_percentile_list.append(result)
    df_fac_overage_percentile = pd.DataFrame(fac_overage_percentile_list, columns=[
        'fac_id', 'ws_id', 'overage_total', 'overage_rate', 'overage_percentile'])
    df_fac_overage_percentile[["fac_id", "ws_id"]] = df_fac_overage[[
        "fac_id", "ws_id"]].astype(str).astype(int)
    df_fac_overage_percentile[["overage_total", "overage_rate", "overage_percentile"]] = df_fac_overage_percentile[[
        "overage_total", "overage_rate", "overage_percentile"]].astype(str)
    conn = vdl.sql_query_conn()
    df_fac_overage_percentile.to_sql('overage_count_and_percentile_fac',
                                     conn, if_exists='replace', index=False)
    conn.close()
    vdl.create_index('overage_count_and_percentile_fac',
                     fac_id='ASC', ws_id='ASC')

    print("FINISH ")

    finish = time.perf_counter()
    print(f'Seconds: {finish - start}')
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

# Jae Test 12/18/22
# 1296.8722965999914
