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


def ws_contam_score(ws_id, contam_list):
    ws_id = ws_id
    all_ws_contam_results = []
    conn = vdl.sql_query_conn()
    test_df = pd.read_sql_query(
        f"SELECT * from 'score_and_percentile_fac_contam' where ws_id={ws_id}", conn)
    conn.close()
    if len(test_df) == 0:
        red_lean_ave_score = 'NA'
        yellow_lean_ave_score = 'NA'
        for contam in contam_list:
            all_ws_contam_results.append(
                [ws_id, contam, red_lean_ave_score, yellow_lean_ave_score])
        return all_ws_contam_results
    for contam in contam_list:
        ws_facs_contam = test_df[(test_df['contam_id'] == int(contam))]
        num_tbd = len(
            ws_facs_contam[ws_facs_contam['red_lean_score'] == 'TBD'])
        num_pmd = len(
            ws_facs_contam[ws_facs_contam['red_lean_score'] == 'PMD'])
        if num_tbd + num_pmd == len(ws_facs_contam):
            if num_tbd > 0:
                red_lean_ave_score = 'TBD'
                yellow_lean_ave_score = 'TBD'
            else:
                red_lean_ave_score = 'PMD'
                yellow_lean_ave_score = 'PMD'
        else:
            df_score_calc = test_df[(test_df['contam_id'] == int(contam)) & (
                test_df['red_lean_score'] != 'TBD') & (test_df['red_lean_score'] != 'PMD')]
            red_lean_scores = [float(i)
                               for i in df_score_calc['red_lean_score'].tolist()]
            yellow_lean_scores = [
                float(i) for i in df_score_calc['yellow_lean_score'].tolist()]
            red_lean_ave_score = sum(red_lean_scores)/len(red_lean_scores)
            yellow_lean_ave_score = sum(
                yellow_lean_scores)/len(yellow_lean_scores)
        all_ws_contam_results.append(
            [ws_id, contam, red_lean_ave_score, yellow_lean_ave_score])

    return all_ws_contam_results


# conn = vdl.sql_query_conn()
# df_timeline = pd.read_sql_query(
#     f"SELECT fac_id, contam_id from 'user_period_timeline'", conn)
# conn.close()
# contam_list = df_timeline['contam_id'].unique().tolist()
# pr = cProfile.Profile()
# pr.enable()
# # print(ws_contam_score(1848, contam_list))
# # print(ws_contam_score(8160, contam_list))
# print(ws_contam_score(2159, contam_list))

# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
# raise ValueError


def ws_contam_score_percentile(compliance_tuple, red_lean_list, yellow_lean_list):
    ws_id = compliance_tuple[0]
    contam_id = compliance_tuple[1]
    red_lean_score = compliance_tuple[2]
    yellow_lean_score = compliance_tuple[3]
    if red_lean_score == 'TBD' or red_lean_score == 'PMD':
        red_percentile = red_lean_score
        yellow_percentile = red_lean_score
    elif red_lean_score == 'NA':
        red_percentile = 'NA'
        yellow_percentile = 'NA'
    else:
        red_calc_abs_comp = (np.abs(red_lean_list) <= float(
            red_lean_score))
        yellow_calc_abs_comp = (np.abs(yellow_lean_list)
                                < float(red_lean_score))
        red_len_list = float(len(red_lean_list))
        yellow_len_list = float(len(yellow_lean_list))
        red_calc_sub = red_calc_abs_comp / red_len_list
        yellow_calc_sub = yellow_calc_abs_comp / yellow_len_list
        red_percentile = math.floor(sum(red_calc_sub)*100)
        yellow_percentile = math.floor(sum(yellow_calc_sub)*100)
    return [ws_id, contam_id, red_lean_score, yellow_lean_score, red_percentile, yellow_percentile]


# conn = vdl.sql_query_conn()
# df_compliance = pd.read_sql_query(
#     f"SELECT * from score_and_percentile_ws_contam where contam_id = {str(101)}", conn)
# conn.close()
# df_comparison = df_compliance[(df_compliance['red_lean_ave_score'] != 'TBD') & (df_compliance['red_lean_ave_score'] != 'PMD') & (
#     df_compliance['red_lean_ave_score'] != 'NA')]
# compliance_tups = [tuple(x) for x in df_compliance.to_numpy()]
# red_lean_list = np.array(
#     df_comparison['red_lean_ave_score'].astype(float).to_list())
# yellow_lean_list = np.array(
#     df_comparison['yellow_lean_ave_score'].astype(float).to_list())
# print(ws_contam_score_percentile(compliance_tups[7],
#                                  red_lean_list, yellow_lean_list))
# raise ValueError


def ave_ws_score(ws_id):
    water_system = int(ws_id)
    conn = vdl.sql_query_conn()
    df_compliance_percentiles = pd.read_sql_query(
        f"SELECT * from score_and_percentile_ws_contam WHERE ws_id= {water_system}", conn)
    conn.close()
    if (df_compliance_percentiles['red_lean_ave_score'] == 'NA').all():
        ave_red_lean_score = 'NA'
        ave_yellow_lean_score = 'NA'
        average_percentiles = [water_system,
                               ave_red_lean_score, ave_yellow_lean_score]
        return average_percentiles
    red_lean_score_list = df_compliance_percentiles[(df_compliance_percentiles['ws_id'] == water_system) & (
        df_compliance_percentiles['red_lean_ave_score'] != 'TBD') & (df_compliance_percentiles['red_lean_ave_score'] != 'PMD')]['red_lean_ave_score'].tolist()
    red_lean_score_list = [float(i) for i in red_lean_score_list]
    yellow_lean_score_list = df_compliance_percentiles[(df_compliance_percentiles['ws_id'] == water_system) & (df_compliance_percentiles['yellow_lean_ave_score'] != 'TBD') & (
        df_compliance_percentiles['yellow_lean_ave_score'] != 'PMD')]['yellow_lean_ave_score'].tolist()
    yellow_lean_score_list = [float(i) for i in yellow_lean_score_list]
    if len(red_lean_score_list) == 0:
        num_tbd = len(
            df_compliance_percentiles[df_compliance_percentiles['red_lean_ave_score'] == 'TBD']['red_lean_ave_score'])
        if num_tbd > 0:
            average_percentiles = [water_system, 'TBD', 'TBD']
        else:
            average_percentiles = [water_system, 'PMD', 'PMD']
    else:
        ave_red_lean_score = sum(
            red_lean_score_list)/len(red_lean_score_list)
        ave_yellow_lean_score = sum(
            yellow_lean_score_list)/len(yellow_lean_score_list)
        average_percentiles = [
            water_system, ave_red_lean_score, ave_yellow_lean_score]

    return average_percentiles


# print(ave_ws_score(4488))
# print(ave_ws_score(5656))
# raise ValueError

def water_system_average_timeline_characteristics(ws_id):
    query = f'''SELECT * from score_and_percentile_ave_fac WHERE ws_id = {ws_id}'''
    conn = vdl.sql_query_conn()
    df_method_priority = pd.read_sql_query(query, conn)
    conn.close()
    df_method_priority = df_method_priority[(df_method_priority['ave_target_timeline'] != 'TBD') & (
        df_method_priority['ave_target_timeline'] != 'PMD')]
    average_target_timeline = df_method_priority['ave_target_timeline'].mode()[
        0]
    average_method_priority_level = df_method_priority['ave_method_priority_level'].astype(
        float).mean()
    average_num_time_segments = df_method_priority['ave_num_time_segments'].astype(
        float).mean()
    average_num_track_switches = df_method_priority['ave_num_track_switches'].astype(
        float).mean()

    return [average_target_timeline, average_method_priority_level, average_num_time_segments, average_num_track_switches]


# print(water_system_average_timeline_characteristics(43))
# raise ValueError


def percentile_of_ave_ws_score(compliance_tuple, red_lean_list, yellow_lean_list):
    ws_id = compliance_tuple[0]
    ave_red_lean_score = compliance_tuple[1]
    ave_yellow_lean_score = compliance_tuple[2]
    if ave_red_lean_score == 'NA':
        red_percentile = 'NA'
        yellow_percentile = 'NA'
    elif ave_red_lean_score == 'TBD':
        red_percentile = 'TBD'
        yellow_percentile = 'TBD'
    elif ave_red_lean_score == 'PMD':
        red_percentile = 'PMD'
        yellow_percentile = 'PMD'
    else:
        red_calc_abs_comp = (np.abs(red_lean_list) < float(ave_red_lean_score))
        yellow_calc_abs_comp = (np.abs(yellow_lean_list)
                                < float(ave_red_lean_score))
        red_len_list = float(len(red_lean_list))
        yellow_len_list = float(len(yellow_lean_list))
        red_calc_sub = red_calc_abs_comp / red_len_list
        yellow_calc_sub = yellow_calc_abs_comp / yellow_len_list
        red_percentile = math.floor(sum(red_calc_sub)*100)
        yellow_percentile = math.floor(sum(yellow_calc_sub)*100)
    if red_percentile in ['NA', 'TBD', 'PMD']:
        ave_target_timeline = red_percentile
        ave_method_priority_level = red_percentile
        ave_num_time_segments = red_percentile
        ave_num_track_switches = red_percentile
    else:
        ws_tl_characteristics = water_system_average_timeline_characteristics(
            ws_id)
        ave_target_timeline = ws_tl_characteristics[0]
        ave_method_priority_level = ws_tl_characteristics[1]
        ave_num_time_segments = ws_tl_characteristics[2]
        ave_num_track_switches = ws_tl_characteristics[3]

    return [ws_id, ave_red_lean_score, ave_yellow_lean_score, red_percentile, yellow_percentile, ave_target_timeline, ave_method_priority_level, ave_num_time_segments, ave_num_track_switches]


# conn = vdl.sql_query_conn()
# df_compliance = pd.read_sql_query(
#     f"SELECT * from score_and_percentile_ave_ws", conn)
# conn.close()
# df_comparison = df_compliance[(df_compliance['ave_red_lean_score'] != 'TBD') & (
#     df_compliance['ave_red_lean_score'] != 'NA')]
# red_lean_list = np.array(
#     df_comparison['ave_red_lean_score'].astype(float).to_list())
# yellow_lean_list = np.array(
#     df_comparison['ave_yellow_lean_score'].astype(float).to_list())
# print(percentile_of_ave_ws_score((8103, 'NA', 'NA'),
#                                  red_lean_list, yellow_lean_list))
# raise ValueError


def ws_overage_counter(ws_compliance):
    ws_id = ws_compliance[0]
    ave_red_lean_score = str(ws_compliance[1])
    if ave_red_lean_score == 'PMD' or ave_red_lean_score == 'TBD' or ave_red_lean_score == 'NA':
        return [ws_id, ave_red_lean_score, ave_red_lean_score]
    conn = vdl.sql_query_conn()
    df_overage_for_ws_facs = pd.read_sql_query(
        f"SELECT * from 'overage_count_and_percentile_fac' WHERE ws_id = {ws_id}", conn)
    conn.close()

    df_overage_for_ws_facs = df_overage_for_ws_facs[(df_overage_for_ws_facs['overage_total'] != 'PMD') & (
        df_overage_for_ws_facs['overage_total'] != 'TBD') & (df_overage_for_ws_facs['overage_total'] != 'NA')]

    df_overage_for_ws_facs[["overage_total", "overage_rate"]] = df_overage_for_ws_facs[[
        "overage_total", "overage_rate"]].astype(float)

    ave_overage_total = df_overage_for_ws_facs['overage_total'].mean()
    ave_overage_rate = df_overage_for_ws_facs['overage_rate'].mean()
    return [ws_id, ave_overage_total, ave_overage_rate]


# ws_overage_counter([320, '326.32459316413', '322.016888890553', '82', '85'])
# raise ValueError

def ws_overage_percentile(ws_overage_list, all_ws_overage_rate_list):
    ws_id = ws_overage_list[0]
    ws_ave_overage_total = ws_overage_list[1]
    ws_ave_overage_rate = ws_overage_list[2]
    if ws_ave_overage_rate == 'PMD' or ws_ave_overage_rate == 'TBD' or ws_ave_overage_rate == 'NA':
        return [ws_id, ws_ave_overage_total, ws_ave_overage_rate, ws_ave_overage_rate]
    try:
        overage_calc_abs_comp = (np.abs(all_ws_overage_rate_list)
                                 < float(ws_ave_overage_rate))
        overage_len_list = float(len(all_ws_overage_rate_list))
        overage_calc_sub = overage_calc_abs_comp / overage_len_list
        ws_overage_percentile = (math.floor(
            sum(overage_calc_sub)*100)-99) * (-1)
    except:
        print(ws_overage_list)
        print(ws_ave_overage_rate)
        print(type(ws_ave_overage_rate))
        print(type(all_ws_overage_rate_list[0]))
        raise ValueError
    return [ws_id, ws_ave_overage_total, ws_ave_overage_rate, ws_overage_percentile]


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()
    prev_start = start
    conn = vdl.sql_query_conn()
    ws_id_list = pd.read_sql_query(
        f"SELECT * from water_system_primary", conn)['id'].unique().tolist()
    df_timeline = pd.read_sql_query(
        f"SELECT contam_id from 'user_period_timeline'", conn)
    conn.close()
    contam_list = df_timeline['contam_id'].unique(
    ).tolist()  # list of all unique contam_id
    contam_list_of_list = [contam_list]*len(ws_id_list)
    df_timeline = pd.DataFrame()  # this is to free up memory
    write_table_name = "score_and_percentile_ws_contam"

    print('Ready to start')
    timeline_start = time.perf_counter()
    df_ws_score = pd.DataFrame()
    sublist_start = time.perf_counter()
    ws_scores_columns = ['ws_id', 'contam_id',
                         'red_lean_ave_score', 'yellow_lean_ave_score']
    ws_scores_list_of_lists = []
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(ws_contam_score,
                               ws_id_list, contam_list_of_list)
        end_results_creation = time.perf_counter()
        print(f'Results creation: {end_results_creation-sublist_start}')
        for result in results:
            ws_scores_list_of_lists.extend(result)

        df_ws_score = pd.DataFrame(
            ws_scores_list_of_lists, columns=ws_scores_columns)
        end_results_iteration = time.perf_counter()
        print(
            f'Results iteration: {end_results_iteration-end_results_creation}')

        start_write_to_sql = time.perf_counter()
        conn = vdl.sql_query_conn()
        df_ws_score.to_sql(write_table_name, conn,
                           if_exists='replace', index=False)
        conn.close()

        df_ws_score = pd.DataFrame()
        end_write_to_sql = time.perf_counter()
        prev_start = end_write_to_sql
        print(
            f'Time so far in water system average score determination: {end_write_to_sql - start}')

    ws_scores_percentiles_list = []
    ws_scores_percentiles_columns = [
        'ws_id', 'contam_id', 'red_lean_ave_score', 'yellow_lean_ave_score', 'red_lean_percentile', 'yellow_lean_percentile']
    for contam in contam_list:
        contam_start = time.perf_counter()
        conn = vdl.sql_query_conn()
        df_all_ws_scores = pd.read_sql_query(
            f"SELECT * from score_and_percentile_ws_contam where contam_id = {str(contam)}", conn)
        conn.close()
        compliance_tups = [tuple(x) for x in df_all_ws_scores.to_numpy()]
        df_comparison = df_all_ws_scores[(df_all_ws_scores['red_lean_ave_score'] != 'TBD') & (
            df_all_ws_scores['red_lean_ave_score'] != 'NA') & (df_all_ws_scores['red_lean_ave_score'] != 'PMD')]
        red_lean_list = np.array(
            df_comparison['red_lean_ave_score'].astype(float).to_list())
        red_lean_list_of_lists = [red_lean_list]*len(compliance_tups)
        yellow_lean_list = np.array(
            df_comparison['yellow_lean_ave_score'].astype(float).to_list())
        yellow_lean_list_of_lists = [yellow_lean_list]*len(compliance_tups)

        with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
            results = executor.map(ws_contam_score_percentile, compliance_tups,
                                   red_lean_list_of_lists, yellow_lean_list_of_lists)
            end_results_creation = time.perf_counter()
            print(f'Results creation: {end_results_creation-contam_start}')
            for result in results:
                ws_scores_percentiles_list.append(result)
                end_results_iteration = time.perf_counter()
            print(
                f'Results iteration: {end_results_iteration-end_results_creation}')
            print(
                f'Time so far (ws_id/contam_id percentile loop {contam_list.index(contam) + 1}): {end_results_iteration - start}')

    df_compliance_scores_percentiles = pd.DataFrame(
        ws_scores_percentiles_list, columns=ws_scores_percentiles_columns)
    start_write_to_sql = time.perf_counter()
    conn = vdl.sql_query_conn()
    df_compliance_scores_percentiles.to_sql(
        'score_and_percentile_ws_contam', conn, if_exists='replace', index=False)
    conn.close()
    end_write_to_sql = time.perf_counter()
    vdl.create_index('score_and_percentile_ws_contam',
                     ws_id='ASC', contam_id='ASC')
    ws_id_contam_id_finish = time.perf_counter()
    print(
        f'Total time to finish ws-contam percentile: {ws_id_contam_id_finish - start}')

    start_average_ws = time.perf_counter()
    conn = vdl.sql_query_conn()
    df_ws_percentiles = pd.read_sql_query(
        f"SELECT * from score_and_percentile_ws_contam", conn)
    conn.close()
    ws_id_list = df_ws_percentiles['ws_id'].unique(
    ).tolist()  # list of all unique water systems
    dict_percentile_averages = {'ws_id': [], 'ave_red_lean_score': [
    ], 'ave_yellow_lean_score': []}
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(ave_ws_score, ws_id_list)
        end_results_creation = time.perf_counter()
        print(
            f'Results creation for average water system scores: {end_results_creation-start_average_ws}')
        for result in results:
            dict_percentile_averages['ws_id'].append(result[0])
            dict_percentile_averages['ave_red_lean_score'].append(
                result[1])
            dict_percentile_averages['ave_yellow_lean_score'].append(
                result[2])
        df_percentile_averages = pd.DataFrame(dict_percentile_averages)
        end_results_iteration = time.perf_counter()
        print(
            f'Results iteration for average water system scores: {end_results_iteration-end_results_creation}')
    df_percentile_averages = pd.DataFrame(dict_percentile_averages)
    conn = vdl.sql_query_conn()
    df_percentile_averages.to_sql(
        'score_and_percentile_ave_ws', conn, if_exists='replace', index=False)
    conn.close()
    ave_ws_score_finish = time.perf_counter()
    print(
        f'Total time to finish average water system scores: {ave_ws_score_finish - start_average_ws}')

    conn = vdl.sql_query_conn()
    df_compliance = pd.read_sql_query(
        f"SELECT * from score_and_percentile_ave_ws", conn)
    conn.close()
    compliance_tups = [tuple(x) for x in df_compliance.to_numpy()]
    df_comparison = df_compliance[(df_compliance['ave_red_lean_score'] != 'TBD') & (
        df_compliance['ave_red_lean_score'] != 'NA') & (df_compliance['ave_red_lean_score'] != 'PMD')]
    red_lean_list = np.array(
        df_comparison['ave_red_lean_score'].astype(float).to_list())
    red_lean_list_of_lists = [red_lean_list]*len(compliance_tups)
    yellow_lean_list = np.array(
        df_comparison['ave_yellow_lean_score'].astype(float).to_list())
    yellow_lean_list_of_lists = [yellow_lean_list]*len(compliance_tups)
    compliance_scores_percentiles_list = []
    compliance_scores_percentiles_columns = [
        'ws_id', 'ave_red_lean_score', 'ave_yellow_lean_score', 'ave_score_red_lean_percentile', 'ave_score_yellow_lean_percentile', 'ave_target_timeline', 'ave_method_priority_level', 'ave_num_time_segments', 'ave_num_track_switches']
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(percentile_of_ave_ws_score, compliance_tups,
                               red_lean_list_of_lists, yellow_lean_list_of_lists)
        end_results_creation = time.perf_counter()
        print(f'Results creation: {end_results_creation-contam_start}')
        for result in results:
            compliance_scores_percentiles_list.append(result)
        end_results_iteration = time.perf_counter()
        print(
            f'Results iteration: {end_results_iteration-end_results_creation}')
        print(
            f'Time so far ws percentile: {end_results_iteration - start}')

    df_ave_ws_scores_percentiles = pd.DataFrame(
        compliance_scores_percentiles_list, columns=compliance_scores_percentiles_columns)
    conn = vdl.sql_query_conn()
    df_ave_ws_scores_percentiles.to_sql(
        'score_and_percentile_ave_ws', conn, if_exists='replace', index=False)
    conn.close()
    vdl.create_index('score_and_percentile_ave_ws', ws_id='ASC')
    ave_ws_percentile_finish = time.perf_counter()
    print(
        f'Total time to finish average water system percentile: {ave_ws_percentile_finish - ave_ws_score_finish}')

    # Including a count of number of GW and SW facs in water systems.
    # This must be done sometime after step 6 to exclude treated water systems or any water systems that are NC or transient.
    conn = vdl.sql_query_conn()
    df_wsp = pd.read_sql_query(
        f"SELECT * from water_system_primary", conn)
    df_facs_with_scores = pd.read_sql_query(
        f"SELECT * from score_and_percentile_ave_fac", conn)
    df_facs = pd.read_sql_query(
        f"SELECT * from facilities", conn)
    conn.close()

    df_facility_ref = pd.merge(
        df_facs, df_facs_with_scores, left_on='id', right_on='fac_id', how='right')[['id', 'ws_id_x', 'activity_xldate', 'facility_type', 'min_xldate', 'max_xldate', 'num_unique_contams']]
    df_facility_ref.rename(columns={'ws_id_x': 'ws_id'}, inplace=True)

    new_items = ['number_gw', 'number_sw', 'ave_act_xldate', 'ave_min_xldate',
                 'ave_max_xldate', 'ave_range_xldate', 'ave_num_unique_contams']
    if any(item in df_wsp.columns.to_list() for item in new_items):
        df_wsp.drop(['number_gw', 'number_sw', 'ave_act_xldate', 'ave_min_xldate',
                    'ave_max_xldate', 'ave_range_xldate', 'ave_num_unique_contams'], axis=1, inplace=True)
    ws_list = []
    for i, j in df_wsp.iterrows():
        ws_facs = df_facility_ref[df_facility_ref['ws_id'] == j['id']]
        num_gw = 0
        num_sw = 0
        act_xldate_list = []
        min_xldate_list = []
        max_xldate_list = []
        num_unique_contams_list = []
        counts_list_dict = {3: act_xldate_list, 4: min_xldate_list,
                            5: max_xldate_list, 7: num_unique_contams_list}
        vals_list_dict = {3: 'activity_xldate', 4: 'min_xldate',
                          5: 'max_xldate', 7: 'num_unique_contams'}
        ave_act_xldate = 'NA'
        ave_min_xldate = 'NA'
        ave_max_xldate = 'NA'
        ave_range_xldate = 'NA'
        ave_num_unique_contams = 'NA'
        output_list_dict = {1: num_gw, 2: num_sw, 3: ave_act_xldate, 4: ave_min_xldate,
                            5: ave_max_xldate, 6: ave_range_xldate, 7: ave_num_unique_contams}
        updated_ws = j.to_list()
        if len(ws_facs) != 0:
            for k, l in ws_facs.iterrows():
                fac_type = vdl.water_system_type(l['facility_type'])
                if fac_type == 'pswt':
                    if 'GW' in j['primary_source_water_type']:
                        fac_type = 'GW'
                    else:
                        fac_type = 'SW'
                if fac_type == 'GW':
                    output_list_dict[1] += 1
                elif fac_type == 'SW':
                    output_list_dict[2] += 1
                else:
                    print(j)
                    raise ValueError
                for i in [3, 4, 5, 7]:
                    if l[vals_list_dict[i]] != None:
                        counts_list_dict[i].append(
                            int(l[vals_list_dict[i]]))
            for i in [3, 4, 5, 7]:
                if len(counts_list_dict[i]) != 0:
                    output_list_dict[i] = int(round(
                        sum(counts_list_dict[i])/len(counts_list_dict[i]), 0))
            if output_list_dict[4] != 'NA':
                output_list_dict[6] = round(
                    output_list_dict[5] - output_list_dict[4], 0)
            for i in range(1, 8):
                updated_ws.append(output_list_dict[i])
        else:
            for i in range(1, 8):
                updated_ws.append(output_list_dict[i])
        ws_list.append(updated_ws)

    df_updated_ws = pd.DataFrame(ws_list, columns=['id', 'water_system_number', 'water_system_name', 'ddw_name', 'ptype', 'pserved', 'ccount', 'service_area_code', 'distribution_system_class',
                                                   'max_treatment_plant_class', 'type', 'primary_source_water_type', 'url', 'ws_link_suffix', 'number_gw', 'number_sw', 'ave_act_xldate',
                                                   'ave_min_xldate', 'ave_max_xldate', 'ave_range_xldate', 'ave_num_unique_contams'])
    print(df_updated_ws)
    conn = vdl.sql_query_conn()
    df_updated_ws.to_sql(
        'water_system_primary', conn, if_exists='replace', index=False)
    conn.close()

    # water system overage calculation:
    conn = vdl.sql_query_conn()
    ws_compliance_list = pd.read_sql_query(
        f"SELECT * from 'score_and_percentile_ave_ws'", conn).values.tolist()
    conn.close()
    ws_overage_tally = []
    start_overage_tally = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(ws_overage_counter, ws_compliance_list)
        end_results_creation = time.perf_counter()
        for result in results:
            ws_overage_tally.append(result)
    df_ws_overage = pd.DataFrame(ws_overage_tally, columns=[
        'ws_id', 'ave_overage_total', 'ave_overage_rate'])

    df_ws_overage[["ws_id"]] = df_ws_overage[["ws_id"]].astype(str).astype(int)
    df_ws_overage[["ave_overage_total", "ave_overage_rate"]] = df_ws_overage[[
        "ave_overage_total", "ave_overage_rate"]].astype(str)
    conn = vdl.sql_query_conn()
    df_ws_overage.to_sql('overage_count_and_percentile_ws',
                         conn, if_exists='replace', index=False)
    conn.close()
    finish_overage_tally = time.perf_counter()
    print(df_ws_overage)
    print(finish_overage_tally - start_overage_tally)

    ws_overage_rates_list = df_ws_overage['ave_overage_rate'].values.tolist(
    )
    filtered_values = ['PMD', 'TBD', 'NA']
    filtered_ws_overage_rates_list = []

    for ws in ws_overage_rates_list:
        if ws not in filtered_values:
            filtered_ws_overage_rates_list.append(float(ws))
    print(len(ws_overage_rates_list))
    print(len(filtered_ws_overage_rates_list))

    ws_overage_list = df_ws_overage.values.tolist()
    ws_overage_percentile_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(ws_overage_percentile, ws_overage_list, [
                               filtered_ws_overage_rates_list]*len(ws_overage_list))
        end_results_creation = time.perf_counter()
        for result in results:
            ws_overage_percentile_list.append(result)
    df_ws_overage_percentile = pd.DataFrame(ws_overage_percentile_list, columns=[
                                            'ws_id', 'ave_overage_total', 'ave_overage_rate', 'overage_percentile'])
    df_ws_overage_percentile[["ws_id"]] = df_ws_overage[[
        "ws_id"]].astype(str).astype(int)
    df_ws_overage_percentile[["ave_overage_total", "ave_overage_rate", "overage_percentile"]] = df_ws_overage_percentile[[
        "ave_overage_total", "ave_overage_rate", "overage_percentile"]].astype(str)
    conn = vdl.sql_query_conn()
    df_ws_overage_percentile.to_sql('overage_count_and_percentile_ws',
                                    conn, if_exists='replace', index=False)
    conn.close()
    vdl.create_index('overage_count_and_percentile_ws', ws_id='ASC')

    print("FINISH ")

    finish = time.perf_counter()
    print(f'Seconds: {finish - start}')
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

# Jae 12/18/222
# 410.4736317000061
