from ipaddress import AddressValueError
from multiprocessing.sharedctypes import Value
from pickletools import read_decimalnl_long
from threading import currentThread
from weakref import ref
import pandas as pd
import numpy as np
import time
import concurrent.futures
import cProfile
import pstats
import io
from pstats import SortKey
import math
import regex as re
import wdc_lib


def color_ratio_reg_contam(reg_contam):
    reg = reg_contam[0]
    contam = reg_contam[1]

    color_dictionary = {"RED": 0, "YELLOW": 0, "GREEN": 0, "BLACK": 0, "TBD": 0}
    # total_days = 0
    # green_days = 0
    # red_days = 0
    # yellow_days = 0

    query = f"""SELECT * FROM user_effective_timeline WHERE regulation= "{reg}" AND contam_id={contam}"""
    conn = wdc_lib.sql_query_conn()
    df_reg_contam = pd.read_sql_query(query, conn)
    conn.close()

    # print(df_reg_contam)

    for i, j in df_reg_contam.iterrows():
        color_dictionary[j["color"]] += j["end_date"] - j["start_date"]

    # print(color_dictionary)

    green_red_yellow_total_days = (
        color_dictionary["GREEN"] + color_dictionary["RED"] + color_dictionary["YELLOW"]
    )

    if green_red_yellow_total_days > 0:
        green_fraction = 0
        red_fraction = 0
        yellow_fraction = 0
        if color_dictionary["GREEN"] > 0:
            green_fraction = color_dictionary["GREEN"] / green_red_yellow_total_days
        if color_dictionary["RED"] > 0:
            red_fraction = color_dictionary["RED"] / green_red_yellow_total_days
        if color_dictionary["YELLOW"] > 0:
            yellow_fraction = color_dictionary["YELLOW"] / green_red_yellow_total_days
        return [reg, contam, green_fraction, red_fraction, yellow_fraction]
    else:
        return []


# test_time = time.perf_counter()
# print(color_ratio_reg_contam(["CA Title 22 subsec 64445", 100]))
# print(time.perf_counter() - test_time)


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()

    reg_query = "SELECT DISTINCT regulation FROM user_effective_timeline"
    contam_query = "SELECT DISTINCT contam_id FROM user_effective_timeline"

    conn = wdc_lib.sql_query_conn()
    reg_list = pd.read_sql_query(reg_query, conn)["regulation"].values.tolist()
    contam_list = pd.read_sql_query(contam_query, conn)["contam_id"].values.tolist()
    conn.close()

    reg_list.remove("NA")
    reg_list.remove("General practice")

    reg_contam_list = []

    for r in reg_list:
        for c in contam_list:
            reg_contam_list.append([r, c])

    color_ratio_columns = [
        "regulation",
        "contam_id",
        "green_fraction",
        "red_fraction",
        "yellow_fraction",
    ]
    color_ratio_data = []
    with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
        results = executor.map(color_ratio_reg_contam, reg_contam_list)
        end_results_creation = time.perf_counter()
        print(f"Results creation: {end_results_creation-start}")
        for result in results:
            color_ratio_data.append(result)

        color_ratio_df = pd.DataFrame(color_ratio_data, columns=color_ratio_columns)

        conn = wdc_lib.sql_query_conn()
        color_ratio_df.to_sql(
            "color_ratio_data", conn, if_exists="replace", index=False
        )
        conn.close()

        print("Total time:")
        print(time.perf_counter() - start)
        print(color_ratio_data)

    # prev_start = start

    # terminal_date_list = vdl.terminal_dates_to_review()

    # conn = vdl.sql_query_conn()
    # df_facilities = pd.read_sql_query(f"SELECT * from 'facilities'", conn)
    # df_contam = pd.read_sql_query(f"SELECT * from 'contam_info'", conn)
    # # Just for getting the unique facid/contamid combos that were actually reviewed
    # df_timeline = pd.read_sql_query(
    #     f"SELECT fac_id, contam_id from 'user_period_timeline' WHERE terminal_date = {terminal_date_list[-1][2]}",
    #     conn,
    # )
    # # df_timeline = pd.read_sql_query(
    # #     f"SELECT fac_id, contam_id from 'user_period_timeline'", conn
    # # )
    # conn.close()
    # list of all unique facilities
#     fac_list = df_timeline["fac_id"].unique().tolist()
#     contam_list = (
#         df_timeline["contam_id"].unique().tolist()
#     )  # list of all unique contam_id
#     df_timeline = pd.DataFrame()  # this is to free up memory

#     all_combos_list = []
#     for f in fac_list:
#         fac_activity_status_date = df_facilities[df_facilities["id"] == f][
#             "activity_xldate"
#         ].values[0]
#         for c in contam_list:
#             reg_effective_date = int(
#                 df_contam[df_contam["id"] == c]["reg_xldate"].values[0]
#             )
#             for t in terminal_date_list:
#                 if (
#                     int(t[2]) >= reg_effective_date
#                     and int(t[2]) >= fac_activity_status_date
#                 ):
#                     all_combos_list.append([f, c, t])

#     # all_combos_list = [[f, c] for f in fac_list for c in contam_list]
#     write_table_name = "score_and_percentile_fac_contam"
#     all_combos_sublists = []
#     for i in range(0, len(all_combos_list), 100000):
#         all_combos_sublists.append(all_combos_list[i : i + 100000])

#     print(f"Ready to start, time so far is: {time.perf_counter() - start}")
#     print(
#         f"Total of {len(all_combos_list)} fac_contam_td combos across {len(all_combos_sublists)} sublists"
#     )
#     timeline_start = time.perf_counter()
#     df_compliance = pd.DataFrame()
#     conn = vdl.sql_query_conn()
#     df_facilities = pd.read_sql_query(f"SELECT * from facilities", conn)[
#         ["id", "ws_id"]
#     ]
#     conn.close()
#     df_facilities.rename(columns={"id": "fac_id"}, inplace=True)

#     # print(f"Number of sublists: {len(all_combos_sublists)}")
#     # for sublist in all_combos_sublists:
#     #     print(f"Starting loop {all_combos_sublists.index(sublist)+1}")
#     #     sublist_start = time.perf_counter()
#     #     dict_compliance = {
#     #         "fac_id": [],
#     #         "contam_id": [],
#     #         "terminal_date": [],
#     #         "red_lean_score": [],
#     #         "yellow_lean_score": [],
#     #         "target_timeline": [],
#     #         "num_time_segments": [],
#     #         "num_track_switches": [],
#     #     }
#     #     last_td = terminal_date_list[-1]
#     #     last_td_list = [last_td] * len(sublist)
#     #     with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
#     #         results = executor.map(fac_contam_score, sublist, last_td_list)
#     #         end_results_creation = time.perf_counter()
#     #         print(f"Results creation: {end_results_creation-sublist_start}")
#     #         for result in results:
#     #             dict_compliance["fac_id"].append(result[0])
#     #             dict_compliance["contam_id"].append(result[1])
#     #             dict_compliance["terminal_date"].append(result[2])
#     #             dict_compliance["red_lean_score"].append(result[3])
#     #             dict_compliance["yellow_lean_score"].append(result[4])
#     #             dict_compliance["target_timeline"].append(result[5])
#     #             dict_compliance["num_time_segments"].append(result[6])
#     #             dict_compliance["num_track_switches"].append(result[7])
#     #         df_compliance = pd.DataFrame.from_dict(dict_compliance)
#     #         end_results_iteration = time.perf_counter()
#     #         print(f"Results iteration: {end_results_iteration-end_results_creation}")

#     #         start_write_to_sql = time.perf_counter()
#     #         df_compliance = df_compliance.astype(
#     #             {
#     #                 "fac_id": int,
#     #                 "contam_id": int,
#     #                 "terminal_date": int,
#     #                 "red_lean_score": str,
#     #                 "yellow_lean_score": str,
#     #                 "target_timeline": str,
#     #                 "num_time_segments": str,
#     #                 "num_track_switches": str,
#     #             }
#     #         )
#     #         df_compliance = pd.merge(
#     #             df_facilities, df_compliance, on="fac_id", how="inner"
#     #         )
#     #         append_or_replace = "replace" * (prev_start == start) + "append" * (
#     #             prev_start != start
#     #         )
#     #         conn = vdl.sql_query_conn()
#     #         df_compliance.to_sql(
#     #             write_table_name, conn, if_exists=append_or_replace, index=False
#     #         )
#     #         conn.close()
#     #         df_compliance = pd.DataFrame()
#     #         end_write_to_sql = time.perf_counter()
#     #         prev_start = end_write_to_sql
#     #         print(f"Time so far in score determination: {end_write_to_sql - start}")
#     # vdl.create_index(write_table_name, fac_id="ASC", contam_id="ASC", terminal_date="ASC")
#     # compliance_scores_percentiles_columns = [
#     #     "fac_id",
#     #     "contam_id",
#     #     "terminal_date",
#     #     "red_lean_score",
#     #     "yellow_lean_score",
#     #     "red_lean_percentile",
#     #     "yellow_lean_percentile",
#     #     "target_timeline",
#     #     "num_time_segments",
#     #     "num_track_switches",
#     # ]
#     # for t in terminal_date_list:
#     #     for contam in contam_list:
#     #         contam_start = time.perf_counter()
#     #         conn = vdl.sql_query_conn()
#     #         df_compliance = pd.read_sql_query(
#     #             f"SELECT * FROM score_and_percentile_fac_contam WHERE contam_id = {str(contam)} AND terminal_date = {t[2]}",
#     #             conn,
#     #         )
#     #         conn.close()
#     #         compliance_scores_percentiles_list = []
#     #         compliance_tups = [tuple(x) for x in df_compliance.to_numpy()]
#     #         df_comparison = df_compliance[
#     #             (df_compliance["red_lean_score"] != "TBD")
#     #             & (df_compliance["red_lean_score"] != "PMD")
#     #         ]
#     #         red_lean_list = np.array(
#     #             df_comparison["red_lean_score"].astype(float).to_list()
#     #         )
#     #         red_lean_list_of_lists = [red_lean_list] * len(compliance_tups)
#     #         yellow_lean_list = np.array(
#     #             df_comparison["yellow_lean_score"].astype(float).to_list()
#     #         )
#     #         yellow_lean_list_of_lists = [yellow_lean_list] * len(compliance_tups)

#     #         with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
#     #             results = executor.map(
#     #                 fac_contam_score_percentile,
#     #                 compliance_tups,
#     #                 red_lean_list_of_lists,
#     #                 yellow_lean_list_of_lists,
#     #             )
#     #             end_results_creation = time.perf_counter()
#     #             print(f"Results creation: {end_results_creation-contam_start}")
#     #             for result in results:
#     #                 compliance_scores_percentiles_list.append(result)
#     #             end_results_iteration = time.perf_counter()
#     #             print(
#     #                 f"Results iteration: {end_results_iteration-end_results_creation}"
#     #             )
#     #             print(
#     #                 f"Time so far (fac_id/contam_id percentile loop {contam_list.index(contam) + 1}): {end_results_iteration - start}"
#     #             )

#     #     df_compliance_scores_percentiles = pd.DataFrame(
#     #         compliance_scores_percentiles_list,
#     #         columns=compliance_scores_percentiles_columns,
#     #     )
#     #     start_write_to_sql = time.perf_counter()
#     #     df_compliance_scores_percentiles = df_compliance_scores_percentiles.astype(
#     #         {
#     #             "fac_id": int,
#     #             "contam_id": int,
#     #             "red_lean_score": str,
#     #             "yellow_lean_score": str,
#     #             "red_lean_percentile": str,
#     #             "yellow_lean_percentile": str,
#     #             "target_timeline": str,
#     #             "num_time_segments": str,
#     #             "num_track_switches": str,
#     #         }
#     #     )
#     #     df_compliance_scores_percentiles = pd.merge(
#     #         df_facilities, df_compliance_scores_percentiles, on="fac_id", how="inner"
#     #     )
#     #     conn = vdl.sql_query_conn()
#     #     df_compliance_scores_percentiles.to_sql(
#     #         "score_and_percentile_fac_contam", conn, if_exists="replace", index=False
#     #     )
#     #     conn.close()
#     #     end_write_to_sql = time.perf_counter()

#     # vdl.create_index(
#     #     "score_and_percentile_fac_contam",
#     #     fac_id="ASC",
#     #     ws_id="ASC",
#     #     contam_id="ASC",
#     #     terminal_date="ASC",
#     # )
#     # fac_id_contam_id_finish = time.perf_counter()
#     # print(
#     #     f"Total time to finish fac-contam percentile loop {contam_list.index(contam) + 1} of {len(contam_list)}: {fac_id_contam_id_finish - start}"
#     # )

#     # Begin generating average of facility scores in order to determine the overall percentile for the facility
#     for t in terminal_date_list:
#         start_average_facs = time.perf_counter()
#         conn = vdl.sql_query_conn()
#         df_fac_percentiles = pd.read_sql_query(
#             f"SELECT * FROM score_and_percentile_fac_contam WHERE terminal_date = {t[2]}",
#             conn,
#         )
#         conn.close()
#         fac_list = (
#             df_fac_percentiles["fac_id"].unique().tolist()
#         )  # list of all unique facilities
#         dict_percentile_averages = {
#             "fac_id": [],
#             "terminal_date": [],
#             "ave_red_lean_score": [],
#             "ave_yellow_lean_score": [],
#             "ave_target_timeline": [],
#             "ave_method_priority_level": [],
#             "ave_num_time_segments": [],
#             "ave_num_track_switches": [],
#         }
#         terminal_date = t[2]
#         td_list = [terminal_date] * len(fac_list)
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             results = executor.map(ave_fac_score, fac_list, td_list)
#             end_results_creation = time.perf_counter()
#             print(
#                 f"Results creation for average facility scores: {end_results_creation-start_average_facs}"
#             )
#             for result in results:
#                 dict_percentile_averages["fac_id"].append(result[0])
#                 dict_percentile_averages["terminal_date"].append(result[1])
#                 dict_percentile_averages["ave_red_lean_score"].append(result[2])
#                 dict_percentile_averages["ave_yellow_lean_score"].append(result[3])
#                 dict_percentile_averages["ave_target_timeline"].append(result[4])
#                 dict_percentile_averages["ave_method_priority_level"].append(result[5])
#                 dict_percentile_averages["ave_num_time_segments"].append(result[6])
#                 dict_percentile_averages["ave_num_track_switches"].append(result[7])
#             df_percentile_averages = pd.DataFrame(dict_percentile_averages)
#             end_results_iteration = time.perf_counter()
#             print(
#                 f"For terminal dates {t}, Results iteration for average facility scores: {end_results_iteration-end_results_creation}"
#             )
#         df_percentile_averages = pd.DataFrame(dict_percentile_averages)
#         df_percentile_averages = df_percentile_averages.astype(
#             {
#                 "fac_id": int,
#                 "terminal_date": int,
#                 "ave_red_lean_score": str,
#                 "ave_yellow_lean_score": str,
#                 "ave_target_timeline": str,
#                 "ave_method_priority_level": str,
#                 "ave_num_time_segments": str,
#                 "ave_num_track_switches": str,
#             }
#         )
#         df_percentile_averages = pd.merge(
#             df_facilities, df_percentile_averages, on="fac_id", how="inner"
#         )
#         conn = vdl.sql_query_conn()
#         df_percentile_averages.to_sql(
#             "score_and_percentile_ave_fac", conn, if_exists="replace", index=False
#         )
#         conn.close()
#         ave_fac_score_finish = time.perf_counter()
#         print(
#             f"Total time to finish average facility scores with terminal_date {t}: {ave_fac_score_finish - start_average_facs}"
#         )
#     vdl.create_index("score_and_percentile_ave_fac", fac_id="ASC", terminal_date="ASC")
#     # !!!!!!!!!!!!! SKIP THIS FOR NOW
#     # conn = vdl.sql_query_conn()
#     # df_compliance = pd.read_sql_query(
#     #     f"SELECT * from score_and_percentile_ave_fac", conn
#     # )
#     # conn.close()
#     # compliance_tups = [tuple(x) for x in df_compliance.to_numpy()]
#     # df_comparison = df_compliance[
#     #     (df_compliance["ave_red_lean_score"] != "TBD")
#     #     & (df_compliance["ave_red_lean_score"] != "PMD")
#     # ]
#     # red_lean_list = np.array(
#     #     df_comparison["ave_red_lean_score"].astype(float).to_list()
#     # )
#     # red_lean_list_of_lists = [red_lean_list] * len(compliance_tups)
#     # yellow_lean_list = np.array(
#     #     df_comparison["ave_yellow_lean_score"].astype(float).to_list()
#     # )
#     # yellow_lean_list_of_lists = [yellow_lean_list] * len(compliance_tups)
#     # compliance_scores_percentiles_list = []
#     # compliance_scores_percentiles_columns = [
#     #     "fac_id",
#     #     "ws_id",
#     #     "ave_red_lean_score",
#     #     "ave_yellow_lean_score",
#     #     "ave_score_red_lean_percentile",
#     #     "ave_score_yellow_lean_percentile",
#     #     "ave_target_timeline",
#     #     "ave_method_priority_level",
#     #     "ave_num_time_segments",
#     #     "ave_num_track_switches",
#     # ]
#     # with concurrent.futures.ProcessPoolExecutor() as executor:
#     #     results = executor.map(
#     #         percentile_of_ave_fac_score,
#     #         compliance_tups,
#     #         red_lean_list_of_lists,
#     #         yellow_lean_list_of_lists,
#     #     )
#     #     end_results_creation = time.perf_counter()
#     #     print(f"Results creation: {end_results_creation-ave_fac_score_finish}")
#     #     for result in results:
#     #         compliance_scores_percentiles_list.append(result)
#     #     end_results_iteration = time.perf_counter()
#     #     print(f"Results iteration: {end_results_iteration-end_results_creation}")
#     #     print(f"Time so far fac percentile: {end_results_iteration - start}")

#     # df_ave_fac_scores_percentiles = pd.DataFrame(
#     #     compliance_scores_percentiles_list,
#     #     columns=compliance_scores_percentiles_columns,
#     # )
#     # df_ave_fac_scores_percentiles = df_ave_fac_scores_percentiles.astype(
#     #     {
#     #         "fac_id": int,
#     #         "ws_id": int,
#     #         "ave_red_lean_score": str,
#     #         "ave_yellow_lean_score": str,
#     #         "ave_score_red_lean_percentile": str,
#     #         "ave_score_yellow_lean_percentile": str,
#     #         "ave_target_timeline": str,
#     #         "ave_method_priority_level": str,
#     #         "ave_num_time_segments": str,
#     #         "ave_num_track_switches": str,
#     #     }
#     # )

#     # conn = vdl.sql_query_conn()
#     # df_ave_fac_scores_percentiles.to_sql(
#     #     "score_and_percentile_ave_fac", conn, if_exists="replace", index=False
#     # )
#     # conn.close()
#     # vdl.create_index("score_and_percentile_ave_fac", fac_id="ASC", ws_id="ASC")
#     # ave_fac_percentile_finish = time.perf_counter()
#     # print(
#     #     f"Total time to finish average facility percentile: {ave_fac_percentile_finish - ave_fac_score_finish}"
#     # )

#     # # facility overage calculation:
#     # conn = vdl.sql_query_conn()
#     # fac_ws_list = pd.read_sql_query(
#     #     f"SELECT * from 'score_and_percentile_ave_fac'", conn
#     # ).values.tolist()
#     # conn.close()

#     # fac_overage_tally = []
#     # start_overage_tally = time.perf_counter()
#     # with concurrent.futures.ProcessPoolExecutor() as executor:
#     #     results = executor.map(facility_overage_counter, fac_ws_list)
#     #     end_results_creation = time.perf_counter()
#     #     for result in results:
#     #         fac_overage_tally.append(result)

#     # df_fac_overage = pd.DataFrame(
#     #     fac_overage_tally, columns=["fac_id", "ws_id", "overage_total", "overage_rate"]
#     # )

#     # df_fac_overage[["fac_id", "ws_id"]] = (
#     #     df_fac_overage[["fac_id", "ws_id"]].astype(str).astype(int)
#     # )
#     # df_fac_overage[["overage_total", "overage_rate"]] = df_fac_overage[
#     #     ["overage_total", "overage_rate"]
#     # ].astype(str)
#     # conn = vdl.sql_query_conn()
#     # df_fac_overage.to_sql(
#     #     "overage_count_and_percentile_fac", conn, if_exists="replace", index=False
#     # )
#     # conn.close()
#     # finish_overage_tally = time.perf_counter()
#     # print(df_fac_overage)
#     # print(finish_overage_tally - start_overage_tally)

#     # facility_overage_rates_list = df_fac_overage["overage_rate"].values.tolist()
#     # filtered_values = ["PMD", "TBD"]
#     # filtered_fac_overage_rates_list = []

#     # for fac in facility_overage_rates_list:
#     #     if fac not in filtered_values:
#     #         filtered_fac_overage_rates_list.append(float(fac))

#     # print(len(facility_overage_rates_list))
#     # print(len(filtered_fac_overage_rates_list))

#     # fac_overage_list = df_fac_overage.values.tolist()
#     # fac_overage_percentile_list = []

#     # with concurrent.futures.ProcessPoolExecutor() as executor:  # This is to use multiprocessing
#     #     results = executor.map(
#     #         facility_overage_percentile,
#     #         fac_overage_list,
#     #         [filtered_fac_overage_rates_list] * len(fac_overage_list),
#     #     )
#     #     end_results_creation = time.perf_counter()
#     #     for result in results:
#     #         fac_overage_percentile_list.append(result)
#     # df_fac_overage_percentile = pd.DataFrame(
#     #     fac_overage_percentile_list,
#     #     columns=[
#     #         "fac_id",
#     #         "ws_id",
#     #         "overage_total",
#     #         "overage_rate",
#     #         "overage_percentile",
#     #     ],
#     # )
#     # df_fac_overage_percentile[["fac_id", "ws_id"]] = (
#     #     df_fac_overage[["fac_id", "ws_id"]].astype(str).astype(int)
#     # )
#     # df_fac_overage_percentile[
#     #     ["overage_total", "overage_rate", "overage_percentile"]
#     # ] = df_fac_overage_percentile[
#     #     ["overage_total", "overage_rate", "overage_percentile"]
#     # ].astype(
#     #     str
#     # )
#     # conn = vdl.sql_query_conn()
#     # df_fac_overage_percentile.to_sql(
#     #     "overage_count_and_percentile_fac", conn, if_exists="replace", index=False
#     # )
#     # conn.close()
#     # vdl.create_index("overage_count_and_percentile_fac", fac_id="ASC", ws_id="ASC")

#     # !!!!!!!!!!!!!!!!!!!! END OF SKIPPED SECTION

#     print("FINISH ")

#     finish = time.perf_counter()
#     print(f"Seconds: {finish - start}")
#     pr.disable()
#     s = io.StringIO()
#     sortby = SortKey.CUMULATIVE
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print(s.getvalue())

# # Jae Test 12/18/22
# # 1296.8722965999914


# # Jae Test 4/3/23
# # 996.4312396999958

# # Jae test 7/18/23
# # Seconds: 1013.047556599995

# # Jae Test 8/05 - 08/06/23
# # 36037.6826302 +  677.6308779000537
