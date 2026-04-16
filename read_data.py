import argparse
import pandas as pd
import numpy as np
import re
from glob import glob
import sys
import os

parser = argparse.ArgumentParser(description="Process sensor data and save outputs.")
parser.add_argument("--output", required=False, default="SPARL3/data_f0_sparl3_with_gyro_baro_2.pkl", help="Output file for data")
parser.add_argument("--input", required=False, default="SPARL3/Flight_Controller_0", help="Input folder for data")
parser.add_argument("--baro_input", required=False, default="SPARL3/Flight_Controller_0/Baro", help="Input folder for gyroscope data")
parser.add_argument("--gyro_input", required=False, default="SPARL3/Flight_Controller_0/Gyro", help="Input folder for gyroscope data")
parser.add_argument("--scale_time", type=float, required=False, default=1e6, help="Scale factor for time column; default is 1e6")
parser.add_argument("--annotations", required=False, default="SPARL3/Annotations/", help="Path to annotations folder; default is 'SPARL3/Annotations/'")
# sparl default
parser.add_argument("--sync_start", required=False, default="12.3831, 5.2510, 17.6155, 12.2572", help="List of start times for synchronization")
parser.add_argument("--sync_end", required=False, default="nan, nan, nan, nan", help="List of end times for synchronization")
# sparl3 default
# parser.add_argument("--sync_start", required=False, default="12.3831, 5.2510, 17.6155, 12.2572", help="List of start times for synchronization")
# parser.add_argument("--sync_end", required=False, default="490.6572, 429.1368, 445.9484, 561.5143", help="List of end times for synchronization")
# sparl3 with microsecond
# parser.add_argument("--sync_start", required=False, default="2429.5888, 3434.7770, 40.9271, 876.1991", help="List of start times for synchronization")
# parser.add_argument("--sync_end", required=False, default="2907.8629, 3858.6628, 469.2601, 1425.4562", help="List of end times for synchronization")

args = parser.parse_args()
# sync_start_list
# Helper functions
def remove_offset(dat):
    acc_cols = ['Acc.x', 'Acc.y', 'Acc.z']
    if all(col in dat.columns for col in acc_cols):
        acc_median = dat[acc_cols].median()
        dat[acc_cols] = dat[acc_cols] - acc_median

    gyro_cols = ['Gyro.x', 'Gyro.y', 'Gyro.z']
    if all(col in dat.columns for col in gyro_cols):
        gyro_median = dat[gyro_cols].median()
        dat[gyro_cols] = dat[gyro_cols] - gyro_median

    # cm = dat.iloc[:, 1:4].median()
    # dat.iloc[:, 1] -= cm.iloc[0]
    # dat.iloc[:, 2] -= cm.iloc[1]
    # dat.iloc[:, 3] -= cm.iloc[2]
    return dat

def extract_scenario(file):
    s = file.split("/")[-1]
    s = re.split(r"_", s)[0]
    s = re.split(r"S", s)[1]
    return int(s)

def extract_experiment(file):
    e = re.split(r"Record", file)[1]
    e = re.split(r"_|/", e)[0]
    return int(e)

def pretty_data(res, files):
    dat = pd.DataFrame({
        'scenario': [extract_scenario(f) for f in files],
        'experiment': [extract_experiment(f) for f in files],
        'data': res
    })

    # Ensure required columns exist in each DataFrame in 'data'
    for i in range(len(dat)):
        for col in ['class', 'transportation', 'container', 'No loading']:
            if col not in dat['data'].iloc[i].columns:
                dat['data'].iloc[i][col] = pd.NA
    return dat

def add_labels(data, ends, data_labels):
    ends = np.array(ends)
    print(data.iloc[0]['data'].columns)
    for i in range(len(data)):
        # Match scenario and experiment 
        ind_labels = data_labels[
            (data_labels['scenario'] == data['scenario'].iloc[i]) &
            (data_labels['experiment'] == data['experiment'].iloc[i])
        ].index

        if len(ind_labels) > 0:
            dl = data_labels['data'][ind_labels[0]].copy()
            # Time synchronization
            if not np.isnan(ends[i]):
                sync_times = dl[dl['Synchronization'] == 1]['time']
                if len(sync_times) >= 2:
                    diffs = np.diff(sync_times)
                    if len(diffs) > 0:
                        max_diff_index = np.argmax(diffs)
                        sync_end = sync_times.iloc[max_diff_index + 1]
                        dl['time'] = dl['time'] * ends[i] / sync_end

            # Match times
            current_time_values = data['data'][i]['time']
            row_ind = [(dl['time'] - ti).abs().idxmin() for ti in current_time_values]
            dl_row = dl.iloc[row_ind].reset_index(drop=True)
            # print(dl_row[dl_row["Electric pedestrian pallet truck"]==1])
            # Assign class
            class_columns = [
                "Driving(straight)",
                "Driving(curve)",
                "Lifting(raising)",
                "Lifting(lowering)",
	            "Lifting and driving",
                "Standing",
	            "Docking",
	            "Forks(entering or leaving front)",
	            "Forks(entering or leaving side)",
	            "Wrapping",
	            "Wrapping(preparation)",
                "Error",
                "Synchronization",
                "None"
            ]
            print(dl_row.head())
            dl_row['class'] = dl_row[class_columns].apply(
                lambda row: row.idxmax() if row.max() > 0 else 'None', axis=1
            )
            # Handle multiple active classes
            for index, row in dl_row.iterrows():
                active_classes = row[class_columns][row[class_columns] == 1].index.tolist()
                if len(active_classes) > 1:
                    if 'Error' in active_classes:
                        dl_row.at[index, 'class'] = 'Error'
                    # elif 'Electric pedestrian pallet truck' in active_classes:
                    #     dl_row.at[index, 'class'] = 'Electric pedestrian pallet truck'
                    else:
                        dl_row.at[index, 'class'] = active_classes[0]
                        #raise ValueError(f"Multiple active classes without 'Error' for row {index}: {active_classes}")

            # Assign the class, transportation, container, No loading
            for col in ['class', 'transportation', 'container', 'No loading']:
                if col in dl_row.columns:
                    data['data'].iloc[i][col] = dl_row[col].values
                else:
                    data['data'].iloc[i][col] = pd.NA
        else:
            print(f"Warning: No matching annotations for row {i}!")
            data['data'].iloc[i][['class', 'transportation', 'container', 'No loading']] = pd.NA
    return data

def rel_data(data, begins):
    for i in range(len(data)):
        df = data['data'].iloc[i].copy()
        df['time'] -= begins[i]
        data.at[i, 'data'] = df
    return data


files_accel = sorted(glob(args.input + "/*.csv"))
res_data = []

if args.gyro_input:
    print(f"Gyro data folder provided: {args.gyro_input}")
sync_start = [np.nan if s.strip().lower() == 'nan' else float(s) 
              for s in args.sync_start.split(",")]
# sync_ctr = 0
for file_accel in files_accel:
    base_filename_accel = os.path.basename(file_accel)
    d_acc = pd.read_csv(file_accel)
    d_acc.columns = ['time', 'Acc.x', 'Acc.y', 'Acc.z']
    d_gyro = None
    if args.gyro_input:
        gyro_filename = base_filename_accel.replace("_ACC_", "_GYR_")
        file_gyro = os.path.join(args.gyro_input, gyro_filename)
        if os.path.exists(file_gyro):
            d_gyro = pd.read_csv(file_gyro)
            d_gyro.columns = ['time', 'Gyro.x', 'Gyro.y', 'Gyro.z']
    
    d_baro = None
    if args.baro_input:
        baro_filename = base_filename_accel.replace("_ACC_", "_BARO_")
        file_baro = os.path.join(args.baro_input, baro_filename)
        if os.path.exists(file_baro):
            d_baro = pd.read_csv(file_baro)
            d_baro.columns = ['time', 'Baro.x'] 

    merged_df = d_acc.copy()
    if d_gyro is not None:
        merged_df = pd.merge_asof(merged_df, d_gyro, on='time', direction='nearest', tolerance=1000)
    
    if d_baro is not None:
        merged_df = pd.merge(merged_df, d_baro, on='time', how='outer', sort=True)
    
    merged_df.interpolate(method='linear', inplace=True)
    
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)
    
    d = merged_df.loc[merged_df['time'].isin(d_acc['time'])].reset_index(drop=True)
    
    t = d['time'] - d['time'].min()
    d['time'] = np.round(t / args.scale_time, 4)
    res_data.append(remove_offset(d))
res_data = pd.Series(res_data)
data_abs = pretty_data(res_data, files_accel)
data_abs = data_abs.sort_values(by=['scenario', 'experiment']).reset_index(drop=True)
files_labels = glob(args.annotations + "/*.csv")
res_labels = []
for file in files_labels:
    print(f"Processing labels from {file}")
    d = pd.read_csv(file)
    d = d.iloc[:, :-2]
    ind_start = d[d['Synchronization'] == 1].index.min()
    print(f"Start index: {ind_start}")
    time = np.arange(len(d)) / 30 - ind_start / 30
    d['time'] = time
    res_labels.append(d)

data_labels = pretty_data(res_labels, files_labels)
sync_start = [np.nan if s.strip().lower() == 'nan' else float(s) 
              for s in args.sync_start.split(",")]
sync_end = [np.nan if s.strip().lower() == 'nan' else float(s) 
            for s in args.sync_end.split(",")]
data = rel_data(data_abs, sync_start)
data_oc = add_labels(data, sync_end, data_labels)
# print(data_oc[
#     (data_oc['scenario'] == 4) & 
#     (data_oc['experiment'] == 1)
# ]['data'].iloc[0].head())
data_oc.to_pickle(args.output)