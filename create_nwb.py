# RUNNING SCRIPT INFO
# For printing options use: 
#   --sf (show input session folders) 
#   --sl (show loading and shapes of files)
#   --noprint (to disable all printing)
# For selecting the session:
#   --nr *int* (use the session folder of the nth folder, alphabetically sorted)
# For folder and current directory pathnames (if specified will override default value):
# After running script once default paths will be overriden by given paths, stored in a dictionary pickle file. 
# Then these arguments are optional.
#   --usecd (to use current directory instead of default) (only use this in the folder which contains the input and output folders)
#   --ip *folder* (to use a specified input folder) (e.g. --ip input_folder)
#   --op *folder* (to use a specified output folder) (e.g. --op output_folder)

# Example structure:
# Rat1:
#   - create_nwb.py
#   - tools
#   - input_folder
#       - session_1
#           - *.csv, .txt, .log files*
#       - session_2
#           - *.csv, .txt, .log files*
#   - output_folder
#       - session_1
#           - *.csv, .txt, .log files*
#       - session_2
#           - *.csv, .txt, .log files*


import sys
import os
import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from datetime import datetime
from uuid import uuid4
from dateutil import tz

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.file import Subject

# imported from self coded tools
from tools.pathnames import find_paths, find_session_folders
from tools.process_log import process_log
import tools.process_dataframe as pdf

# safely read and load files from found paths based on a key (for more information read pathnames.py doc)
# every .ext is loaded using the corresponding function
# returns pd.dataframe for every .ext except for .npy it returns np.array
def safe_read(key, paths, ext):
    ext = ext.lower()
    if ext == "log":
        function = process_log
    # to be added
    #elif ext == "txt" 
    #    function = None
    elif ext == "csv":
        function = pd.read_csv
    elif ext == "npy":
        function = np.load
    else:
        raise Exception("Invalid .ext!")
    try:
        # pathname is not case sensitive
        result = function(paths.get(key.lower()))
        if print_loading: print(f"Loaded {key} [Shape: {result.shape}]")
        return result
    except:
        if print_loading: print(f"{key}.{ext} not loaded.")
        return None

# parses the given date into ISO_8601 Duration format https://en.wikipedia.org/wiki/ISO_8601#Durations
# e.g.: A Rat born on jan 1 2025 and current date is jan 2 2026 will return P1Y1D
# Time can also be included but is usually unnecessary
def parse_ISO_8601(start, end=None, show_T = False):
    end = end or datetime.now()
    
    y = end.year - start.year - ((end.month, end.day) < (start.month, start.day))
    anniv = start.replace(year=start.year + y)

    delta = end - anniv
    s = int(delta.total_seconds())
    
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    
    return "P{}{}{}{}{}{}{}".format(
        f"{y}Y" if y else "",
        f"{d}D" if d else "",
        "T" if any((h, m, s)) and show_T else "",
        f"{h}H" if h and show_T else "",
        f"{m}M" if m and show_T else "",
        f"{s}S" if s and show_T else "",
        "" if any((y, d, h, m, s)) else "T0D"
    )

# takes dataframe (with column 'Time (seconds)' indicating time in seconds) and returns the element closest to 0
def find_index_time_zero(df):
    index_positive = np.where(df['Time (seconds)']>=0)[0][0] # first positive element
    if np.abs(df['Time (seconds)'][index_positive]) < np.abs(df['Time (seconds)'][index_positive-1]):
        return index_positive
    else:
        return index_positive-1

# creates a nwb file using pynwb
# uses global variables nwb_*
def create_nwb_file():
    return NWBFile(
        session_description=nwb_session_description,  # required
        identifier=str(uuid4()),  # required
        session_start_time=nwb_session_start_time,  # required
        session_id=str(nwb_session_id),  # optional
        experimenter=[
            nwb_experimenter
        ],  # optional
        lab=nwb_lab,  # optional
        institution=nwb_institution,  # optional
        experiment_description=nwb_experiment_description,  # optional
        keywords=nwb_keywords,  # optional
        related_publications=nwb_related_publications,  # optional
    )

# creates a subject to be added in nwb_file
# uses global variables subject_*
def create_subject():
    return Subject(
        subject_id= subject_id,
        age=rat_age,
        description=f"Rat {rat_nr}",
        species=species_name,
        sex=sex,
    )

# adds the created subject to nwb file
# if there is an existing subject already (e.g. in case of editing files) it will return an error
def add_subject():
    subject = create_subject()
    if nwbfile.subject is not None:
        print(f"Failed to add subject! Rat {subject.subject_id} might already exist in nwb file!")
    else:
        nwbfile.subject = subject
        print(f"Added Rat {subject.subject_id}.")
    return subject

# creates timeseries
# uses global variables lfp_*
def create_timeseries():
    return TimeSeries(
        name=lfp_name,
        description=lfp_description,
        data=lfp_data,
        unit=lfp_unit,
        timestamps=lfp_timestamps,
    )

# add timeseries to nwbfile
# if there is an existing timeseries with the same name already (e.g. in case of editing files) it will return an error
def add_timeseries():
    lfp = create_timeseries()
    try:
        nwbfile.get_acquisition('lfp')
        print(f"Failed to add subject! {lfp.name} might already exist in nwb file!")
    except:
        print(f"Added {lfp.name}.")
        nwbfile.add_acquisition(lfp)
    return lfp

# creates a behavior module to add data such as spatial_series/position
# takes global arguments behavior_*
def create_behavior_module():
    return nwbfile.create_processing_module(
        name=behavior_name,
        description=behavior_description
    )

# splits the labels of positional data in loaded dataframe
# e.g. "Rat_X" -> X
def split_labels(df):
    return list({
        col.rsplit('_', 1)[0]
        for col in df.columns
        if col.endswith(('_X', '_Y'))
    })

# creates position object to hold positional data in the form of SpatialSeries
# takes str(pos_name) and dataframe with original data
def create_position_obj(pos_name, df):
    position_obj = Position(name=pos_name)
    labels = split_labels(df)
    for label in labels:
        spatial_series_obj = SpatialSeries(
            name=f"{label}",
            description=f"(x,y) {label} position in HexMaze",
            data=np.array([df[f'{label}_X'].values, df[f'{label}_Y'].values]).T,
            timestamps=df['Time (seconds)'].values,
            unit = "pixels",
            reference_frame="(0,0) is bottom left corner",
        )

        position_obj.add_spatial_series(spatial_series_obj)
    return position_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes arguments"
    )
    parser.add_argument("--sf", action="store_true", help = "Show input folders")
    parser.add_argument("--noprint", action="store_true", help = "Disable all console output/printing")
    parser.add_argument("--sl", action="store_true", help = "Show loading of files, including shapes")
    parser.add_argument("--usecd", action="store_true", help = "Use current directory as root")
    parser.add_argument("--ip", action = "store", type=str, required=False, help = "Specify input folder (e.g. python file.py ip='input_folder'")
    parser.add_argument("--op", action = "store", type=str, required=False, help = "Specify output folder (e.g. python file.py op='output_folder' \n If not given op is same as ip")
    parser.add_argument("--nr", action = "store", type=int, required=False, default=0, help = "Specify what session should be selected (int)")
    args = parser.parse_args()

    if args.noprint:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    #boolean for printing loading files
    print_loading = args.sl


    # --- Filenames --- # (CHANGE TO CORRECT PATHS OR USE GIVE ARGUMENTS IN TERMINAL)


    # open dictionary of ip op folders or create new if not existing
    try: 
        with open('ip_op_folders.pkl', 'rb') as f:
            ip_op_folders = pickle.load(f)
        print("Loaded IP & OP folder paths!")
    except:
        ip_op_folders = {}
        print("IP & OP folder paths not existing. Created dictonary to store them.")

    # root to input folder
    if args.usecd:
        root = os.getcwd() + "/"
        ip_op_folders["root"] = root
    # if dictionary contains the filepath for root
    elif "root" in ip_op_folders:
        root = ip_op_folders["root"]
    # else use default value
    else:
        root = r"C:\Users\Jacob\OneDrive\Documenten\University\Masters Internship\Scripts\Data_Structure/"

    # input folder after root containing folders with session data
    if args.ip is not None:
        input_folder = args.ip + "/"
        # store specified ip to dictonary
        ip_op_folders["ip"] = input_folder
    # if dictionary contains the filepath for ip
    elif "ip" in ip_op_folders:
        input_folder = ip_op_folders["ip"]
    # else use default value
    else:
        input_folder= r"nwb_data/" # Folder indicating input of a certain subject, e.g. can be ip_rat1 or ip_rat2

    # output folder/paths for created nwbfile
    if args.op is not None:
        output_folder = args.op + "/"
        # store specified op to dictionary
        ip_op_folders["op"] = input_folder
    # if dictionary contains the filepath for op
    elif "op" in ip_op_folders:
        output_folder = ip_op_folders["op"]
    # else use default value
    else:
        output_folder = input_folder


    # save given input and output folders
    if args.ip is not None or args.op is not None:
        with open('ip_op_folders.pkl', 'wb') as f:
            pickle.dump(ip_op_folders, f)



    print(f"Input folder: {root + input_folder}")
    print(f"Output folder: {root + output_folder}")
    session_folders = find_session_folders(Path(root + input_folder)) # finds all folders in the input folder, these should be 1 folder per session
    session_paths = []


    # --- END Filenames --- #


    for session_dir in session_folders:
        # this appends the paths, in each session directory, to a list of all paths
        paths = find_paths(session_dir)
        session_paths.append(paths)
        # optionally prints this based on arg "--sf"
        if args.sf:
            print(f"Session {session_dir.name}:")
            print(paths, '\n')

    # specify which session folder should be read with nr
    paths = session_paths[args.nr]
    # safe_read to corresponding variables
    df_coordinates = safe_read('Coordinates_Full', paths.csv_paths, 'csv')
    df_coordinates_with_frames = safe_read('Coordinates_Full_with_frames', paths.csv_paths, 'csv')
    df_framewise_ts = safe_read('framewise_ts', paths.csv_paths, 'csv')
    df_framewise_seconds = safe_read('stitched_framewise_seconds', paths.csv_paths, 'csv')
    df_log = safe_read(next(iter(paths.log_paths)), paths.log_paths, 'log')
    lfp_channels = safe_read('lfp_channels', paths.numpy_paths, 'npy')
    lfp_data = safe_read('lfp_data', paths.numpy_paths, 'npy')
    lfp_timestamps = safe_read('lfp_timestamps', paths.numpy_paths, 'npy')

    df = pdf.merge_df(df_coordinates_with_frames, df_framewise_seconds, 'Time (seconds)')


    # --- METADATA --- # Can change this


    # start time of session
    # get Time (seconds) from dataframe close to 0
    index_time_zero = find_index_time_zero(df)
    nwb_session_start_time = datetime.fromtimestamp(df_coordinates_with_frames['Timestamp'][index_time_zero])
    timezone = tz.gettz('Europe/Amsterdam')
    nwb_session_start_time = nwb_session_start_time.replace(tzinfo=timezone)
    
    # nwb metadata
    nwb_session_id = session_folders[0].name
    nwb_session_description = f"Rat1 Hexmaze Session {nwb_session_id}"
    nwb_experimenter = "Person"
    nwb_lab = "Genzel Lab"
    nwb_institution = "Donders Institute, Radboud University"
    nwb_experiment_description = "Rat HexMaze"
    nwb_keywords=["behavior", "ephys", "maze"],
    nwb_related_publications="N/A",

    # rat/subject metadata
    rat_nr = str(1)
    subject_id = rat_nr.zfill(3)
    rat_birthday = datetime(2019, 1, 1, 0, 0, 0, tzinfo = timezone)
    rat_age = parse_ISO_8601(rat_birthday, nwb_session_start_time)
    species_name = "Rattus norvegicus"
    sex = "M"

    # lfp/timeseries metadata
    lfp_name="lfp"
    lfp_description="lfp voltage"
    lfp_unit="uV"

    # behavior metadata
    behavior_name = "Behavior"
    behavior_description = "Positional data"

    # position metadata
    position_name = "Position"


    # --- END METADATA --- #


    # create nwb file
    nwbfile = create_nwb_file()
    print(f"Created nwb file for session {nwb_session_id}.")

    # add subject to nwb
    subject = add_subject()

    # add lfp timeseries data
    lfp = add_timeseries()

    # create behavior module
    behavior_module = create_behavior_module()
    
    # create position object
    position_obj = create_position_obj(position_name, df)

    # add position object to behavior module
    behavior_module.add(position_obj)

    # save nwbfile
    output_name = str(session_dir.stem) + ".nwb"
    try:
        with NWBHDF5IO(root + output_folder + output_name, "w") as io:
            io.write(nwbfile)
        print(f"Saved to {root + output_folder + output_name}!")
    except:
        print(f"Saving NWB file to {root + output_folder + output_name} failed! File might already exist.")