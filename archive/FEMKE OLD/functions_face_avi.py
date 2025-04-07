import csv
import os
import pandas as pd

def get_avi_file(path_to_analysis_folder, path_to_avi_files, participant_id):
    # Filter AVI files for the current participant and role
    role = 'navigator' if int(participant_id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    print(role)

    csv_filename = f"pp{participant_id}_{role}_video_frames.csv"
    path_to_frame_csv_file = os.path.join(path_to_analysis_folder, csv_filename)
    print(f"Reading CSV file: {path_to_frame_csv_file}")
    frame_data_csv = pd.read_csv(path_to_frame_csv_file)

    if len(path_to_avi_files) ==2:
        avi_file = next((file for file in path_to_avi_files if f"{role}" in file), None)
        if not avi_file:
            print(f"No AVI file found for role {role}")
        return avi_file
    print(f"Processing AVI file: {avi_file}")
    if len(path_to_avi_files) >=2:
        csv_filename_avi = f"pp{participant_id}_{role}_video_frames-PC-41823.csv"
        path_to_frame_csv_file_avi = os.path.join(path_to_analysis_folder, csv_filename_avi)
        frame_data_csv_avi = pd.read_csv(path_to_frame_csv_file_avi)
        for index, row in frame_data_csv_avi.iterrows():
            video_count = int(row['video_cnt'])
            return avi_file[video_count]
        
