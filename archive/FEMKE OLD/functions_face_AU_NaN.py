#adding NaN values on the missing frames in the AU csv 
import os
import pandas as pd
import numpy as np

def NaN_AU_csv(path_to_analysis_folder, participant_id):
    role = 'navigator' if int(participant_id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    # Determine the CSV file name with markers
    path_to_video_folder = os.path.join(path_to_analysis_folder, 'Video')
    csv_filename = f"pp{participant_id}_{role}_video_frames.csv"
    path_to_frame_csv_file = os.path.join(path_to_video_folder, csv_filename)
    frame_data_csv = pd.read_csv(path_to_frame_csv_file)

    # Define the path to the facial_expression folder within the analysis folder
    facial_expression_folder = os.path.join(path_to_analysis_folder, 'facial_expression')

    for index, row in frame_data_csv.iterrows():
        name = frame_data_csv['name'][index]
        # Save the recovered data to a new CSV file
        recovered_AU_csv_filename = f"pp{participant_id}_{role}_{name}_output_au_no_black_NaN.csv"
        # Define the new output path
        path_to_recovered_AU_csv_file = os.path.join(facial_expression_folder, recovered_AU_csv_filename)

        # Check if the output file already exists
        if os.path.exists(path_to_recovered_AU_csv_file):
            #print(f"Output file {recovered_AU_csv_filename} already exists. Skipping adding NaN of {name} for pp{participant_id}")
            continue

        # Finding actual frames
        txt_filename = f"pp{participant_id}_{role}_{row['name']}_video_frames.txt"
        path_to_frame_txt_file = os.path.join(path_to_video_folder, txt_filename)
        #print(path_to_frame_txt_file)
        if os.path.exists(path_to_frame_txt_file):
            with open(path_to_frame_txt_file, 'r') as file:
                actual_frames = [int(frame) for frame in file.read().strip().split(',')]
                #print(f"Length of actual frames: {len(actual_frames)}")
        else:
            #print(f"Text file not found: {path_to_frame_txt_file}")
            continue 

        #Finding CSV file including AU
        AU_csv_filename =  f"pp{participant_id}_{role}_{name}_output_au_no_black.csv"
        path_to_AU_csv_file = os.path.join(facial_expression_folder, AU_csv_filename)
        if not os.path.exists(path_to_AU_csv_file):
            #print(f"AU CSV file not found: {path_to_AU_csv_file}")
            continue
        AU_data_csv = pd.read_csv(path_to_AU_csv_file)

        if len(AU_data_csv) != len(actual_frames):
            #print(f"Length mismatch: AU data ({len(AU_data_csv)}) vs actual frames ({len(actual_frames)}). Skipping adding NaN of {name} for pp{participant_id}")
            continue 
    
        # Create a DataFrame to hold the recovered data
        recovered_AU_data = pd.DataFrame(columns=['frame'] + list(AU_data_csv.columns))
        # Writing actual frames (and adding NaN to missing frames)
        for frame_num in range(len(actual_frames)):
            if frame_num > 0:
                frame_diff = actual_frames[frame_num] - actual_frames[frame_num-1]
                if frame_diff > 1:
                    for j in range(1, frame_diff):
                        # Create a row of NaN values with the correct frame number
                        nan_row = pd.Series([actual_frames[frame_num-1] + j] + [np.nan] * len(AU_data_csv.columns), index=recovered_AU_data.columns)
                        recovered_AU_data = pd.concat([recovered_AU_data, nan_row.to_frame().T], ignore_index=True)
            # Append the actual frame data with the frame number
            actual_row = pd.Series([actual_frames[frame_num]] + list(AU_data_csv.iloc[frame_num]), index=recovered_AU_data.columns)
            recovered_AU_data = pd.concat([recovered_AU_data, actual_row.to_frame().T], ignore_index=True)

        # save the recovered data
        recovered_AU_data.to_csv(path_to_recovered_AU_csv_file, index=False)
        print(f"Recovered AU data of pp{participant_id} saved to: {recovered_AU_csv_filename}")