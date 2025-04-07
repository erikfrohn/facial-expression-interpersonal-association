#Interpolate missing values 
import pandas as pd
import numpy as np
import os

def handle_missing_data(path_to_analysis_folder, participant_id):
    role = 'navigator' if int(participant_id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    # Define the path to the facial_expression folder within the analysis folder
    facial_expression_folder = os.path.join(path_to_analysis_folder, 'facial_expression')
    path_to_video_folder = os.path.join(path_to_analysis_folder, 'Video')
    # List all files in the analysis folder
    all_files = os.listdir(facial_expression_folder)
    

    # Determine the CSV file name with markers
    csv_filename = f"pp{participant_id}_{role}_video_frames.csv"
    path_to_frame_csv_file = os.path.join(path_to_video_folder, csv_filename)
    frame_data_csv = pd.read_csv(path_to_frame_csv_file)

    # Iterate over all .avi files in the folder
    for index, row in frame_data_csv.iterrows():
        name = frame_data_csv['name'][index]
        input_AU_NaN_file = f"pp{participant_id}_{role}_{name}_output_au_no_black_NaN.csv"
        
        if input_AU_NaN_file in all_files:
            #print(f'Input file for NaN recovery: {input_AU_NaN_file}')
            path_input_AU_NaN_file = os.path.join(facial_expression_folder, input_AU_NaN_file)
            
            # Check if the input file exists
            if not os.path.exists(path_input_AU_NaN_file):
                #print(f"Input file {path_input_AU_NaN_file} not found. Skipping {name} of pp{participant_id}.")
                continue
            
            # Load the CSV file
            csv_AU_NaN = pd.read_csv(path_input_AU_NaN_file)
            #print(csv_AU_NaN.shape)

            # Generate output file and path
            output_csv_AU_recovered = f"pp{participant_id}_{role}_{name}_output_au_no_black_recovered.csv"
            path_output_csv_AU_recovered = os.path.join(facial_expression_folder, output_csv_AU_recovered)
           
            max_consecutive_nans = csv_AU_NaN.apply(lambda col: col.isna().astype(int).groupby(col.notna().astype(int).cumsum()).cumsum().max())
            if max_consecutive_nans.max() > 8:
                print(f"More than 8 consecutive frames are missing. Excluding {name} of pp{participant_id}.")
                continue
            else:
                # Interpolate missing values
                data_interpolated = csv_AU_NaN.interpolate(method='linear', limit_direction='forward', axis=0)
                    
            #check if the file already exists
            if os.path.exists(path_output_csv_AU_recovered):
                #print(f"Output file {output_csv_AU_recovered} already exists. Skipping recovery of {name} for pp{participant_id}")
                continue
            # Save the interpolated data to a new CSV file
            data_interpolated.to_csv(path_output_csv_AU_recovered, index=False)
            print(f"Interpolated missing values of pp{participant_id} and saved to {output_csv_AU_recovered}")