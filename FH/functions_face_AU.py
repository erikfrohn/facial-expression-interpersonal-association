import subprocess
import os
import cv2
import pandas as pd
import numpy as np

def get_AU(path_to_analysis_folder, participant_id): 
    role = 'navigator' if int(participant_id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even

    # Define the path to the OpenFace executable
    openface_path = r"C:\Users\Erik\Documents\facial-expression-synchrony\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    # Define the path to the facial_expression folder within the analysis folder
    path_to_video_folder = os.path.join(path_to_analysis_folder, 'Video')
    facial_expression_folder = os.path.join(path_to_analysis_folder, 'facial_expression')
    # List all files in the facial_expression folder
    all_files = os.listdir(facial_expression_folder)

    # Determine the CSV file name with markers
    csv_filename = f"pp{participant_id}_{role}_video_frames.csv"
    path_to_frame_csv_file = os.path.join(path_to_video_folder, csv_filename)
    frame_data_csv = pd.read_csv(path_to_frame_csv_file)

    # Iterate over all rows in the CSV file
    for index, row in frame_data_csv.iterrows():
        name = row['name']
        input_avi_file = f"pp{participant_id}_{role}_{name}_reconstructed_video_no_black.mp4"
        #print(input_avi_file)
        if input_avi_file in all_files:
            #print(f'Input file for AU generation: {input_avi_file}')
            path_input_avi = os.path.join(facial_expression_folder, input_avi_file)
            # Generate output file and path
            output_csv_AU = f"pp{participant_id}_{role}_{name}_output_au_no_black.csv"
            path_output_csv_AU = os.path.join(facial_expression_folder, output_csv_AU)
            print(f'Output file for AU generation: {output_csv_AU}')

            # Check if the output file already exists
            if os.path.exists(path_output_csv_AU):
                print(f"Output file {output_csv_AU} already exists. Skipping AU generation of {name} for pp{participant_id}")
                continue

            # Call OpenFace on the video file to extract AUs
            command = [
                openface_path,
                "-f", path_input_avi,  # Input video file
                "-aus",  # Extract action units
                "-of", path_output_csv_AU  # Output CSV file
            ]

            print(f'Running command: {" ".join(command)}')

            # Run the command and capture output
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.returncode != 0:
                print(f"Error processing file {input_avi_file}: {process.stderr}")
                continue
            else:
                print(f"Successfully generated AU of pp{participant_id}")
    