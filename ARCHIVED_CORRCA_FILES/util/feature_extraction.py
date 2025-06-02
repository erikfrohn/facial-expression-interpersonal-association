import subprocess
import os
import cv2
import pandas as pd
import numpy as np

def extract_features(input, id, output): 
    openface_path = r"C:\Users\Erik\Documents\facial-expression-synchrony\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
   
    role = 'navigator' if int(id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    file = os.path.join(input, "Video", f"pp{id}_{role}_video_frames.csv")
    df_phases = pd.read_csv(file)

    # iterate over all phases
    for _, row in df_phases.iterrows():
        input_file = f"pp{id}_{role}_reconstructed_video_and_audio_{row['name']}.mp4"
        input_path = os.path.join(input, "VideoAudio", input_file)
        output_file = f"pp{id}_{role}_{row['name']}.csv"
        output_path = os.path.join(output, output_file)
        
        if os.path.exists(output_path):
            print(f"{output_file} has already been processed. Continuing...")
            continue

        print(f"Extracting features from {input_file} to {output_file}")
        os.system(f"{openface_path} -f \"{input_path}\" -aus -of \"{output_path}\"")

def extract_features2(input, output): 
    openface_path = r"C:\Users\Erik\Documents\facial-expression-synchrony\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
   
    role = 'navigator' if int(id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    file = os.path.join(input, "Video", f"pp{id}_{role}_video_frames.csv")
    df_phases = pd.read_csv(file)

    # iterate over all phases
    for _, row in df_phases.iterrows():
        input_file = f"pp{id}_{role}_reconstructed_video_and_audio_{row['name']}.mp4"
        input_path = os.path.join(input, "VideoAudio", input_file)
        output_file = f"pp{id}_{role}_{row['name']}.csv"
        output_path = os.path.join(output, output_file)
        
        if os.path.exists(output_path):
            print(f"{output_file} has already been processed. Continuing...")
            continue

        print(f"Extracting features from {input_file} to {output_file}")
        os.system(f"{openface_path} -f \"{input_path}\" -aus -of \"{output_path}\"")