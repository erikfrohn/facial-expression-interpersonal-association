import os
import pandas as pd
import numpy as np

def corrCA_dataframe(path_to_new_data, all_path_AU_files):
    
    combined_data = {
        'discussion_phase_0': [],
        'instructional_video_0': [],
        'discussion_phase_1': [],
    }
    # combined_data = {} 
    path_to_result_folder = os.path.join(path_to_new_data, 'Results facial expressions', 'corrCA')
    names =  ['instructional_video_0','discussion_phase_0', 'discussion_phase_1'] 
    for path_AU_file in all_path_AU_files: 
        for name in names:
            if name in path_AU_file and os.path.exists(path_AU_file):
            #if os.path.exists(path_AU_file):
                #print(f'File {path_AU_file} found. Add data')
                AU_df = pd.read_csv(path_AU_file)
                au_columns = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                           ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                           ' AU25_r', ' AU26_r', ' AU45_r']
                AU_df = AU_df[au_columns]
                # Transpose the DataFrame to DxT format
                AU_df = AU_df.T

                combined_data[name].append(AU_df)
            else:
                #print(f"File {path_AU_file} not found. Skipping.")
                continue

    # Save the combined data for each phase to separate CSV files
    for phase, data_list in combined_data.items():
        if data_list:
            combined_df = pd.concat(data_list, ignore_index=True)
            combined_df.to_csv(os.path.join(path_to_result_folder, f"{phase}_combined_AU_data.csv"), index=False)
            print(f"All pp data for {phase} saved to {phase}_combined_AU_data.csv with shape {combined_df.shape}")
        else:
            print(f"No data found for {phase}")