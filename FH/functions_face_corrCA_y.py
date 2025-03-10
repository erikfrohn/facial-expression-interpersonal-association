
import os
import pandas as pd
import numpy as np

def get_y(path_to_new_data, path_to_analysis_folder, participant_id):
    # Define the path to the w CSV file
    path_to_result_folder = os.path.join(path_to_new_data, 'Results facial expressions')
    w_folder = os.path.join(path_to_result_folder, "corrCA")
    w_files = [f for f in os.listdir(w_folder) if f.endswith('.csv') and 'w' in f] 
    #print(w_files)

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
        input_AU_file = f"pp{participant_id}_{role}_{name}_output_au_no_black_recovered.csv"

          # Select the input_AU_file from all files
        if input_AU_file in all_files:
            path_input_AU_file = os.path.join(facial_expression_folder, input_AU_file)
            #print(path_input_AU_file)
            # Check if the input file exists
            if not os.path.exists(path_input_AU_file):
                #print(f"Input file for AU {path_input_AU_file} not found. Skipping {name} of pp{participant_id}.")
                continue
        else:
            #print(f"Input file {input_AU_file} not found in all files. Skipping {name} of pp{participant_id}.")
            continue

        input_w_file = f'{name}_AU_data_w.csv'
        #print(input_w_file)
        if input_w_file in w_files:
            path_input_w_file = os.path.join(w_folder, input_w_file)

        # Check if the input w file exists
        if not os.path.exists(path_input_w_file):
            #print(f"Input file for w {path_input_w_file} not found. Skipping {name} of pp{participant_id}.")
            continue

        # Load the CSV file
        AU_df = pd.read_csv(path_input_AU_file)
        au_columns = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                           ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                           ' AU25_r', ' AU26_r', ' AU45_r']
        AU_df = AU_df[au_columns]
        #print(AU_df.shape)
                # Transpose the DataFrame to DxT format
        w = pd.read_csv(path_input_w_file)
        #print(w.shape)

        # generate y - dot product of AU (DxT) and w (vector)
        y = np.dot(AU_df,w)

        # Generate output file and path
        output_csv_corrca_y = f"pp{participant_id}_{role}_{name}_output_corrca_y.csv"
        path_output_csv_corrca_y = os.path.join(facial_expression_folder, output_csv_corrca_y)
        #print(f'Output file for NaN recovery: {output_csv_AU_recovered}')

        # if os.path.exists(path_output_csv_corrca_y):
        #     #print(f"Output file {path_output_csv_corrca_y} already exists. Skipping recovery of {name} for pp{participant_id}")
        #     continue
        
        # Save w to a CSV file in the corrCA folder
        y_df = pd.DataFrame(y, columns=['w'])
        y_df.to_csv(path_output_csv_corrca_y, index=False)
        print(f"CorrCA y saved to {path_output_csv_corrca_y} of {participant_id}")






        

      