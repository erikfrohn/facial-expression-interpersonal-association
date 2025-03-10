#combines the action units of the participants into factors 
import pandas as pd
import numpy as np
import os

def get_factors(path_to_analysis_folder, participant_id):
    role = 'navigator' if int(participant_id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even

    # Determine the CSV file name with markers
    csv_filename = f"pp{participant_id}_{role}_video_frames.csv"
    path_to_frame_csv_file = os.path.join(path_to_analysis_folder,'Video', csv_filename)
    frame_data_csv = pd.read_csv(path_to_frame_csv_file)


    # Define the path to the facial_expression folder within the analysis folder
    facial_expression_folder = os.path.join(path_to_analysis_folder, 'facial_expression')
    # List all files in the analysis folder
    all_files = os.listdir(facial_expression_folder)
    

    # Iterate over all rows in the CSV file
    for index, row in frame_data_csv.iterrows():
        name = row['name']
        input_AU_recovered_file = f"pp{participant_id}_{role}_{name}_output_au_no_black_recovered.csv"
        
        if input_AU_recovered_file in all_files:
            #print(f'Input file for AU generation: {input_AU_recovered_file}')
            path_input_AU_recovered_file = os.path.join(facial_expression_folder, input_AU_recovered_file)
            
            # Check if the input file exists
            if not os.path.exists(path_input_AU_recovered_file):
                #print(f"Input file {path_input_AU_recovered_file} not found. Skipping making factors of {name} for pp{participant_id}.")
                continue
            
            # Load the CSV file
            csv_AU_recovered = pd.read_csv(path_input_AU_recovered_file)

            # Generate output file and path
            output_csv_factors = f"pp{participant_id}_{role}_{name}_output_factors_no_black.csv"
            path_output_csv_factors = os.path.join(facial_expression_folder, output_csv_factors)
            #print(f'Output file for factors: {output_csv_factors}')

            if os.path.exists(path_output_csv_factors):
                #print(f"Output file {output_csv_factors} already exists. Skipping {name} of pp{participant_id}")
                continue

            # Combine action units into factors
            f1 = csv_AU_recovered[' AU06_r'].values + csv_AU_recovered[' AU07_r'].values + csv_AU_recovered[' AU12_r'].values
            f2 = csv_AU_recovered[' AU01_r'].values + csv_AU_recovered[' AU02_r'].values 
            f3 = csv_AU_recovered[' AU20_r'].values + csv_AU_recovered[' AU25_r'].values + csv_AU_recovered[' AU26_r'].values
            f4 = csv_AU_recovered[' AU14_r'].values + csv_AU_recovered[' AU17_r'].values + csv_AU_recovered[' AU23_r'].values
            f5 = csv_AU_recovered[' AU04_r'].values + csv_AU_recovered[' AU07_r'].values + csv_AU_recovered[' AU09_r'].values
            f6 = csv_AU_recovered[' AU10_r'].values + csv_AU_recovered[' AU15_r'].values + csv_AU_recovered[' AU17_r'].values

            factors_df = pd.DataFrame({
                'frame': csv_AU_recovered['frame'],
                'f1': f1,
                'f2': f2,
                'f3': f3,
                'f4': f4,
                'f5': f5,
                'f6': f6
            })

            factors_df.to_csv(path_output_csv_factors, index=False)
            print(f"Factors of pp{participant_id} saved to {path_output_csv_factors}")
