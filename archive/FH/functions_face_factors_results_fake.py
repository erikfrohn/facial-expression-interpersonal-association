

import pandas as pd
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def compute_results_factors_fake(path_to_new_data, path_to_result_folder, participant_id1, participant_id2):
    # Find the folder containing participant_id1
    team_folder1 = next((folder for folder in os.listdir(path_to_new_data) if f'{participant_id1}' in folder), None)
    #print(team_folder1)
    team_folder2 = next((folder for folder in os.listdir(path_to_new_data) if f'{participant_id2}' in folder), None)
    #print(team_folder2)
    path_to_team_folder1 = os.path.join(path_to_new_data, f'{team_folder1}') #, "analysis")
    path_to_team_folder2 = os.path.join(path_to_new_data, f'{team_folder2}')#, "analysis")
    path_to_analysis_folder1 = os.path.join(path_to_team_folder1)
    path_to_analysis_folder2 = os.path.join(path_to_team_folder2)



    
    role1 = 'navigator' if int(participant_id1) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    role2 = 'navigator' if int(participant_id2) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even)

    # Define the path to the facial_expression folder within the analysis folder
    facial_expression_folder1 = os.path.join(path_to_analysis_folder1, 'facial_expression')
    facial_expression_folder2 = os.path.join(path_to_analysis_folder2, 'facial_expression')

    # List all files in the analysis folders
    all_files1 = os.listdir(facial_expression_folder1)
    all_files2 = os.listdir(facial_expression_folder2)


    names =  ['instructional_video_0', 'discussion_phase_0', 'discussion_phase_1']
    # Iterate over all rows in the CSV file for participant 1
    for name in names:
        path_to_factor_folder = os.path.join(path_to_result_folder, 'Factor')
        os.makedirs(path_to_factor_folder, exist_ok=True)
        output_csv_path = os.path.join(path_to_factor_folder, f"pp{participant_id1}_pp{participant_id2}_fake_{name}_result_factor.csv")
       
        # Initialize variables to store CSV data
        csv_factors1 = None
        csv_factors2 = None

        # Find the correct input files for both participants
        input_file1 = f"pp{participant_id1}_{role1}_{name}_output_factors_no_black.csv"
        input_file2 = f"pp{participant_id2}_{role2}_{name}_output_factors_no_black.csv"
        
        if input_file1 in all_files1:
            #print(f'Input file for factor result - pp1: {input_file1}')
            path_input_AU_factors_file1 = os.path.join(facial_expression_folder1, input_file1)
            # Load the CSV file
            csv_factors1 = pd.read_csv(path_input_AU_factors_file1)

        if input_file2 in all_files2:
            #print(f'Input file for factor result - pp2: {input_file2}')
            path_input_AU_factors_file2 = os.path.join(facial_expression_folder2, input_file2)
            # Load the CSV file
            csv_factors2 = pd.read_csv(path_input_AU_factors_file2)   
        # Check if both CSV files are loaded after the loop
        if csv_factors1 is None or csv_factors2 is None:
            print(f"Error: Could not find the required CSV files of {name} for both participants. Skipping this pair: pp{participant_id1}-pp{participant_id2}.")
            continue

        # Check if the lengths of both CSV files are the same, and pad with zeros if necessary
        len_diff = len(csv_factors1) - len(csv_factors2)
        if len_diff > 0:
            padding = pd.DataFrame(0, index=np.arange(len_diff), columns=csv_factors2.columns)
            csv_factors2 = pd.concat([csv_factors2, padding], ignore_index=True)
        elif len_diff < 0:
            padding = pd.DataFrame(0, index=np.arange(-len_diff), columns=csv_factors1.columns)
            csv_factors1 = pd.concat([csv_factors1, padding], ignore_index=True)


        # Extract factors from CSV files (excluding the 'frame' column)
        factors1 = csv_factors1.iloc[:, 1:].values
        factors2 = csv_factors2.iloc[:, 1:].values

        # Initialize results list
        results = []

        # Iterate over each factor (column)
        for i in range(factors1.shape[1]):
            f_pil = factors1[:, i]
            f_nav = factors2[:, i]
            label = f'f{i+1}'

            # Compute DTW 
            distance_pp, path = fastdtw(f_pil.reshape(-1, 1), f_nav.reshape(-1, 1), dist=euclidean)
            # Normalize distance_pp
            distance_pp /= (len(f_pil)+len(f_nav))
            
            # Calculate overall Pearson correlation for the entire series
            pearson_correlation_pp = np.corrcoef(f_nav, f_pil)[0, 1]

            # Append results as a dictionary or list
            results.append({
                'Factor': label,
                'Pearson_Correlation': pearson_correlation_pp,
                'DTW distance': distance_pp
            })

            # Plot the factors over time
            plt.figure(figsize=(12, 6))
            plt.plot(f_pil, label=f'Participant {participant_id1}', color='b')
            plt.plot(f_nav, label=f'Participant {participant_id2}', color='r')
            plt.legend()
            plt.title(f'Factor Comparison: {label} of {name} (Participant {participant_id1} & Participant {participant_id2})')
            plt.xlabel('Time Points')
            plt.ylabel('Factor Value')
            # Save the plot to a file in the specified folder
            plot_filename = f'Factor Comparison_{label}_of_{name}_{participant_id1}_{participant_id2}_fake.png'
            path_to_figures_folder = os.path.join(path_to_factor_folder,'figures')
            os.makedirs(path_to_figures_folder, exist_ok=True)
            plot_filepath = os.path.join(path_to_figures_folder, plot_filename)
            #plot_filepath = os.path.join(path_to_result_folder, plot_filename)
            plt.savefig(plot_filepath)
            #plt.show()
            plt.close()


        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        # Check if the output CSV file already exists
        if os.path.exists(output_csv_path):
            #print(f"Output file {output_csv_path} already exists. Skipping calculations.. ")
            continue
        results_df.to_csv(output_csv_path, index=False)
        print(f"Factors results saved to {output_csv_path} - fake - pp{participant_id1}-pp{participant_id2}")