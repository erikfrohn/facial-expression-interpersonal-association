import pandas as pd
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def compute_results_corrCA(path_to_analysis_folder, path_to_result_folder, participant_id1, participant_id2):

 # Determine the CSV file name with markers
    role1 = 'navigator' if int(participant_id1) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    role2 = 'navigator' if int(participant_id2) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    
    # Define the path to the facial_expression folder within the analysis folder
    facial_expression_folder1 = os.path.join(path_to_analysis_folder, 'facial_expression')
    facial_expression_folder2 = os.path.join(path_to_analysis_folder, 'facial_expression')

    # List all files in the analysis folders
    all_files1 = os.listdir(facial_expression_folder1)
    all_files2 = os.listdir(facial_expression_folder2)

    names =  ['instructional_video_0', 'discussion_phase_0', 'discussion_phase_1']
    # Iterate over all rows in the CSV file for participant 1
    for name in names:
        # Generate output CSV file path
        output_csv_path = os.path.join(path_to_result_folder, 'corrCA', f"pp{participant_id1}_pp{participant_id2}_{name}_result_corrCA.csv")
        # Check if the output CSV file already exists
        if os.path.exists(output_csv_path):
            #print(f"Output file {output_csv_path} already exists. Skipping calculations.. ")
            continue
        
        # Initialize variables to store CSV data
        csv_y1 = None
        csv_y2 = None

        # Find the correct input files for both participants
        input_file1 = f"pp{participant_id1}_{role1}_{name}_output_corrca_y.csv"
        input_file2 = f"pp{participant_id2}_{role2}_{name}_output_corrca_y.csv"
        
        if input_file1 in all_files1:
            #print(f'Input file for factor result - pp1: {input_file1}')
            path_input_AU_y_file1 = os.path.join(facial_expression_folder1, input_file1)
            # Load the CSV file
            csv_y1 = pd.read_csv(path_input_AU_y_file1)

        if input_file2 in all_files2:
            #print(f'Input file for factor result - pp2: {input_file2}')
            path_input_AU_y_file2 = os.path.join(facial_expression_folder2, input_file2)
            # Load the CSV file
            csv_y2 = pd.read_csv(path_input_AU_y_file2)   
        # Check if both CSV files are loaded after the loop
        if csv_y1 is None or csv_y2 is None:
            print(f"Error: Could not find the required CSV files of {name} for both participants. Skipping {name} of pp{participant_id1}-pp{participant_id2}.")
            continue

        # Check if the lengths of both CSV files are the same, and pad with zeros if necessary
        len_diff = len(csv_y1) - len(csv_y2)
        if len_diff > 0:
            padding = pd.DataFrame(0, index=np.arange(len_diff), columns=csv_y2.columns)
            csv_y2 = pd.concat([csv_y2, padding], ignore_index=True)
        elif len_diff < 0:
            padding = pd.DataFrame(0, index=np.arange(-len_diff), columns=csv_y1.columns)
            csv_y1 = pd.concat([csv_y1, padding], ignore_index=True)

        # Extract y from CSV files
        y1 = csv_y1.values
        y2 = csv_y2.values

        # Initialize results list
        results = []

        # Compute DTW 
        distance_pp, path = fastdtw(y1.reshape(-1, 1), y2.reshape(-1, 1), dist=euclidean)
        # Normalize distance_pp
        distance_pp /= (len(y1)+len(y2))
        # Calculate overall Pearson correlation for the entire series
        pearson_correlation_pp = np.corrcoef(y1[:,0], y2[:,0])[0, 1]

        # Append results as a dictionary or list
        results.append({
            'Pearson_Correlation': pearson_correlation_pp,
            'DTW distance': distance_pp
            })

        # Plot the y over time
        plt.figure(figsize=(12, 6))
        plt.plot(y1, label=f'Participant {participant_id1}', color='b')
        plt.plot(y2, label=f'Participant {participant_id2}', color='r')
        plt.legend()
        plt.title(f'CorrCA y of {name} (Participant {participant_id1} & Participant {participant_id2})')
        plt.xlabel('Time Points')
        plt.ylabel('Intensity')
        path_to_figures_folder = os.path.join(path_to_result_folder, 'corrCA', 'figures')
        #os.makedirs(path_to_figures_folder)
        plt.savefig(os.path.join(path_to_figures_folder, f"pp{participant_id1}_pp{participant_id2}_{name}_plot_corrCA.png"))
        #plt.show()
        plt.close()


        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_path, index=False)
        print(f"CorrCA Results saved to {output_csv_path} of real: pp{participant_id1}-pp{participant_id2}")
