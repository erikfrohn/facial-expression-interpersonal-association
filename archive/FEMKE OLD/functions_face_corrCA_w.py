import os
import pandas as pd
import numpy as np
from CorrCA import fit, calc_corrca
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_apply_corrca(path_to_new_data):
    # Define the path to the combined CSV file
    path_to_result_folder = os.path.join(path_to_new_data, 'Results facial expressions')
    combined_csv_path = os.path.join(path_to_result_folder, "corrCA") 
    # List all CSV files in the corrCA folder
    csv_files = [f for f in os.listdir(combined_csv_path) if f.endswith('.csv') and 'combined' in f] 
    #csv_files = [f for f in os.listdir(combined_csv_path) if f.endswith('.npy') and 'combined' in f] 

    for csv_file in csv_files:
        base_name = os.path.splitext(csv_file)[0].replace("_combined", "")
        file_path = os.path.join(combined_csv_path, csv_file)
        combined_df = pd.read_csv(file_path) #shape is T(NxD)
        #combined_df = np.load(file_path)

         # Ensure the data is reshaped correctly
        data_array = combined_df.values
        #print("Original shape:", data_array.shape)
        # Round the data to two decimal places
        data_array = np.round(data_array, 2)
        # Check for inf and NaN values
        if not np.isfinite(data_array).all():
            #print(f"Data in {csv_file} contains inf or NaN values. Cleaning data...")
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)


         # Define the Action Units you want to use (intensity values, ending in '_r')
        au_columns = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                          ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                          ' AU25_r', ' AU26_r', ' AU45_r']
        # Assuming D is 17 and N is the number of participants
        D = len(au_columns)
        N = data_array.shape[0] // D
        T = data_array.shape[1]
        data_array = data_array.reshape(N, D, T)
    
        if not np.isfinite(data_array).all():
            #print(f"Data in {csv_file} contains inf or NaN values. Cleaning data...")
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        # else:
        #     print(f'No inf/NaN in reshaped data_array of {base_name}')

        #apply corrCA 
        W = fit(data_array)
        W = W[0] # to make from a tuple a 2D array

        if W.ndim ==2: 
            # Plot the correlation weights matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(W, cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
            plt.title(f'CorrCA Correlation Weights - {base_name}')
            plt.xlabel('Component')
            plt.ylabel('Features')
            plt.savefig(os.path.join(combined_csv_path, f"CorrCA Correlation Weights - {base_name}"))
            #plt.show()
            plt.close()

            # get vector w from weight matrix 
            w = W[:,0] + W[:,1] + W[:,2]

            # Save w to a CSV file in the corrCA folder
            output_w_csv_path = os.path.join(combined_csv_path, f"{base_name}_w.csv")
            w_df = pd.DataFrame(w, columns=['w'])
            w_df.to_csv(output_w_csv_path, index=False)
            print(f"CorrCA w saved to {output_w_csv_path}")
       
