import pandas as pd
import numpy as np
import os
import util.CorrCA as CorrCA

def au_to_factors(df):
    # Combine action units into factors
    f1 = df[' AU06_r'].values + df[' AU07_r'].values + df[' AU12_r'].values
    f2 = df[' AU01_r'].values + df[' AU02_r'].values 
    f3 = df[' AU20_r'].values + df[' AU25_r'].values + df[' AU26_r'].values
    f4 = df[' AU14_r'].values + df[' AU17_r'].values + df[' AU23_r'].values
    f5 = df[' AU04_r'].values + df[' AU07_r'].values + df[' AU09_r'].values
    f6 = df[' AU10_r'].values + df[' AU15_r'].values + df[' AU17_r'].values

    factors_df = pd.DataFrame({
        'frame': df['frame'],
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f4': f4,
        'f5': f5,
        'f6': f6
    })
    
    return factors_df

def corrCA(file_location, df_nav, df_pil):
    df_nav = df_nav[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                          ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                          ' AU25_r', ' AU26_r', ' AU45_r']]
    df_pil = df_pil[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                          ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                          ' AU25_r', ' AU26_r', ' AU45_r']]
    df_nav = np.round(df_nav.T.values,2)
    df_pil = np.round(df_pil.T.values,2)

    if df_nav.size != df_pil.size:
        if df_nav.size < df_pil.size:
            df_pil = df_pil[:,:df_nav.shape[1]]
        else:
            df_nav = df_nav[:,:df_pil.shape[1]]
        print(f"files of pairs in {file_location} do not have the same amount of datapoints - temporary fix has made them equal length")
        
    # Stack the data along the first dimension
    pair_data = np.stack([df_nav, df_pil], axis=0)  # Shape: (2, 17, 100)

    # Apply CorrCA to the pair
    W, ISC, _ = CorrCA.fit(pair_data)

    # W contains the weights for the 17 AUs
    #print("Weights (W):", W)
    #print("Inter-Subject Correlation (ISC):", ISC)
    
    corrCA_df = pd.DataFrame(W[:,0], columns=['w']) # TODO: check whether you want multiple components!
    corrCA_df['isc'] = ISC
    corrCA_df.to_csv(os.path.join(file_location, "corrca.csv"), index=False)

    # Transform the data using weights
    pair_data = np.transpose(pair_data, (0, 2, 1))
    Y = np.dot(pair_data, W)  # Shape: (2, 17, T)

    # Extract the first component for each participant
    participant1_component = Y[0, :, 0]  # Shape: (T,) # TODO: check whether you want multiple components!
    participant2_component = Y[1, :, 0]  # Shape: (T,)

    # Create a DataFrame for the first component
    first_component_df = pd.DataFrame({
        'time': range(len(participant1_component)),  # Time points
        'navigator': participant1_component,      # First component for Participant 1
        'pilot': participant2_component       # First component for Participant 2
    })

    # Save the DataFrame to a CSV file
    first_component_df.to_csv(os.path.join(file_location, "corrca_component1.csv"), index=False)
