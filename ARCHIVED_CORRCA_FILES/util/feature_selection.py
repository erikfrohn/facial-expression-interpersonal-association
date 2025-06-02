import pandas as pd
import numpy as np
import os
import util.CorrCA as CorrCA
from sklearn.preprocessing import MinMaxScaler

def au_to_factors(df):
    # Combine action units into factors
    f1 = np.round(df[' AU06_r'].values + df[' AU07_r'].values + df[' AU12_r'].values ,2)
    f2 = np.round(df[' AU01_r'].values + df[' AU02_r'].values, 2)
    f3 = np.round(df[' AU20_r'].values + df[' AU25_r'].values + df[' AU26_r'].values, 2)
    f4 = np.round(df[' AU14_r'].values + df[' AU17_r'].values + df[' AU23_r'].values, 2)
    f5 = np.round(df[' AU04_r'].values + df[' AU07_r'].values + df[' AU09_r'].values, 2)
    f6 = np.round(df[' AU10_r'].values + df[' AU15_r'].values + df[' AU17_r'].values, 2)

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

def corrCA_weights(df_nav, df_pil, number_of_components=3):
    number_of_components = min(number_of_components, 17)
    # extract needed columns
    df_nav = df_nav[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                          ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                          ' AU25_r', ' AU26_r', ' AU45_r']]
    df_pil = df_pil[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                          ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                          ' AU25_r', ' AU26_r', ' AU45_r']]
    
    # reshape for the corrca function and round
    df_nav = np.round(df_nav.T.values,2)
    df_pil = np.round(df_pil.T.values,2)
       
    # Stack the data along the first dimension
    pair_data = np.stack([df_nav, df_pil], axis=0)  # Shape: (2, 17, 100)

    # Apply CorrCA to the pair
    W, ISC, _ = CorrCA.fit(pair_data)

    ## start output creation
    # first the weights and the inter-subject
    cols = [f"w{i}" for i in range(number_of_components)]
    corrCA_df = pd.DataFrame(W[:,:number_of_components], columns = cols) 
    corrCA_df['isc'] = ISC

    return corrCA_df

def make_equal_length(file_location, df_nav, df_pil):
    if len(df_nav) != len(df_pil):
        min_length = min(len(df_nav), len(df_pil))
        df_nav = df_nav.iloc[:min_length]
        df_pil = df_pil.iloc[:min_length]
        print(f"files of pairs in {file_location} do not have the same amount of datapoints - temporary fix has made them equal length")
    return df_nav, df_pil


def apply_corrCA_weights(au_data, w, number_of_components=3):
    frames = au_data['frame'] # extract frames
    au_data = au_data[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                          ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                          ' AU25_r', ' AU26_r', ' AU45_r']] # extract only essential action units
    w = w.drop(columns = ['isc']) 
    number_of_components = min(number_of_components, w.shape[1]) # make sure we do not want more components than we initially extracted
    
    # linear projection
    Y = np.dot(au_data, w)  # Shape: (W, T) 
    Y_elements = {'frame' : frames}
    for i in range(1,number_of_components+1):
        Y_elements[f'c{i}'] = Y[:,i-1]
    corrca_df = pd.DataFrame(Y_elements)
    return corrca_df


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def binarize_component(
    component,
    threshold=None,
    percentile=50,
    scale=True
):
    """
    Scale a single component to [0, 1] and binarize based on a threshold or percentile.

    Parameters:
    -----------
    component : np.ndarray, shape (n_timepoints,)
        Time series of a single component.
    threshold : float or None, optional
        Fixed threshold (if provided, overrides percentile).
    percentile : float, default=50
        Percentile to use if threshold=None (e.g., 25, 50, 75).

    Returns:
    --------
    binary_component : np.ndarray, shape (n_timepoints,)
        Binarized component (0 or 1).
    threshold_used : float
        The actual threshold value used.
    """
    # Ensure input is 1D
    component = component.flatten()
    
    # Scale to [0, 1]
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(component.reshape(-1, 1)).flatten()
        
        # Compute threshold
        if threshold is not None:
            threshold_used = threshold
        else:
            threshold_used = np.percentile(scaled, percentile)

        # Binarize
        binary_component = (scaled >= threshold_used).astype(int)
        print(threshold_used)
        return binary_component
    
    binarize_component = (component >= threshold).astype(int)
    return binarize_component