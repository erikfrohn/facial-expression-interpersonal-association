import pandas as pd
import numpy as np
import os
import util.CorrCA as CorrCA

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

def corrCA_weights(file_location, df_nav, df_pil):
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

    # TODO: this should not be needed and should be fixed in pre
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

    ## start output creation
    # first the weights and the inter-subject
    corrCA_df = pd.DataFrame(W[:,0], columns=['w']) # TODO: check whether you want multiple components!
    corrCA_df['isc'] = ISC
    corrCA_df.to_csv(os.path.join(file_location, "corrca_weights.csv"), index=False)

def apply_corrCA_weights(au_data, w):
    frames = au_data['frame']
    au_data = au_data[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r',
                          ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r',
                          ' AU25_r', ' AU26_r', ' AU45_r']]
    # Transform the data using weights
    Y = np.dot(au_data, w['w'])  # Shape: (W, T) now using W=1 so not interesting 

    corrca_df = pd.DataFrame({
        'frame': frames,
        'component1': Y[0]
    }
    )
    return corrca_df