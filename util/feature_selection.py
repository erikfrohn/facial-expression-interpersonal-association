import pandas as pd
import numpy as np

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
