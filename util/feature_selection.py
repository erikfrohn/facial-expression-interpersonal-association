import pandas as pd

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

def corrCA(df):
    print("nee")