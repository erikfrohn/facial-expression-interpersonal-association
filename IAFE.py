import glob
import os
import numpy as np
import pandas as pd
import sys
#import util.CorrCA as corrca
import util.correlation_measure as cm
import util.feature_extraction as fe
import util.feature_selection as fs
import util.video_transformation as vt


if len(sys.argv) <= 1:
    print("No input parameter provided.")
else:
    pair = sys.argv[1]  # The first argument is at index 1
    print(f"Input parameter received: {pair}")

    # assumes all video's have been preprocessed and turned into action units. 
    data_loc = f'data-out/{pair}'
    au_loc = f'data-out/{pair}/au'
    feat_loc = f'data-out/{pair}/selection'
    corr_loc = f'data-out/{pair}/extraction'

    sets = ['corrca', 'factors']
    phases = ['instructional_video_0', 'discussion_phase_0', 'discussion_phase_1'] #add RESCHU phases
    factors = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

    os.makedirs(data_loc, exist_ok=True)
    os.makedirs(au_loc, exist_ok=True)
    os.makedirs(feat_loc, exist_ok=True)
    os.makedirs(corr_loc, exist_ok=True)

    print(f"Calculating Correlated Component Analysis Weights for pair {pair}")

    data = {}
    nav, pil = pair.split("_")
    nav_df = pd.DataFrame()
    pil_df = pd.DataFrame()
    
    # add all phases to a single dataframe
    for file in os.listdir(au_loc):
        if ".csv" in file: # skip .txt files

            df = pd.read_csv(os.path.join(au_loc, file))
            if ".csv" in file and nav in file:
                nav_df = pd.concat([nav_df, df])
            if ".csv" in file and pil in file:
                pil_df = pd.concat([pil_df, df])

    nav_df, pil_df = fs.make_equal_length(pair, nav_df, pil_df)
    w = fs.corrCA_weights(nav_df, pil_df)
    w.to_csv(os.path.join(data_loc, f"{pair}_corrca_weights.csv"), index=False)
    

    print(f"Extracting features for pair {pair}")
    
    for file in os.listdir(au_loc):
        if ".csv" in file: 
            filename = os.path.join(au_loc, file)
            participant, _ = file.split("_",1)
            df = pd.read_csv(filename)
            for name in phases:
                if name in file:
                    factors = fs.au_to_factors(df)
                    factors.to_csv(os.path.join(feat_loc, f"{participant}_{name}_factors.csv"), index=False)

                    w = pd.read_csv(os.path.join(data_loc, f'{pair}_corrca_weights.csv'))
                    corrca = fs.apply_corrCA_weights(df, w)
                    corrca.to_csv(os.path.join(feat_loc, f"{participant}_{name}_corrca.csv"), index=False)
                    continue
    

    print("hahahahaha ik zie je denken die is gek")