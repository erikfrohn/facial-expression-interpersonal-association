import pandas as pd
import os
import numpy as np
import util.correlation_measure as cm

LOCATION = "data"
FEATURE_FOLDER = "features"
PHASES = [f'{name}_{i}' for name, num in  [("instructional_video", 1), ("discussion_phase", 2), ('reschu_run',8)] for i in range(num)]#, ("reschu_run", 8)] for i in range(num)]
SETS = ['corrca', 'factors']
FACTORS = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
COMPONENTS = ['c1', 'c2', 'c3']
PAIRS = [f'0{i}_0{i+1}' for i in np.arange(1,9,2)]
PAIRS.append("09_10")
PAIRS.extend([f'{i}_{i+1}' for i in np.arange(11,104,2)])
AVAILABLE_PAIRS = ['05_06', '07_08', '09_10', '103_104', '27_28', '83_84', '85_86', '87_88', '91_92', '93_94', '95_96', '97_98']
SKIP_PAIRS = ['53_54', '55_56', "63_64", "89_90"]

EXPERIMENT = True
def crqa_radius_gridsearch(radii = [0.1, 0.2, 0.3, 0.4, 0.5]):
    # Initialize DataFrame to store all results
    results_df = pd.DataFrame(columns=[
        'pair', 'phase', 'method', 'component_factor', 'radius', 
        'non_event_matches', 'condition', 'RR'
    ])

    def process_analysis(p1_df, p2_df, component, radius, remove_non_events, method, pair, phase, condition):
        output = cm.crqa_lag_analysis(
            p1_df[component].values, 
            p2_df[component].values, 
            radius=radius,
            remove_non_event_matches=remove_non_events
        )
        
        new_row = {
            'pair': pair,
            'phase': phase,
            'method': method,
            'component_factor': component,
            'radius': radius,
            'non_event_matches': 'excluded' if remove_non_events else 'included',
            'condition': condition,
            'RR': output['RR']
        }
        
        return new_row

    # Process real pairs - Factors
    print("\nREAL PAIRS - FACTORS\n")
    for f in FACTORS:
        print(f"Processing factor {f}")
        for r in radii:
            for pair in AVAILABLE_PAIRS[:10]:
                p1, p2 = pair.split("_")
                for phase in PHASES:
                    p1_loc = os.path.join(LOCATION, pair, FEATURE_FOLDER, f'pp{p1}_{phase}_factors.csv')
                    p2_loc = os.path.join(LOCATION, pair, FEATURE_FOLDER, f'pp{p2}_{phase}_factors.csv')
                    
                    if os.path.exists(p1_loc) and os.path.exists(p2_loc):
                        p1_df = pd.read_csv(p1_loc)
                        p2_df = pd.read_csv(p2_loc)
                        
                        # With non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, f, r, False, 
                            'factor', pair, phase, 'real'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        # Without non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, f, r, True, 
                            'factor', pair, phase, 'real'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Process real pairs - CORRCA
    print("\nREAL PAIRS - CORRCA\n")
    for c in ['c1', 'c2', 'c3']:
        print(f"Processing component {c}")
        for r in radii:
            for pair in AVAILABLE_PAIRS[:10]:
                p1, p2 = pair.split("_")
                for phase in PHASES:
                    p1_loc = os.path.join(LOCATION, pair, FEATURE_FOLDER, f'pp{p1}_{phase}_corrca.csv')
                    p2_loc = os.path.join(LOCATION, pair, FEATURE_FOLDER, f'pp{p2}_{phase}_corrca.csv')
                    
                    if os.path.exists(p1_loc) and os.path.exists(p2_loc):
                        p1_df = pd.read_csv(p1_loc)
                        p2_df = pd.read_csv(p2_loc)
                        
                        # With non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, c, r, False, 
                            'corrca', pair, phase, 'real'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        # Without non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, c, r, True, 
                            'corrca', pair, phase, 'real'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Process surrogate pairs - Factors
    print("\nSURROGATE PAIRS - FACTORS\n")
    index_real = np.arange(10)
    index_fake = np.append(np.arange(1,10),0)

    for f in FACTORS:
        print(f"Processing factor {f} for surrogate pairs")
        for r in radii:
            for i in range(10):
                pair1 = AVAILABLE_PAIRS[index_real[i]]
                pair2 = AVAILABLE_PAIRS[index_fake[i]]
                p1, _ = pair1.split("_")
                _, p2 = pair2.split("_")
                
                for phase in PHASES:
                    p1_loc = os.path.join(LOCATION, pair1, FEATURE_FOLDER, f'pp{p1}_{phase}_factors.csv')
                    p2_loc = os.path.join(LOCATION, pair2, FEATURE_FOLDER, f'pp{p2}_{phase}_factors.csv')
                    
                    if os.path.exists(p1_loc) and os.path.exists(p2_loc):
                        p1_df = pd.read_csv(p1_loc)
                        p2_df = pd.read_csv(p2_loc)
                        
                        # With non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, f, r, False, 
                            'factor', f"{pair1}_{pair2}", phase, 'fake'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        # Without non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, f, r, True, 
                            'factor', f"{pair1}_{pair2}", phase, 'fake'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Process surrogate pairs - CORRCA
    print("\nSURROGATE PAIRS - CORRCA\n")
    for c in ['c1', 'c2', 'c3']:
        print(f"Processing component {c} for surrogate pairs")
        for r in radii:
            for i in range(10):
                pair1 = AVAILABLE_PAIRS[index_real[i]]
                pair2 = AVAILABLE_PAIRS[index_fake[i]]
                p1, _ = pair1.split("_")
                _, p2 = pair2.split("_")
                
                for phase in PHASES:
                    p1_loc = os.path.join(LOCATION, pair1, FEATURE_FOLDER, f'pp{p1}_{phase}_corrca.csv')
                    p2_loc = os.path.join(LOCATION, pair2, FEATURE_FOLDER, f'pp{p2}_{phase}_corrca.csv')
                    
                    if os.path.exists(p1_loc) and os.path.exists(p2_loc):
                        p1_df = pd.read_csv(p1_loc)
                        p2_df = pd.read_csv(p2_loc)
                        
                        # With non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, c, r, False, 
                            'corrca', f"{pair1}_{pair2}", phase, 'fake'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        # Without non-event matches
                        new_row = process_analysis(
                            p1_df, p2_df, c, r, True, 
                            'corrca', f"{pair1}_{pair2}", phase, 'fake'
                        )
                        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save results to CSV
    results_df.to_csv('crqa_results_all_pairs.csv', index=False)
    print("Processing complete. Results saved to crqa_results_all_pairs.csv")

    # Display sample of the results
    print("\nSample of the results DataFrame:")
    print(results_df.head())