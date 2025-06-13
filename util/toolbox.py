import pandas as pd
import os
import numpy as np
import util.correlation_measure as cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

LOCATION = "data"
FEATURE_FOLDER = "features"
PHASES = [f'{name}_{i}' for name, num in  [("instructional_video", 1), ("discussion_phase", 2), ('reschu_run',8)] for i in range(num)]
FACTORS = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

FACTOR_LABELS = {
    'f1': "Enjoyment Smile",
    'f2': "Eyebrows Up",
    'f3': "Mouth Open",
    'f4': "Mouth Tightening",
    'f5': "Eye Tightening",
    'f6': "Mouth Frown"
}

def crqa_radius_gridsearch(name = "gridsearch", radii = [0.1, 0.2, 0.3, 0.4, 0.5], AVAILABLE_PAIRS = ['05_06', '07_08', '09_10', '103_104', '27_28', '83_84', '85_86', '87_88', '91_92', '93_94', '95_96', '97_98'] ):
    # Initialize DataFrame to store all results
    results_df = pd.DataFrame(columns=[
        'pair', 'phase', 'method', 'factor', 'radius', 
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
            'factor': component,
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
            for pair in AVAILABLE_PAIRS:
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

    # Process surrogate pairs - Factors
    print("\nSURROGATE PAIRS - FACTORS\n")
    index_real = np.arange(len(AVAILABLE_PAIRS))
    index_fake = np.append(np.arange(1,len(AVAILABLE_PAIRS)),0)

    for f in FACTORS:
        print(f"Processing factor {f} for surrogate pairs")
        for r in radii:
            for i in range(len(AVAILABLE_PAIRS)):
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

    # Save results to CSV
    results_df.to_csv(f'{name}.csv', index=False)
    print("Processing complete. Results saved to crqa_results_all_pairs.csv")

    # Display sample of the results
    print("\nSample of the results DataFrame:")
    print(results_df.head())



# THIS IS ONLY FOR REAL DATA GRIDSEARCH (SEE ARCHIVE FOR THE REAL / FAKE COMPARISON GRIDSEARCH)
def plot_crqa_radius_gridsearch(results_df, name='gridsearch'):

    # Set up plot grid
    fig = plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid", font_scale=1.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Create 3x3 grid (3 CORRCA components + 6 factors)
    axes = fig.subplots(2, 3)

    # Common plotting function
    def plot_component(ax, data, title):
        # Calculate statistics
        agg_data = data.groupby(['radius', 'non_event_matches'])['RR'].agg(['mean', 'std', 'count']).reset_index()
        agg_data['ci'] = 1.96 * agg_data['std'] / np.sqrt(agg_data['count'])
        
        # Plot lines with error bands
        sns.lineplot(
            data=agg_data,
            x='radius',
            y='mean',
            #hue='condition',
            style='non_event_matches',
            #palette={'real': '#1f77b4', 'fake': '#ff7f0e'},
            style_order=['included', 'excluded'],
            markers=True,
            dashes=[(1,0), (2,2)],
            markersize=6,
            linewidth=1.5,
            ax=ax
        )
        
        # Add error bands

        for match in ['included', 'excluded']:
            subset = agg_data[
                            (agg_data['non_event_matches'] == match)]
            ax.fill_between(
                subset['radius'],
                subset['mean'] - subset['ci'],
                subset['mean'] + subset['ci'],
                alpha=0.2,
                color='#1f77b4'
            )
        # Add reference lines
        ax.axhline(y=0.02, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(y=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        # Add text labels for reference lines
        ax.text(0.5, 0.021, 'RR=0.02', color='gray', ha='center', va='bottom', fontsize=9)
        ax.text(0.5, 0.051, 'RR=0.05', color='gray', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Radius', labelpad=8, fontsize=12)
        ax.set_ylabel('RR ± 95% CI', labelpad=8, fontsize=12)
        ax.set_ylim(-0.05, 0.4)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first plot
        if ax == axes[0,2]:
            ax.legend().remove()
        else:
            ax.get_legend().remove()

    # Plot Facial Factors
    for idx, factor in enumerate(['f1', 'f2', 'f3', 'f4', 'f5', 'f6']):
        ax_row = idx//3
        ax_col = idx%3
        ax = axes[ax_row, ax_col]
        
        factor_data = results_df[results_df['factor'] == factor]
        plot_component(ax, factor_data, f'{factor}: {FACTOR_LABELS[factor]}')
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title='NEM included/excluded', loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=6)
    plt.tight_layout()
    fig.suptitle("Factor radius gridsearch (goal RR = 2-5%)", fontsize=24, y=1.10)
    plt.savefig(f'{name}.png', dpi='figure', bbox_inches='tight')
    plt.show()

def factor_gridsearch(AVAILABLE_PAIRS, name = "gridsearch"):
    # Initialize DataFrame to store all results
    results_df = pd.DataFrame(columns=[
        'pair', 'phase', 'factor', 
        'condition', 'value'
    ])

    def process_analysis(p1_df, p2_df, component, pair, phase, condition):
        value = np.mean(p1_df) + np.mean(p2_df)

        
        new_row = {
            'pair': pair,
            'phase': phase,
            'factor': component,
            'condition': condition,
            'value': value
        }
        
        return new_row

    # Process real pairs - Factors
    print("\nREAL PAIRS - FACTORS\n")
    for f in FACTORS:
        print(f"Processing factor {f}")
        for pair in AVAILABLE_PAIRS:
            p1, p2 = pair.split("_")
            for phase in PHASES:
                p1_loc = os.path.join(LOCATION, pair, FEATURE_FOLDER, f'pp{p1}_{phase}_factors.csv')
                p2_loc = os.path.join(LOCATION, pair, FEATURE_FOLDER, f'pp{p2}_{phase}_factors.csv')
                
                if os.path.exists(p1_loc) and os.path.exists(p2_loc):
                    p1_df = pd.read_csv(p1_loc)
                    p2_df = pd.read_csv(p2_loc)
                    p1_values = p1_df[f].values
                    p2_values = p2_df[f].values
                    # With non-event matches
                    new_row = process_analysis(
                        p1_values, p2_values, f, pair, phase, 'real'
                    )
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    

    # Process surrogate pairs - Factors
    print("\nSURROGATE PAIRS - FACTORS\n")
    index_real = np.arange(len(AVAILABLE_PAIRS))
    index_fake = np.append(np.arange(1,len(AVAILABLE_PAIRS)),0)

    for f in FACTORS:
        print(f"Processing factor {f} for surrogate pairs")
        for i in range(len(AVAILABLE_PAIRS)):
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
                    p1_values = p1_df[f].values
                    p2_values = p2_df[f].values
                    # With non-event matches
                    new_row = process_analysis(
                        p1_values, p2_values, f, pair, phase, 'fake'
                    )
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    
 
    # Save results to CSV
    results_df.to_csv(f'{name}.csv', index=False)
    print(f"Processing complete. Results saved to {name}.csv")

    # Display sample of the results
    print("\nSample of the results DataFrame:")
    print(results_df.head())