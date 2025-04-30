import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# Configuration
LOCATION = "data"
FEATURE_FOLDER = "features"
PHASES = [f'{name}_{i}' for name, num in [("instructional_video", 1), ("discussion_phase", 2), ('reschu_run',8)] for i in range(num)]
FACTORS = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
COMPONENTS = ['c1', 'c2', 'c3']
N_SPLITS = 5  # For cross-validation

def crqa_lag_analysis(p1, p2, radius, **kwargs):
    """Modified CRQA analysis with fixed radius"""
    # [Keep all your original preprocessing code from crqa_lag_analysis]
    
    # Calculate distances and RR
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p1 = p1[:len(p2)]
    p2 = p2[:len(p1)]
    distances = np.linalg.norm(p1 - p2, axis=-1)
    valid_distances = distances[np.isfinite(distances)]
    rr = np.mean(valid_distances < radius)
    
    return {
        'RR': rr,
        'radius': radius
    }

def load_pair_data(pair, analysis_type, phase):
    """Load data for a pair and analysis type"""
    p1, p2 = pair.split('_')
    base_path = os.path.join(LOCATION, pair, FEATURE_FOLDER)
    
    if analysis_type == 'corrca':
        file1 = f'pp{p1}_{phase}_corrca.csv'
        file2 = f'pp{p2}_{phase}_corrca.csv'
    else:
        file1 = f'pp{p1}_{phase}_factors.csv'
        file2 = f'pp{p2}_{phase}_factors.csv'
        
    try:
        df1 = pd.read_csv(os.path.join(base_path, file1))
        df2 = pd.read_csv(os.path.join(base_path, file2))
        return df1, df2
    except FileNotFoundError:
        return None, None

def calculate_component_rr(pairs, analysis_type, component, radius):
    """Calculate mean RR for real pairs and fake pairs"""
    real_rr, fake_rr = [], []
    
    # Real pairs
    for pair in pairs:
        for phase in PHASES:
            df1, df2 = load_pair_data(pair, analysis_type, phase)
            if df1 is not None and df2 is not None:
                result = crqa_lag_analysis(df1[component].values, df2[component].values, radius)
                real_rr.append(result['RR'])
    
    # Fake pairs (surrogate combinations)
    index_real = np.arange(len(pairs))
    index_fake = np.append(np.arange(1,len(pairs)),0)
    
    for i in range(len(pairs)):
        pair1 = pairs[index_real[i]]
        pair2 = pairs[index_fake[i]]
        for phase in PHASES:
            df1, _ = load_pair_data(pair1, analysis_type, phase)
            _, df2 = load_pair_data(pair2, analysis_type, phase)
            if df1 is not None and df2 is not None:
                result = crqa_lag_analysis(df1[component].values, df2[component].values, radius)
                fake_rr.append(result['RR'])
    
    return np.nanmean(real_rr), np.nanmean(fake_rr)

def find_global_optimal_radii(AVAILABLE_PAIRS):
    """Find optimal radii for each component using cross-validation"""
    optimal_radii = {
        'corrca': {c: None for c in COMPONENTS},
        'factor': {f: None for f in FACTORS}
    }
    
    candidate_radii = np.linspace(0.1, 2.0, 50)
    kf = KFold(n_splits=2, shuffle=True)
    
    # Optimize CORRCA components
    for component in COMPONENTS:
        print(f"\nOptimizing CORRCA component {component}")
        best_score = -np.inf
        best_radius = 0.5  # Default
        
        for radius in candidate_radii:
            fold_scores = []
            
            for train_idx, test_idx in kf.split(AVAILABLE_PAIRS):
                train_pairs = [AVAILABLE_PAIRS[i] for i in train_idx]
                test_pairs = [AVAILABLE_PAIRS[i] for i in test_idx]
                
                # Train score (real RR on training pairs)
                train_real_rr, _ = calculate_component_rr(train_pairs, 'corrca', component, radius)
                
                # Test contrast (real - fake RR on test pairs)
                test_real_rr, test_fake_rr = calculate_component_rr(test_pairs, 'corrca', component, radius)
                fold_score = (train_real_rr * 0.8) + (test_real_rr - test_fake_rr) * 0.2
                
                fold_scores.append(fold_score)
            
            mean_score = np.nanmean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_radius = radius
                
        optimal_radii['corrca'][component] = best_radius
        print(f"{component}: Optimal radius {best_radius:.3f} (Score: {best_score:.3f})")
    
    # Optimize Facial Factors
    for factor in FACTORS:
        print(f"\nOptimizing Facial Factor {factor}")
        best_score = -np.inf
        best_radius = 0.5  # Default
        
        for radius in candidate_radii:
            fold_scores = []
            
            for train_idx, test_idx in kf.split(AVAILABLE_PAIRS):
                train_pairs = [AVAILABLE_PAIRS[i] for i in train_idx]
                test_pairs = [AVAILABLE_PAIRS[i] for i in test_idx]
                
                # Train score (real RR on training pairs)
                train_real_rr, _ = calculate_component_rr(train_pairs, 'factor', factor, radius)
                
                # Test contrast (real - fake RR on test pairs)
                test_real_rr, test_fake_rr = calculate_component_rr(test_pairs, 'factor', factor, radius)
                fold_score = (train_real_rr * 0.8) + (test_real_rr - test_fake_rr) * 0.2
                
                fold_scores.append(fold_score)
            
            mean_score = np.nanmean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_radius = radius
                
        optimal_radii['factor'][factor] = best_radius
        print(f"{factor}: Optimal radius {best_radius:.3f} (Score: {best_score:.3f})")
    
    return optimal_radii

def plot_optimal_radii(optimal_radii):
    """Visualize the optimal radii"""
    plt.figure(figsize=(14, 6))
    
    # CORRCA components
    plt.subplot(1, 2, 1)
    components = list(optimal_radii['corrca'].keys())
    radii = list(optimal_radii['corrca'].values())
    plt.bar(components, radii, color='skyblue')
    plt.title('Optimal Radii - CORRCA Components')
    plt.ylabel('Radius')
    plt.ylim(0, 2)
    
    # Facial factors
    plt.subplot(1, 2, 2)
    factors = list(optimal_radii['factor'].keys())
    radii = list(optimal_radii['factor'].values())
    plt.bar(factors, radii, color='salmon')
    plt.title('Optimal Radii - Facial Factors')
    plt.ylabel('Radius')
    plt.ylim(0, 2)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    AVAILABLE_PAIRS = ['05_06', '07_08']#, '09_10', '99_100', '101_102', '103_104', '13_14', '17_18', '19_20', '21_22', '25_26', '27_28', '29_30', '31_32', '33_34', '35_36', '37_38', '39_40', '41_42', '43_44', '45_46', '47_48', '49_50', '51_52', '59_60', '61_62', '65_66', '67_68', '69_70', '71_72', '73_74', '75_76', '77_78', '79_80', '81_82', '83_84', '85_86', '87_88', '91_92', '93_94', '95_96', '97_98']
    
    # Step 1: Find optimal radii
    optimal_radii = find_global_optimal_radii(AVAILABLE_PAIRS)
    
    # Step 2: Visualize results
    plot_optimal_radii(optimal_radii)
    
    # Save results
    pd.DataFrame({
        'component': list(optimal_radii['corrca'].keys()) + list(optimal_radii['factor'].keys()),
        'type': ['corrca']*3 + ['factor']*6,
        'optimal_radius': list(optimal_radii['corrca'].values()) + list(optimal_radii['factor'].values())
    }).to_csv('global_optimal_radii.csv', index=False)