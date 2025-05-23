import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

def crqa_lag_analysis(p1, p2, 
                     sampling_rate=60,
                     max_lag_seconds=6,
                     normalize=False,
                     remove_non_event_matches=False,
                     downsample=True,
                     debug=False,
                     radius=2.0):
    # Input validation
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p1 = p1[:len(p2)]
    p2 = p2[:len(p1)]
    if p1.size == 0 or p2.size == 0:
        raise ValueError("Empty input time series")


    # Add singleton dimension if 1D
    if p1.ndim == 1:
        p1 = p1[:, None]
    if p2.ndim == 1:
        p2 = p2[:, None]
    
    # Downsampling   
    if downsample:
        ratio = max(1, sampling_rate // 10)
        def average_bin(ts, ratio):
            n_bins = len(ts) // ratio
            ts = ts[:n_bins * ratio]  # Trim excess
            return ts.reshape(n_bins, ratio, -1).mean(axis=1)
        
        if debug: 
            print(f"Downsampling from {len(p1)} to {len(p1) // ratio} points using mean over bins of size {ratio}")
        p1 = average_bin(p1, ratio)
        p2 = average_bin(p2, ratio)
        sampling_rate = 10

    
    # Normalization with variance check
    if normalize and not remove_non_event_matches:
        for ts in [p1, p2]:
            if np.any(np.std(ts, axis=0) == 0):
                raise ValueError("Constant time series after normalization")
        p1 = (p1 - np.mean(p1, axis=0)) / np.std(p1, axis=0)
        p2 = (p2 - np.mean(p2, axis=0)) / np.std(p2, axis=0)
        if debug:
            print(f"Normalized ranges - P1: [{np.min(p1):.2f}, {np.max(p1):.2f}]")
            print(f"Normalized ranges - P2: [{np.min(p2):.2f}, {np.max(p2):.2f}]")
            
    if remove_non_event_matches:
        p1[np.abs(p1)<=0.08] = -100
        p2[np.abs(p2)<=0.08] = 100
        
    # Calculate initial distances
    distances = np.linalg.norm(p1 - p2, axis=-1)
    valid_distances = distances[np.isfinite(distances)]
    
    if len(valid_distances) == 0:
        raise ValueError("All distances are NaN/infinite after normalization")
    
    # Compute full distance matrix
    dist_matrix = cdist(p1, p2, metric='euclidean')
    overall_rr = np.mean(dist_matrix < radius)

    # Lag analysis
    max_lag = int(max_lag_seconds * sampling_rate)
    lags = np.arange(-max_lag, max_lag+1)
    rr_profile = []
    
    for lag in lags:
        diag = np.diag(dist_matrix, k=lag)
        
        if len(diag) == 0:
            rr_profile.append(0.0)
            continue
            
        rr = np.mean(diag < radius)
        rr_profile.append(rr)
    rr_profile = np.round(rr_profile, 4)
    lags = lags/sampling_rate
    overall_rr = np.round(overall_rr, 4)
    if debug:
        print(f"Final radius: {radius:.4f}")
        print(f"Max RR: {np.max(rr_profile):.4f}")
        print(f"Min RR: {np.min(rr_profile):.4f}")
    
        plt.figure(figsize=(10, 5))
        plt.plot(lags, rr_profile, 'b-o')
        plt.axvline(lags[np.argmax(np.array(rr_profile))], color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Lag (seconds)')
        plt.ylabel('Recurrence Rate')
        plt.title(f'Mimicry Lag Profile (Radius = {radius:.2f})')
        plt.grid(True)
        plt.show()
        
    return {
        'RR': overall_rr,
        'lags': lags,
        'rr_profile': rr_profile,
        'radius': radius
    }