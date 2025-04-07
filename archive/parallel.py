import os
from multiprocessing import Pool, cpu_count
import subprocess

# Config (adjust based on your system)
OPENFACE_PATH =  r"C:\Users\Erik\Documents\facial-expression-synchrony\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
DATA_IN = r"D:\data"
DATA_OUT = r"data-out"
FOLDER = "au"
MAX_WORKERS = max(1, cpu_count() - 1)  # Leave 1 core free for system
PHASES = [("instructional_video", 1), ("discussion_phase", 2), ("reschu_run", 8)]

def process_video(input_path, output_path):
    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"SKIPPING: Output exists - {output_path}")
        return True
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input missing - {input_path}")
        return False
    
    # Create output directory ONLY when needed (atomic operation)
    output_dir = os.path.dirname(output_path)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create {output_dir} - {e}")
        return False
    
    # Run OpenFace
    print(f"PROCESSING: {input_path}")
    cmd = [
        OPENFACE_PATH,
        "-f", input_path,
        "-aus",
        "-of", output_path,
        "-no2Dfp", "-no3Dfp", "-noMparams", "-noPose", "-noGaze"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {input_path}\nError: {e.stderr.decode().strip()}")
        return False

def prepare_tasks():
    tasks = []
    for i in range(5, 100, 2):
        j = i + 1
        i_str = f"0{i}" if i < 10 else str(i)
        j_str = f"0{j}" if j < 10 else str(j)
        
        for phase, count in PHASES:
            for c in range(count):
                nav = f"pp{i_str}_navigator_{phase}_{c}"
                pil = f"pp{j_str}_pilot_{phase}_{c}"
                
                for participant in [nav, pil]:
                    input_file = os.path.join(DATA_IN, f"{participant}_reconstructed_video.avi")
                    output_file = os.path.join(DATA_OUT, f"{i_str}_{j_str}", FOLDER, f"{participant}.csv")
                    tasks.append((input_file, output_file))
    return tasks

def main():
    tasks = prepare_tasks()
    print(f"Total videos to process: {len(tasks)}")
    
    with Pool(MAX_WORKERS) as pool:
        results = pool.starmap(process_video, tasks)
    
    success_rate = sum(results) / len(tasks) * 100
    print(f"Finished! Success rate: {success_rate:.2f}%")

if __name__ == '__main__':
    main()