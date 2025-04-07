# ------- This function is version 2, it uses the different attempts ------ 
import os
import cv2
import pandas as pd
import numpy as np

def process_participant_videos2(path_to_analysis_folder, path_to_avi_files, participant_id):
    # Finding the CSV file including the frames for the participant
    role = 'navigator' if int(participant_id) % 2 != 0 else 'pilot'  # navigator = odd, pilot = even
    
    csv_filename = f"pp{participant_id}_{role}_video_frames.csv"
    path_to_frame_csv_file = os.path.join(path_to_analysis_folder, csv_filename)
    #print(f"Reading CSV file: {path_to_frame_csv_file}")
    frame_data_csv = pd.read_csv(path_to_frame_csv_file)

    for index, row in frame_data_csv.iterrows():

        name = frame_data_csv['name'][index]
        # Filter AVI files for the current participant
        if 'video_cnt' in row:
            videocount = int(row['video_cnt'])
            if videocount < len(path_to_avi_files):
                # from avi_files only select role > then select videocount
                avi_files_all_role = [file for file in path_to_avi_files if f'{role}' in file]
                avi_files_with_rewrapped = ['_rewrapped' in avi_file for avi_file in avi_files_all_role]
                for i, avi_file in enumerate(avi_files_all_role):
                        if avi_files_with_rewrapped[i]:
                            start_str = avi_file.find('_rewrapped')
                            filename_wo_rewrapped = avi_file[0:start_str] + avi_file[start_str + len('_rewrapped'):]
                            avi_files_all_role = [avi_file for avi_file in avi_files_all_role if avi_file != filename_wo_rewrapped]
                avi_file = avi_files_all_role[videocount]
            else:
                return
        else:
            avi_file = next((file for file in path_to_avi_files if str(role) in file), None)
        # Open the video file
        cap = cv2.VideoCapture(avi_file)
        if not cap.isOpened():
            #print(f"Error: Could not open video file {avi_file}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(f"FPS: {fps}, Width: {width}, Height: {height}, Total Frames: {length}")

        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        output_filename = f'pp{participant_id}_{role}_{name}_reconstructed_video.avi'
        # Create the new folder path
        facial_expression_folder = os.path.join(path_to_analysis_folder, 'facial_expression')
        # Create the folder if it doesn't exist
        os.makedirs(facial_expression_folder, exist_ok=True)
        # Define the new output path
        path_output_filename = os.path.join(facial_expression_folder, output_filename)
        path_output_filename_no_black = path_output_filename.replace('.avi', '_no_black.avi')

        # Skip processing if output files already exist
        if os.path.exists(path_output_filename) and os.path.exists(path_output_filename_no_black):
            #print(f"Output video files already exist for segment {index}, skip this segment {index} of pp{participant_id}")
            continue
        #print(f"Output filename: {output_filename}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(path_output_filename), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path_output_filename, fourcc, fps, (width, height))
        out_no_black = cv2.VideoWriter(path_output_filename_no_black, fourcc, fps, (width, height))
        if pd.isna(row['start_frame']) or row['start_frame'] == '':
            #print(f"Skipping participant at index {index} due to empty start_frame")
            continue
        start_frame = int(row['start_frame'])
        finish_frame = int(row['finish_frame'])

        # Finding actual frames
        txt_filename = f"pp{participant_id}_{role}_{row['name']}_video_frames.txt"
        path_to_frame_txt_file = os.path.join(path_to_analysis_folder, txt_filename)
        if os.path.exists(path_to_frame_txt_file):
            with open(path_to_frame_txt_file, 'r') as file:
                actual_frames = [int(frame) + start_frame for frame in file.read().strip().split(',')]
        else:
            #print(f"Text file not found: {path_to_frame_txt_file}")
            continue

         # Writing actual frames (and adding black frames to missing frames)
        for frame_num in range(0, len(actual_frames)):
             if frame_num > 0:
                frame_diff = actual_frames[frame_num] - actual_frames[frame_num-1]
                if frame_diff > 1:
                    for j in range(0, frame_diff-1):
                        out.write(black_frame)
             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num+start_frame)
             ret, frame = cap.read()
             if ret:
                 out.write(frame)
                 out_no_black.write(frame)
                 #print(f"Writing frame {frame_num}")
            #  else:
            #      print(f"Error: Could not read frame {frame_num}")
            #      #print(f"Writing black frame at {frame_num}")

        out.release()
        out_no_black.release()
        print(f"Video reconstruction complete for segment {index} of pp{participant_id}.")

    cap.release()
