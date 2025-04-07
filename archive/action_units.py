### all in one doc:
import util.feature_extraction as fe
import os

openface_path = r"C:\Users\Erik\Documents\facial-expression-synchrony\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
  
data_in = r"D:\data"
data_out = r"data-out"
folder = "au"
phases = [("instructional_video", 1), ("discussion_phase", 2), ("reschu_run", 8)]
for i in range(5,100)[::2]:
    j = i + 1
    if i < 10: i = "0" + str(i)
    if j < 10: j = "0" + str(j)
    pair = f"{i}_{j}"
    print(pair)
    for phase, count in phases:
        for c in range(count):
            nav = f"pp{i}_navigator_{phase}_{c}"
            pil = f"pp{j}_pilot_{phase}_{c}"
            for participant in [nav, pil]:
                input = os.path.join(data_in, participant + "_reconstructed_video.avi")
                output = os.path.join(data_out, pair, folder, participant + ".csv")
                if os.path.exists(input):
                    if os.path.exists(output):
                        print(f"{participant} already processed, skipping")
                        continue
                    else:
                        print(f"{participant} extraction starting \n\n\n\n\n")
                        os.system(f"{openface_path} -f \"{input}\" -aus -of \"{output}\"")
                else:
                    print(f"{participant} avi file does not exist in the input folder")
