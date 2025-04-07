import cv2
import os

def rewrap_video(file, path):
    print(file)

    cap = cv2.VideoCapture(file)

    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0 :
        print('Rewrapping video')

        # Add '_restored' to the filename
        folder = os.path.basename(os.path.dirname(file))
        filename = os.path.basename(file)
        name, ext = os.path.splitext(filename)
        restored_file = f"{path}\\{folder}\\{name}_rewrapped{ext}"

        # Check if rewrapped video already exists
        if not os.path.exists(restored_file):

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(restored_file), fourcc, int(cap.get(cv2.CAP_PROP_FPS)),(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            while cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
                out.release()
        else:
            print('Video was already rewrapped')

    else:
        print('Video already has metadata')
        cap.release()
    

    cv2.destroyAllWindows()
    print('\n')

