import cv2
import torch
import time
import argparse
import utils
from PIL import Image
from facenet_pytorch import MTCNN

# computation device
device = torch.device('cuda')
mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# set the save path
save_path = f'data/webcam.mp4'
# define codec and create VideoWriter object
out = cv2.VideoWriter(save_path,
cv2.VideoWriter_fourcc(*'mp4v'), 30,
(frame_width, frame_height))
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame).convert('RGB')
        # get the start time
        start_time = time.time()
        # the detection module returns the bounding box coordinates and confidence …
        # … by default, to get the facial landmarks, we have to provide …
        # … `landmarks=True`
        bounding_boxes, conf, landmarks = mtcnn.detect(pil_image, landmarks=True)
        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        wait_time = max(1, int(fps/4))
        # color conversion for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # draw the bounding boxes around the faces
        print(bounding_boxes)
        if bounding_boxes is None:
            print('NO RESULTS')
        frame = utils.draw_bbox(bounding_boxes, frame)
        # plot the facial landmarks
        frame = utils.plot_landmarks(landmarks, frame)
        cv2.imshow('Face detection frame', frame)
        out.write(frame)
        # press `q` to exit
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f'Average FPS: {avg_fps:.3f}')