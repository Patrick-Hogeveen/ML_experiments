import cv2

# draw the bounding boxes for face detection
def draw_bbox(bounding_boxes, image):
    if bounding_boxes is not None:
        for i in range(len(bounding_boxes)):
            x1, y1, x2, y2 = bounding_boxes[i]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 0, 255), 2)
    
    return image

# plot the facial landmarks
def plot_landmarks(landmarks, image):
    if landmarks is not None:
        for i in range(len(landmarks)):
            for p in range(landmarks[i].shape[0]):
                cv2.circle(image, 
                        (int(landmarks[i][p, 0]), int(landmarks[i][p, 1])),
                        2, (0, 0, 255), -1, cv2.LINE_AA)
    return image