import cv2
import numpy as np

## PARAMETERS
face_detection_model = "../S3B-dnn_face_detection/face_detection_yunet_2022mar.onnx" # download from https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
score_threshold = 0.9 # Filtering out faces of score < score_threshold (used to eliminate unlikely faces)
nms_threshold = 0.3 # Suppress bounding boxes of iou >= nms_threshold (used to eliminate same bboxes)
top_k = 5000 # Keep top_k bounding boxes before NMS.

def visualize(input, faces, thickness=2):
    if faces is None:
        print("No face found")
        return
    for face in faces:
        coords = face[:-1].astype(np.int32) # necessary to convert coordinates to integers before plotting

        # draw rectangles of face face
        cv2.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)

        # draw points of facial features
        cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness) # right eye
        cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness) # left eye
        cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness) # nose tip
        cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness) # right corner of mouth
        cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness) # left corner of mouth


cap = cv2.VideoCapture(0)

## do initial capture to obtain image height and width to set model
if not cap.isOpened():
   raise RuntimeError("Camera could not be opened.")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Frame was not properly read")

imgWidth = int(frame.shape[1])
imgHeight = int(frame.shape[0])
detector = cv2.FaceDetectorYN.create(face_detection_model, "", (imgWidth, imgHeight), score_threshold, nms_threshold, top_k)

## webcam code to continuously grab frame   
while True:
    ret, frame = cap.read() # Capture frame-by-frame

    if not ret: 
        raise RuntimeError("Frame was not properly read")

    faces = detector.detect(frame)[1]
    visualize(frame, faces)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'): # exits if Q key pressed
        break

cap.release()
cv2.destroyAllWindows()

