import numpy as np
import os
import cv2 

## PARAMETERS
image = "../img/object_detection/person1.jpg"
face_detection_model = "../S3B-dnn_face_detection/face_detection_yunet_2022mar.onnx" # download from https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
score_threshold = 0.9 # Filtering out faces of score < score_threshold (used to eliminate unlikely faces)
nms_threshold = 0.3 # Suppress bounding boxes of iou >= nms_threshold (used to eliminate same bboxes)
top_k = 5000 # Keep top_k bounding boxes before NMS.

eye_cascade_name = os.path.join(cv2.data.haarcascades,'haarcascade_eye_tree_eyeglasses.xml')
eye_cascade = cv2.CascadeClassifier()
eye_cascade.load(cv2.samples.findFile(eye_cascade_name))

def visualize(input, faces, thickness=2):
    if faces is None:
        print("No face found")
        return
    for face in faces:
        coords = face[:-1].astype(np.int32) # necessary to convert coordinates to integers before plotting

        # draw rectangles of face face
        x = coords[0]
        y = coords[1]
        w = coords[2]
        h = coords[3]
        cv2.rectangle(input, (x,y), (x+w, y+h), (0, 255, 0), thickness)

        faceROI = input[y:y+h, x:x+w] # limits eye detection to only within where faces were detected
        cv2.imshow("image", faceROI)
        cv2.waitKey(0)
        faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
        faceROI = cv2.equalizeHist(faceROI)


        eyes = eye_cascade.detectMultiScale(faceROI)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(input, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), thickness)
        

img = cv2.imread(cv2.samples.findFile(image))
imgWidth = int(img.shape[1])
imgHeight = int(img.shape[0])
detector = cv2.FaceDetectorYN.create(face_detection_model, "", (imgWidth, imgHeight), score_threshold, nms_threshold, top_k)

faces = detector.detect(img)[1]
visualize(img, faces)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
