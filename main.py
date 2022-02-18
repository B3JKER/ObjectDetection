import cv2  # OpenCV-python import
from os import listdir  # File import library
from os.path import isfile, join  # File import library


def detection(imgtype, img, thresh, nms_thresh):
    classIds, confs, bbox = net.detect(img, confThreshold=thresh)  # Given the input frame, create input blob,
    bbox = list(bbox)  # conversion array to list                  # run net and return result detections.
    if len(classIds) != 0:  # if something on image was classified
        confs = confs.tolist()  # conversion to list
        confs = list(map(float, confs))  # conversion to list of floats
    indices = cv2.dnn.NMSBoxes(bbox, confs, thresh, nms_thresh)  # Performs non maximum suppression given boxes
    for i in indices:  # and corresponding scores.
        box = bbox[i]
        confidence = confs[i]
        x, y, w, h = box[0], box[1], box[2], box[3],
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)  # Draw box
        cv2.putText(img, classNames[classIds[i] - 1].upper(), (box[0] + 10, box[1] + 30),  # Draw name
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 60),  # Draw confidence
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Output', img)   # Show image with detected objects
    cv2.waitKey(imgtype)


thresh = 0.55  # Threshold to detect object
nms_thresh = 0.3  # NMS Threshold, lower better

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Video width
cap.set(4, 1024)  # Video height

classNames = []
classFile = 'coco.names'  # file with all classes names
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  # list of objects classes

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Text file contains network configuration.
weightsPath = 'frozen_inference_graph.pb'  # Binary file contains trained weights.

net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Creates net
net.setInputSize(320, 320)  # net settings
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

choice = input('If you want to detect objects on image, write "i", if you want to use camera, then write "c"\n')
# Checking all images in "Resources" and use function
# "detection" on them
if choice == 'i':
    onlyfiles = [f for f in listdir('Resources') if isfile(join('Resources', f))]
    print("All images:")
    print(onlyfiles)
    for imageName in onlyfiles:
        print("Current image: " + imageName)
        img = cv2.imread('Resources/' + imageName)
        detection(0, img, thresh, nms_thresh)
# Object detection for camera
elif choice == 'c':
    while True:
        success, img = cap.read()
        detection(1, img, thresh, nms_thresh)
else:
    print("You passed wrong input")
