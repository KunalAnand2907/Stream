import cv2
import numpy as np


# Step 1 - Reading, Showing & taking All the Coco names in a list
# img = cv2.imread(r"C:\Users\KUNAL\PycharmProjects\IndustryProject\lena.png")

# Step 2 - Defining The Threshold and Nms Threshold
thres = 0.5 # Threshold to detect object
nms_threshold = 0.08
path = r'Car.png'
img = cv2.imread(path)

# Step 4 - Creating The Empty List and Adding All the Class Names to it
classNames = []
classFile = r'coco.names'
with open(classFile, 'rt') as f:  # rt for read only
    classNames = f.read().rstrip('\n').split('\n')  # Split w.r.t new line
print(classNames)

# Step 5- loading weights and mobile net Model  and creating a Mobilenet _V3 model
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

# Step 6 - Creating a Detection Model by taking Weights and Mobilenet Module
net = cv2.dnn_DetectionModel(weightsPath, configPath)
# Basic Measurements For Detection Model
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=thres)  # less threshold value to uniquely identify the class
bbox = list(bbox)  # conv from np array to list
confs = list(np.array(confs).reshape(1, -1))[0]
confs = list(map(float, confs))
# print(type(confs[0]))
# print(confs)
# print(classIds)

# Step 8 - Non Maximum Suppresion- Removing the Duplicate BBox value which has least Min Threshold Value so as to improve accuracy
indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
# print(indices)
# print(type(indices[0][0]))
# Step 9 - Creating a Rectangle & Adding Text To the input Frame
for i, confidence in zip(indices, confs):
    i = i[0]
    box = bbox[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
    cv2.putText(img, classNames[classIds[i][0] - 1], (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Conf:" + str(round(confidence * 100, 2)), (box[0] + 290, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
print(classNames[classIds[i][0] - 1], "Conf:" + str(round(confidence * 100, 2)),
      "Count:" + str(np.count_nonzero(classIds[0])))

cv2.imshow("Detected image",img)

