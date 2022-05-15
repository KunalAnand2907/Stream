###############################################  Vehicle Detection And Count  ###############################################################
import cv2
import numpy as np
import os


# Step 1 - Reading, Showing & taking All the Coco names in a list
#img = cv2.imread(r"C:\Users\KUNAL\PycharmProjects\IndustryProject\lena.png")
thres = 0.45 # Threshold to detect object
nms_threshold=0.08

cap = cv2.VideoCapture(r'two_wheeler2.mp4') # 1 for multiple videos

cap.set(3,1280) # width3.mp4
cap.set(4,720)  # height
cap.set(10,150) # Brightness

classNames=[]
classFile= 'coco.names'
with open(classFile,'rt') as f: # rt for read only
    classNames = f.read().rstrip('\n').split('\n') # Split w.r.t new line
print(classNames)

# Step 2- loading weights and mobile net  and creating a Mobilenet _V3 model
configPath = r'C:\Users\KUNAL\PycharmProjects\IndustryProject\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'C:\Users\KUNAL\PycharmProjects\IndustryProject\frozen_inference_graph.pb'

net= cv2.dnn_DetectionModel(weightsPath,configPath)
# Basic Measurements
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:

      success,img = cap.read()
      classIds,confs,bbox = net.detect(img,confThreshold=thres)
      bbox=list(bbox) # conv from np array to list
      confs =list(np.array(confs).reshape(1,-1))[0]
      confs= list(map(float,confs))
      # print(type(confs[0]))
      # print(confs)
      print(classIds)

      indices =cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
      #print(indices)
      #print(type(indices[0][0]))

      for i,confidence in zip(indices,confs):
          i = i[0]
          box = bbox[i]
          x, y, w, h = box[0], box[1], box[2], box[3]
          cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
          cv2.putText(img, classNames[classIds[i][0] - 1], (box[0] -3, box[1]-5),
                      cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255), 1)
          cv2.putText(img,str(round(confidence * 10, 2)), (box[0] +35, box[1] - 5),
                      cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)


      print(classNames[classIds[i][0] - 1], "Conf:" + str(round(confidence * 100, 2)), "Count:" + str(np.count_nonzero(classIds[0])))




# zip for 3 diff list and flatten it
# Create a rectangle wrt bbox, names, conf
      """if len(classIds) != 0:
               for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                         cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                         cv2.putText(img, "Conf:"+str(round(confidence * 100, 2)), (box[0] + 250, box[1] + 30),
                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)"""

      cv2.imshow('Output',img)
      cv2.waitKey(5)




   


