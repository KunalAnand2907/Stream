import cv2
import numpy as np
import time
import streamlit as st



def Imp():
    prevTime = 0
    fps = 0
    i = 0
    thres = 0.45  # Threshold to detect object
    nms_threshold = 0.08

    min_width = 80
    min_height = 80
    count_line_position = 550
    # Function to find center point of Bounding Box of a vehicle
    def center_handle(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    # List to append x,y,w,h points of detected vehicles
    detections = []
    offset = 6  # Allowable error between pixel
    counter = 0

    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    cap = cv2.VideoCapture(r'C:\Users\KUNAL\PycharmProjects\IndustryProject\Object_Tracker\Combined.mp4')  # 1 for multiple videos

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)  # 1100,720z

    result = cv2.VideoWriter(r'C:\Users\KUNAL\PycharmProjects\IndustryProject\OutputImp.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             20, size)

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:  # rt for read only
        classNames = f.read().rstrip('\n').split('\n')  # Split w.r.t new line
    print(classNames)

    # Step 2- loading weights and mobile net  and creating a Mobilenet _V3 model
    configPath = r'C:\Users\KUNAL\PycharmProjects\IndustryProject\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = r'C:\Users\KUNAL\PycharmProjects\IndustryProject\frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    # Basic Measurements
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    allowed_objects = ["car", "truck", "traffic lights"]

    while True:

        success, img = cap.read()
        i=i+1
        # Part 1
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)

        # Applying on each frame
        img_sub = object_detector.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        contours, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(img, (25, count_line_position), (1200, count_line_position), (255, 0, 0), 3)

        height, width, _ = img.shape
        # print(height,width)

        for (i, c) in enumerate(contours):
            # Calculate area and remove small elements
            # cv2.drawContours(roi1, [c], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w >= min_width) and (h >= min_height)
            if not validate_counter:
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Vehicle:" + str(counter), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)

            center = center_handle(x, y, w, h)
            # Append center points in detections
            detections.append(center)
            cv2.circle(img, center, 4, (0, 0, 255), -1)

            for (x, y) in detections:
                # If it crosses this line
                if y < (count_line_position + offset) and y > (count_line_position - offset):
                    counter += 1
                    cv2.line(img, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                    detections.remove((x, y))
                    print("Vehicle Counter:" + str(counter))

        cv2.rectangle(img, (0, 0), (400, 100), (0, 150, 0), -1)
        cv2.putText(img, "3.) ATCC", (20, 60), 0, 2, (255, 255, 255), 3)
        cv2.putText(img, "VEHICLE COUNTER:" + str(counter), (460, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.imshow("Output2", img)

        # Part 2 - Object Detection
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)  # conv from np array to list
        confs = list(np.array(confs).reshape(1, -1))[0]
        confs = list(map(float, confs))
        print(classIds)

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        for i, confidence in zip(indices, confs):
            i = i[0]
            box = bbox[i]
            if classNames[classIds[i][0] - 1] in allowed_objects:
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classIds[i][0] - 1], (box[0] - 3, box[1] - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255), 1)
                cv2.putText(img, str(round(confidence * 10, 2)), (box[0] + 35, box[1] - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        if save_img:
            # st.checkbox("Recording", value=True)
            out.write(frame)
            # Dashboard
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_column_width=True)

        result.write(img)
        cv2.imshow("Output1", img)

        if cv2.waitKey(50) == 13:
            break

    cv2.destroyAllWindows()
    cap.release()