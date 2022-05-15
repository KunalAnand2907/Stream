from ObjectCount_1 import *
import streamlit as st
import cv2
import pickle
import tempfile
import numpy as np

def main():
  st.set_page_config(layout="wide",page_title="Traffic Management & Forecasting Solutions",
                       page_icon=":vertical_traffic_light:")
  thres = 0.45  # Threshold to detect object
  nms_threshold = 0.01
  classFile = 'coco.names'
  with open(classFile, 'rt') as f:  # rt for read only
        classNames = f.read().rstrip('\n').split('\n')  # Split w.r.t new line

  st.title("""Vehicle Counter & Object Detection Dashboard :blue_car:  :red_car:""")
  st.subheader('Streamlit App by [Traffic Mgmt. & Forecasting Solutions](https://www.linkedin.com/in/kunal-anand-b36730169/)')

  st.markdown('''It is a high–speed A.I. Proctored traffic data collection that **Detects the type of vehicle**, **count's the no. of vehicles**
  ,**Classifies Vehicles- either it is 2- Wheeler (Scooty, Bikes) or 4- Wheelers(Car, Truck, Bus)**,**Colour** and **Keep track of vehicles via Object Tracking** by giving them unique id & also mentions if they are Found or lost w.r.t position in the video. **Parking Mgmt. System** notifies the user when parking space is vacant 
  via message with a surrounding green box and also detects vehicles which are already parked with a red box.
  
  If you are interested in how this app was developed check out my [Medium article](https://medium.com/@kunalanand2907/fifa-19-a-data-visualization-project-with-tableau-6ca13b0518ab)''')


  st.sidebar.title("Settings")
  st.markdown(
      """<style>
         [data-test-id="stSidebar"][aria-expanded="true"]>div:first-child{width: 400px;}
         [data-testid="stSidebar"][aria-expanded="false"]> div:first-child{width:400px; margin-left: -400px}
         </style>
         """,
      unsafe_allow_html = True,
  )
  def_values = {'conf': 0.45, 'nms': 0.01}
  keys = ['conf', 'nms']

  st.sidebar.markdown('---')
  confidence_1 = st.sidebar.slider('Model Threshold', min_value=0.0, max_value=1.0, value=0.45)
  confidence_1 = thres
  st.sidebar.markdown('---')
  confidence_2 = nms_threshold
  confidence_2 = st.sidebar.slider('Overlapping Threshold', min_value=0.0, max_value=1.0, value=0.01)
  st.sidebar.markdown('---')

  # Checkboxes
  save_img= st.sidebar.checkbox('Save Video')
  if save_img:
      st.checkbox("Recording", value=True)
  enable_GPU = st.sidebar.checkbox('Enable Gpu')
  custom_classes =st.sidebar.checkbox('Use Custom Classes')
  assigned_class_id=[]

  # custom classes
  if custom_classes:
      assigned_class = st.sidebar.multiselect('Select The Custom Classes:',list(classNames),default='car')
      a1 = st.sidebar.multiselect('Select The Custom Classes:', list(classNames), default='truck/Suv')
      for each,each1 in zip(assigned_class,a1):
          assigned_class_id.append(classNames.index(each))
          assigned_class_id.append(classNames.index(each1))

  # Uploading the input video

  video_file_buffer = st.sidebar.file_uploader("Upload a video",type=["mp4","mov","avi","m4v","asf"])
  DEMO_VIDEO =r"C:\Users\KUNAL\PycharmProjects\IndustryProject\Object_Tracker\Combined.mp4"
  tfflie = tempfile.NamedTemporaryFile(suffix='.mp4',delete= False)

  # We get Output video Here
  if not video_file_buffer:
      img = cv2.VideoCapture(DEMO_VIDEO)
      tfflie.name = DEMO_VIDEO
      dem_vid =open(tfflie.name,'rb')
      demo_bytes =dem_vid.read()

      st.sidebar.text('Input Video')
      st.sidebar.video(demo_bytes)
  else:
      tfflie.write(video_file_buffer.read())
      dem_vid = open(tfflie.name, 'rb+')
      demo_bytes = dem_vid.read()

  # width = int(img.get(cv2.CAP_PROP_FRAME_WIDTH))
  # height = int(img.get(cv2.CAP_PROP_FRAME_HEIGHT))



  print(tfflie.name)
  st.sidebar.text('Old Video')
  st.sidebar.video(demo_bytes)


  stframe = st.empty()
  st.sidebar.markdown("---")

  kpi1,kpi2,kpi3 = st.columns(3)

  with kpi1:
      st.subheader("**Frame Rate**")
      kpi1_text = st.markdown("0")

  with kpi2:
      st.subheader("**Tracked Objects**")
      kpi2_text = st.markdown("0")

  with kpi3:
      st.subheader("**Vehicle Type**")
      kpi3_text = st.markdown("0")
  st.markdown("<hr/>", unsafe_allow_html=True)




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

  cap = cv2.VideoCapture(
      r'C:\Users\KUNAL\PycharmProjects\IndustryProject\Combined.mp4')  # 1 for multiple videos

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

  allowed_objects = ["car", "truck","motorbike","Traffic Sign","person"]

  while True:
      timer = cv2.getTickCount()
      success, img = cap.read()
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
              a=classNames[classIds[i][0] - 1]
              cv2.putText(img, classNames[classIds[i][0] - 1], (box[0] - 3, box[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 255), 2)
              cv2.putText(img, str(round(confidence * 10, 2)), (box[0] + 39, box[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

      fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)  # Get frames per second


      if save_img:
          #st.checkbox("Recording", value=True)
          result.write(img)
          # Dashboard
      kpi1_text.write(f"<h2 style='color: green;'>{'{:.2f}'.format(int(fps))}</h1>", unsafe_allow_html=True)
      kpi2_text.write(f"<h2 style='text-align: center;color: green;'>{counter}</h1>", unsafe_allow_html=True)
      if classNames[classIds[i][0] - 1]=='Traffic Sign':
          kpi3_text.write(f"<h2 style='color: green;'>{'{}'.format(classNames[classIds[i][0] - 1])}</h1>", unsafe_allow_html=True)
      else:
          kpi3_text.write(f"<h2 style='color: green;'>{'{}--4 Wheeler'.format(classNames[classIds[i][0] - 1])}</h1>",
                          unsafe_allow_html=True)


      #img = cv2.resize(img, (600,400))
      stframe.image(img, channels='BGR', use_column_width=True)

      cv2.imshow("Output1", img)

      if cv2.waitKey(1) & 0xFF == ord('s'):
          break

  st.text('Video Processed')

  #output_video = open('OutputImp.avi', 'rb+')


  cv2.destroyAllWindows()
  cap.release()

main()
