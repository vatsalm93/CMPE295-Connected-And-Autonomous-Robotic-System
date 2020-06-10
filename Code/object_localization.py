import jetson.inference
import jetson.utils
import numpy as np
import cv2
import pyzed.sl as sl
import math
import time
import serial
import re
from multiprocessing import Process
from threading import Thread, Event
from time import sleep
from PIL import Image

event = Event()
zed = 0
x_dwm = 0
y_dwm = 0
z_dwm = 0

# Get the zed camera object
zed = sl.Camera()

# Get the UART Instance and set the baudrate
serr=serial.Serial(
      port = '/dev/ttyACM0',
      baudrate = 115200,
      bytesize = serial.EIGHTBITS,
      parity = serial.PARITY_NONE,
      stopbits = serial.STOPBITS_ONE,
)
sleep(0.5)
serr.write(b'\x0d'+b'\x0d'+b"lep\n")
print("\n\nSignal sent to DWM\n\n")
sleep(0.1)

# Implementing threading for getting the data from the UART continuously into global variables
def modify_variable(ser,zedd):
  global x_dwm
  global y_dwm
  global z_dwm
  while True:
    out = ser.readline()
    data = re.split('[,\r\n]', out.decode('utf-8'))
    if event.is_set():
      break
    try:
      x_dwm = round(float(data[1]) - 0.3, 2)
      y_dwm = round(float(data[2]), 2)
      z_dwm = round(float(data[3]),2)
    except ValueError:
      continue
  print('\nThread for getting data from UART stopped\n')

# Create the thread, assign a function to it and start it
t = Thread(target=modify_variable, args=(serr,zed))
t.start()

# Set configuration parameters
init_parameters = sl.InitParameters()
init_parameters.camera_resolution = sl.RESOLUTION.HD720
init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA
init_parameters.coordinate_units = sl.UNIT.METER
init_parameters.depth_minimum_distance = 0.3
init_parameters.camera_fps = 30

# Open the camera
err = zed.open(init_parameters)
if err != sl.ERROR_CODE.SUCCESS:
  exit(-1)

#Create a zed image object
zed_image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()

runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD

# Load the model into a variable
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.4)

#open the display to view the video feed
display = jetson.utils.glDisplay()

# Get the initial frame and store the width and height of the images
if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
  zed.retrieve_image(zed_image,sl.VIEW.LEFT)
  width = zed_image.get_width()
  height = zed_image.get_height()

while display.IsOpen():

  # A new image is available if grab() returns SUCCESS
  if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # Retrieve left image
    zed.retrieve_image(zed_image,sl.VIEW.LEFT)

    # Convert from zed image to numpy format
    numpy_img = zed_image.get_data()
    b,g,r,a = cv2.split(numpy_img)
    # Change image to RGBA color format
    numpy_img_1 = cv2.merge((r,g,b,a))
    # Convert to numpy array to cuda image capsule
    cuda_img = jetson.utils.cudaFromNumpy(numpy_img_1)

    # Retrieve depth map. Depth is aligned on the left image
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    # Retrieve colored point cloud. Point cloud is aligned on the left image.
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    # Run the model to detect the objects
    detections = net.Detect(cuda_img, width, height)

    # If objects are detected label the image with bounding boxes and distance information
    if(len(detections)):
      for detection in detections:

        (x_c, y_c) = detection.Center
        x_w = round(detection.Width)
        x_h = round(detection.Height)
        try:

        # Retrive the point cloud i.e. get the depth information for every pixel
          err, point_cloud_value = point_cloud.get_value(x_c, y_c)
          depth_sq = point_cloud_value[2] * point_cloud_value[2]

        # Calculate the distance using the Euclidean distance formula
          distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] + point_cloud_value[1] * point_cloud_value[1] + depth_sq)

        # Label the distance value for each object detected in the image
          font = jetson.utils.cudaFont(size=jetson.utils.adaptFontSize(x_w))

          # Overlay text with information such object coordinates (X, Y, Z) for each object w.r.t the UWB and 
          # the distance from camera to the objects in meters
          font.OverlayText(cuda_img, width, height, "Obj Coordinates:", round(x_c), round(y_c), font.White, font.Gray40)
          font.OverlayText(cuda_img, width, height, "{},{},{}m".format(round(x_dwm - point_cloud_value[2],2), round(y_dwm + point_cloud_value[0],2), round(z_dwm - point_cloud_value[1],2)), round(x_c), round(y_c)+35, font.White, font.Gray40)
          font.OverlayText(cuda_img, width, height, "dist = {}m".format(round(distance,3)), round(x_c), round(y_c)+70, font.White, font.Gray40)
          
          jetson.utils.cudaDeviceSynchronize()
        except ValueError:
          continue;

	  #Overlay text with Drone coordinates (X, Y, Z) w.r.t the UWB in meters
      font = jetson.utils.cudaFont(size=jetson.utils.adaptFontSize(width))
      font.OverlayText(cuda_img, width, height, "Drone position: X={}m, Y={}m, Z={}m".format(x_dwm, y_dwm, z_dwm), 10, 10, font.White, font.Gray40)

    jetson.utils.cudaDeviceSynchronize()

    # Open a window for displaying the images 
    display.RenderOnce(cuda_img, width, height)
    display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

zed.close()