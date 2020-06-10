#!/usr/bin/python

import jetson.inference
import jetson.utils
import numpy as np
import cv2
from PIL import Image
import time
import pyzed.sl as sl

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

image_a = sl.Mat()
image_b = sl.Mat()
runtime_parameters = sl.RuntimeParameters()
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

#Raspberry PI camera
#camera = jetson.utils.gstCamera(2560, 720, "/dev/video0")

#Create an instance of the ZED Camera
zed = sl.Camera()

display = jetson.utils.glDisplay()

for i in range(1,10):
# A new image is available if grab() returns SUCCESS

    #Get a new image from the ZED camera
	if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
		zed.retrieve_image(image_a,sl.VIEW.LEFT)
		#zed.retrieve_image(image_b,sl.VIEW.RIGHT)



#while display.IsOpen():
#img, width, height = camera.CaptureRGBA()
#image = Image.open('test2.jpg',)

    # Get height and width of the image
	width = image_a.get_width()
	height = image_a.get_height()

    #Get the image data
	#img1 = np.asarray(image)
	img1 = image_a.get_data()
	#img2 = image_b.get_data()
	#dst = Image.new('RGBA', (1280 + 1280, min(720, 720)))

    # add the image to GPU memory using CUDA
	img = jetson.utils.cudaFromNumpy(img1)

    #Peform inferences

	detections = net.Detect(img, width, height)

    #Print the results
	print(detections)
	if(len(detections) != 0):
		print(detections[0].Center)
	display.RenderOnce(img, width, height)
    
    #Display the image in real-time
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

zed.close()