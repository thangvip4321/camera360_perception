#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
import logging
from os import wait
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from threading import Thread
from time import time,sleep
import numpy as np
import cv2
import yaml
from bugcar_image_segmentation.bev import bev_transform_tools
from bugcar_image_segmentation.models import ENET
from bugcar_image_segmentation.occgrid_to_ros import convert_to_occupancy_grid_msg
from cameraType.baseCamera import BaseCamera
from cameraType.ros_camera import ROSCamera, ROSStitchCamera
import tensorflow as tf
import rospy
from bugcar_image_segmentation.image_processing_utils import contour_noise_removal
from stitcher import Stitcher
from nav_msgs.msg import OccupancyGrid
from queue import Empty, Full 
from multiprocessing import Pipe, Process,Queue

logger = logging.Logger('bff')
logger.setLevel(logging.DEBUG)
params = rospy.get_param('camera360_insta')
model_path = rospy.get_param('model_path')
camera_file = rospy.get_param('camera_intrinsic_file_path')
# model = ENET(model_path)
sub_object       = {}
inference_queues = {}





def read_and_process(orientation):
	rospy.init_node('read_camera_{}'.format(orientation), anonymous=True,disable_signals=True)
	lens =params[orientation]["sub_topics"]
	print("ori in subproc read",orientation,lens)
	camera = ROSCamera(lens[0],camera_file)
	while not rospy.is_shutdown():
		img =camera.get_bgr_frame()
		t0 = time()
		print("hello",orientation)
		if(img is not None):
			cv2.imshow("read img {}".format(orientation),img)
			cv2.waitKey(1)
		# #=======================================================================================================



all_processes = []
for orientation in params:
	print("ori in main 140",orientation)
	inference_queues[orientation] = Queue(10)
	read_and_preprocess_subproc = Process(target=read_and_process,args=[orientation])
	all_processes.append(read_and_preprocess_subproc)
for i in all_processes:
	i.start()

rospy.init_node('camera360', anonymous=True,disable_signals=True)
r = rospy.Rate(30)
while not rospy.is_shutdown():
	r.sleep()

for job in all_processes:
	job.join()
