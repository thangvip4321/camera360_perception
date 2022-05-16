#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
import logging
from os import wait
import os
import os
import signal
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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

class ROS_handler:
	def __init__(self, publishing_topic,camera):
		self.publishing_topic  = publishing_topic
		self.Publisher = rospy.Publisher(self.publishing_topic, OccupancyGrid, queue_size=1)
		self.camera = camera

	def publish(self, occupancy_grid):
		self.Publisher.publish(occupancy_grid)




gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_memory_growth(gpus[0], True)
		tf.config.experimental.set_virtual_device_configuration(gpus[0],
																[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])
	except RuntimeError as e:
		print(e)

def get_all_in_queue_block_first(q:Queue):
	arr = []
	obj = q.get()
	arr.append(obj)
	try:
		while True:
			obj =q.get_nowait()
			arr.append(obj)
	except Empty as e:
		return arr


params = rospy.get_param('camera360_insta')
sub_object       = {}
inference_queues = {}



#===========================================================================================================
# Create objects to subscribe into image topics




def read_and_process(orientation):
	# print("orientation 97",orientation)
	rospy.init_node('read_camera_{}'.format(orientation), anonymous=True,disable_signals=True)
	# print("ori in subproc read",orientation)
	while not rospy.is_shutdown():
		img =sub_object[orientation].camera.get_bgr_frame()
		t0 = time()
		# cv2.imshow("read img {}".format(orientation),img)
		# cv2.waitKey(1)
		location = (orientation,t0)
		# #=======================================================================================================
		# # Pre-process before feeding into deep learning model
		front_pp =  model.preprocess(img)
		# front_pp = img
		#======================================================================================================
		try:
			predict_queue.put_nowait((front_pp,location))
		except Full:
			predict_queue.get()
			predict_queue.put((front_pp,location))
		# print("end loop read",orientation)


def after_segmentation_work(orientation):
	rospy.init_node('publish_camera_{}'.format(orientation), anonymous=True,disable_signals=True)
	# print("96",orientation)
	bev_file = params[orientation]['bev_calib_file']
	frame_id = params[orientation]['base_frame']
	pose = params[orientation]['pose_to_base_frame']
	occ_grid_width,occ_grid_height=params[orientation]['occ_grid_size']
	cell_size_in_m = 0.1
	bev_tool =bev_transform_tools.fromJSON(bev_file)
	q = inference_queues[orientation]
	while not rospy.is_shutdown():
		segmap,location = q.get()
		segmap = contour_noise_removal(segmap)
		# print("FPS: ",orientation,1/(time()-location[1]))
		t = rospy.Time.now()
		resized_img = cv2.resize(segmap,(bev_tool.input_height,bev_tool.input_width))
		occ_grid_front = convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(resized_img,occ_grid_width,occ_grid_height,cell_size_in_m),\
													cell_size_in_m,occ_grid_width,occ_grid_height,t,frame_id,pose)
		sub_object[orientation].publish(occ_grid_front)


def interrupt_handler(a,b):
	print("ok")
	# rospy.signal_shutdown("haha")
	for job in all_processes:
		job.kill()
if __name__ == "__main__":
	model_path = rospy.get_param('model_path')
	model = ENET(model_path)
	camera_file = rospy.get_param("camera_intrinsic_file_path")
	for orientations in params:
		pub_topic_name = params[orientations]["pub_topics"]
		lens =params[orientations]["sub_topics"]
		stitch_file = params[orientations]['stitch_calib_file']
		num_topics = len(lens)
		assert (len(lens)==2) == (stitch_file !=""), "either 2 topic and 1 stitch file or 1 topic and 0 stitch file"
		if(len(lens)==2):
			with open(stitch_file,"r") as f:
				obj = yaml.safe_load(f)
				homography_matrix = np.reshape(np.array(obj["homography_matrix"]),(3,3))
				crop_x = obj["crop_area_x"]
				crop_y = obj["crop_area_y"]
				stitcher = Stitcher(homography_matrix,(*crop_x,*crop_y))
				sub_object.update({orientations: ROS_handler(pub_topic_name,ROSStitchCamera(lens[0],lens[1],camera_file,stitcher))})
		else:
			sub_object.update({orientations: ROS_handler(pub_topic_name,ROSCamera(lens[0],camera_file))})
	all_processes = []
	# print(params)
	predict_queue = Queue(10)
	for orientation in params:
		inference_queues[orientation] = Queue(10)
		read_and_preprocess_subproc = Process(target=read_and_process,args=[orientation])
		bev_and_pub_subproc =Process(target=after_segmentation_work,args=[orientation])
		all_processes.append(read_and_preprocess_subproc)
		all_processes.append(bev_and_pub_subproc)
	for i in all_processes:
		i.start()
	rospy.init_node('clgt', anonymous=True,disable_signals=True)
	r = rospy.Rate(30)
	while not rospy.is_shutdown():
		t1 = time()
		arr_img_and_loc = get_all_in_queue_block_first(predict_queue)
		arr_img = []
		for img,loc in arr_img_and_loc:
			arr_img.append(img)
		batch = np.concatenate(arr_img)
		# print("batch shape",batch.shape)
		t2 = time()
		# print("time_take_pictures ",1/(t2-t1))
		segmaps = model.predict(batch)
		t3 = time()
		# print("inference time ",1/(t3-t2))
		# print("shape 160",segmaps.shape)
		for i,img in enumerate(segmaps):
			# print(i,arr_img_and_loc[i])
			location = arr_img_and_loc[i][1]
			orientation = location[0]
			try:
				inference_queues[orientation].put_nowait((img,location))
			except (Full):
				inference_queues[orientation].get()
				inference_queues[orientation].put((img,location))
		r.sleep()
	signal.signal(signal.SIGINT,interrupt_handler)

	for job in all_processes:
		job.kill()
