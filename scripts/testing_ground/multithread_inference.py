#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
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
from stitcher import Stitcher
from nav_msgs.msg import OccupancyGrid
from queue import Full 
from multiprocessing import Pipe, Process,Queue
import asyncio


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_memory_growth(gpus[0], True)
		tf.config.experimental.set_virtual_device_configuration(gpus[0],
																[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=800)])
	except RuntimeError as e:
		print(e)
class ROS_handler:
	def __init__(self, publishing_topic,camera):
		self.publishing_topic  = publishing_topic
		self.Publisher = rospy.Publisher(self.publishing_topic, OccupancyGrid, queue_size=1)
		self.camera = camera

	def publish(self, occupancy_grid):
		self.Publisher.publish(occupancy_grid)

params = rospy.get_param('camera360_insta')
# orientations = rospy.get_param('~orientation')
model_path = rospy.get_param('model_path')
camera_file = rospy.get_param('camera_intrinsic_file_path')
model = ENET(model_path)
sub_object            = {}
bev_file = params["front"]['bev_calib_file']
frame_id = params["front"]['base_frame']
pose = params["front"]['pose_to_base_frame']
# receive_image_queue = Queue(10)
preprocess_queue = Queue(10)
a,b = Pipe(False)
predict_queue = Queue(10)
bev_queue = Queue(10)
publish_queue = Queue(10)

occ_grid_width=10
occ_grid_height= 40
cell_size_in_m = 0.1
bev_tool =bev_transform_tools.fromJSON(bev_file)
#===========================================================================================================
# Create objects to subscribe into image topics

for orientations in params:
	print(orientations)
	# print(params[orientations])
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

def receive_image(orientation):
	# print("no treats?",orientation)
	img =sub_object[orientation].camera.get_bgr_frame()
	sleep(0.5)
	t0 = time()
	location = (orientation,t0)
	if(img is not None):
		try:
			preprocess_queue.put_nowait((img,location))
		except (Full):
			preprocess_queue.get()
			preprocess_queue.put((img,location))
def receive_image_work():
	while not rospy.is_shutdown():
		for orientation in ["front","rear","right","left"]:
			receive_image(orientation)
	
def receive_image_work_wrapper():
	asyncio.run(receive_image_work(),debug=True)
def preprocess_work():
	while not rospy.is_shutdown():
		bgr_img,location =  preprocess_queue.get()
		print("time to transfer data between process via queue",time()-location[1])
		pp_img = model.preprocess(bgr_img)
		try:
			predict_queue.put_nowait((pp_img,location))
		except (Full):
			predict_queue.get()
			predict_queue.put((pp_img,location))
def predict_work():
	while not rospy.is_shutdown():
		pp_img,location = predict_queue.get()
		segmap = model.predict(pp_img)
		try:
			bev_queue.put_nowait((segmap,location))
		except (Full):
			bev_queue.get()
			bev_queue.put((segmap,location))
def bev_work():
	while not rospy.is_shutdown():
		segmap,location = bev_queue.get()
		segmap =cv2.resize(segmap,(bev_tool.input_height,bev_tool.input_width))
		occgrid = bev_tool.create_occupancy_grid(segmap,occ_grid_width,occ_grid_height,cell_size_in_m)
		try:
			publish_queue.put_nowait((occgrid,location))
		except (Full):
			publish_queue.get()
			publish_queue.put((occgrid,location))
def publish_work():
	pose = (0,0,0,0,0,0)
	while not rospy.is_shutdown():
		occ_grid,location = publish_queue.get()
		orientation,t0 = location
		print("FPS for {} is {}".format(orientation,1/(time()-t0)))
		msg = convert_to_occupancy_grid_msg(occ_grid,cell_size_in_m,occ_grid_width,occ_grid_height,rospy.Time.from_sec(t0),"base_link",pose)
		sub_object[orientation].publish(msg)

pp_thread =  Process(target=preprocess_work)
bev_thread = Process(target=bev_work)
publish_thread = Process(target=publish_work)
receive_img_thread = Process(target=receive_image_work)
receive_img_thread.start()
pp_thread.start()
bev_thread.start()
publish_thread.start()


rospy.init_node('camera360', anonymous=True,disable_signals=True)
r = rospy.Rate(30)
while not rospy.is_shutdown():
	pp_img,location = predict_queue.get()
	segmap = model.predict(pp_img)
	try:
		bev_queue.put_nowait((segmap,location))
	except (Full):
		bev_queue.get()
		bev_queue.put((segmap,location))
	r.sleep()


