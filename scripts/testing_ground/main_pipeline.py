#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
from threading import Thread
from time import time
import traceback
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
from queue import Empty, Full 
from multiprocessing import Process
from faster_fifo import Queue
import faster_fifo_reduction

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
model_path = rospy.get_param('model_path')
model = ENET(model_path)
sub_object            = {}
bev_file = params["front"]['bev_calib_file']
frame_id = params["front"]['base_frame']
pose = params["front"]['pose_to_base_frame']
# receive_image_queue = Queue(10)
preprocess_queue = Queue(2000000*10)
predict_queue = Queue(2000000*10)
bev_queue = Queue(2000000*10)
publish_queue = Queue(2000000*10)

occ_grid_width=10
occ_grid_height= 40
cell_size_in_m = 0.1
bev_tool =bev_transform_tools.fromJSON(bev_file)
camera_file = rospy.get_param('camera_intrinsic_file_path')
#===========================================================================================================
# Create objects to subscribe into image topics

# for orientations in params:
# 	print(orientations)
# 	# print(params[orientations])
# 	pub_topic_name = params[orientations]["pub_topics"]
# 	lens =params[orientations]["sub_topics"]
# 	stitch_file = params[orientations]['stitch_calib_file']

# 	num_topics = len(lens)
# 	assert (len(lens)==2) == (stitch_file !=""), "either 2 topic and 1 stitch file or 1 topic and 0 stitch file"
# 	if(len(lens)==2):
# 		with open(stitch_file,"r") as f:
# 			obj = yaml.safe_load(f)
# 			homography_matrix = np.reshape(np.array(obj["homography_matrix"]),(3,3))
# 			crop_x = obj["crop_area_x"]
# 			crop_y = obj["crop_area_y"]
# 			stitcher = Stitcher(homography_matrix,(*crop_x,*crop_y))
# 			sub_object.update({orientations: ROS_handler(pub_topic_name,ROSStitchCamera(lens[0],lens[1],camera_file,stitcher))})
# 	else:
# 		sub_object.update({orientations: ROS_handler(pub_topic_name,ROSCamera(lens[0],camera_file))})
def get_cam_work(orientation):
	rospy.init_node('read_camera_{}'.format(orientation), anonymous=True,disable_signals=True)
	pub_topic_name = params[orientation]["pub_topics"]
	lens =params[orientation]["sub_topics"]
	stitch_file = params[orientation]['stitch_calib_file']
	print("haha",camera_file)
	assert (len(lens)==2) == (stitch_file !=""), "either 2 topic and 1 stitch file or 1 topic and 0 stitch file"
	if(len(lens)==2):
		with open(stitch_file,"r") as f:
			obj = yaml.safe_load(f)
			homography_matrix = np.reshape(np.array(obj["homography_matrix"]),(3,3))
			crop_x = obj["crop_area_x"]
			crop_y = obj["crop_area_y"]
			stitcher = Stitcher(homography_matrix,(*crop_x,*crop_y))
			sub_object.update({orientation: ROS_handler(pub_topic_name,ROSStitchCamera(lens[0],lens[1],camera_file,stitcher))})
	else:
		sub_object.update({orientation: ROS_handler(pub_topic_name,ROSCamera(lens[0],camera_file))})
	# print("ori in subproc read",orientation)
	r = rospy.Rate(20)
	while not rospy.is_shutdown():
		# logger.debug("getting the camera")
		img =sub_object[orientation].camera.get_bgr_frame()
		location = (orientation,time())
		preprocess_queue.put((img,location))

def preprocess_work():
	while True:
		bgr_img,location =  preprocess_queue.get()
		pp_img = ENET.preprocess(bgr_img)
		try:
			predict_queue.put_nowait((pp_img,location))
		except (Full):
			predict_queue.get()
			predict_queue.put((pp_img,location))

def bev_work():
	bev_dict = {}
	for orientation in ["front","rear","right","left"]:
		bev_file = params[orientation]['bev_calib_file']
	occ_grid_width,occ_grid_height=params[orientation]['occ_grid_size']
	cell_size_in_m = 0.1
	bev_tool =bev_transform_tools.fromJSON(bev_file)
	while True:
		segmap,location = bev_queue.get()
		segmap = cv2.resize(segmap,(bev_tool.input_height,bev_tool.input_width))
		occgrid = bev_tool.create_occupancy_grid(segmap,occ_grid_width,occ_grid_height,cell_size_in_m)
		try:
			publish_queue.put_nowait((occgrid,location))
		except (Full):
			publish_queue.get()
			publish_queue.put((occgrid,location))
def publish_work():
	rospy.init_node('publish_camera', anonymous=True,disable_signals=True)
	pose = (0,0,0,0,0,0)
	while True:
		occ_grid,location = publish_queue.get()
		orientation,t0 = location
		print("FPS for {} is {}".format(orientation,1/(time()-t0)))
		msg = convert_to_occupancy_grid_msg(occ_grid,cell_size_in_m,occ_grid_width,occ_grid_height,rospy.Time.from_sec(t0),"xxx",pose)
		sub_object[orientation].publish(msg)

pp_thread =  Process(target=preprocess_work)
bev_thread = Process(target=bev_work)
publish_thread = Process(target=publish_work)
get_cam_arr = []
for i in ["front","right","rear","left"]:
	get_cam_thread = Process(target=get_cam_work,args=[i])
	get_cam_arr.append(get_cam_thread)
	get_cam_thread.start()
pp_thread.start()
bev_thread.start()
publish_thread.start()


rospy.init_node('camera360', anonymous=True,disable_signals=True)
r = rospy.Rate(30)
while not rospy.is_shutdown():
	# for orientation in ["front"]:
	# 	img =sub_object[orientation].camera.get_bgr_frame()
	# 	t0 = time()
	# 	location = (orientation,t0)
	# 	if(img is not None):
	# 		try:
	# 			preprocess_queue.put_nowait((img,location))
	# 		except (Full):
	# 			preprocess_queue.get()
	# 			preprocess_queue.put((img,location))
	try:
		arr_img_and_loc = predict_queue.get_many(max_messages_to_get=5)
	except Empty:
		traceback.print_exc()
		r.sleep()
		continue
	# print(arr_img_and_loc)
	arr_img = []
	for img,loc in arr_img_and_loc:
		arr_img.append(img)
	batch = np.concatenate(arr_img)
	print(batch.shape)
	tx = time()
	segmap = model.predict(batch)
	print("inference time: ",time()-tx)
	# print(pp_img.shape)
	for i,img in enumerate(segmap):
			# print(i,arr_img_and_loc[i])
			location = arr_img_and_loc[i][1]
			orientation = location[0]
			try:
				bev_queue.put_nowait((img,location))
			except (Full):
				bev_queue.get()
				bev_queue.put((img,location))
	# try:
	# 	bev_queue.put_nowait((segmap,location))
	# except (Full):
	# 	bev_queue.get()
	# 	bev_queue.put((segmap,location))
	r.sleep()


