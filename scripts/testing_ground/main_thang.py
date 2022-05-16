from time import time
import numpy as np
import cv2
from bugcar_image_segmentation.bev import bev_transform_tools
from bugcar_image_segmentation.models import ENET
from bugcar_image_segmentation.occgrid_to_ros import convert_to_occupancy_grid_msg
from cameraType.baseCamera import BaseCamera
from cameraType.ros_camera import ROSCamera, ROSStitchCamera
import tensorflow as tf
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from queue import Full, Queue 
import threading
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

rospy.init_node('camera360', anonymous=True,disable_signals=True)
params = rospy.get_param('camera360')
model = ENET("./bugcar_image_segmentation/pretrained_models/enet.pb")
sub_object            = {}

# receive_image_queue = Queue(10)
preprocess_queue = Queue(10)
predict_queue = Queue(10)
bev_queue = Queue(10)
publish_queue = Queue(10)

occ_grid_width=10
occ_grid_height= 40
cell_size_in_m = 0.1
#===========================================================================================================
# Create objects to subscribe into image topics

for orientations in params:
	pub_topic_name = params[orientations]["pub_topics"]
	lens =params[orientations]["sub_topics"]
	num_topics = len(lens)
	if(num_topics==2):
		sub_object.update({orientations: ROS_handler(pub_topic_name,ROSStitchCamera(lens[0],lens[1]))})
	elif(num_topics==1): # this mean there is only 1 camera
		sub_object.update({orientations: ROS_handler(pub_topic_name,ROSCamera(lens[0]))})
	else:
		pass
def preprocess_work():
	while True:
		bgr_img,orientation =  preprocess_queue.get()
		pp_img = model.preprocess(bgr_img)
		try:
			predict_queue.put_nowait((pp_img,orientation))
		except (Full):
			predict_queue.get()
			predict_queue.put((pp_img,orientation))
def predict_work():
	while True:
		pp_img,orientation = predict_queue.get()
		segmap = model.predict(pp_img)
		try:
			bev_queue.put_nowait((segmap,orientation))
		except (Full):
			bev_queue.get()
			bev_queue.put((segmap,orientation))
def bev_work():
	while True:
		segmap,orientation = bev_queue.get()
		occgrid = bev_tool.create_occupancy_grid(segmap,occ_grid_width,occ_grid_height,cell_size_in_m)
		try:
			publish_queue.put_nowait((occgrid,orientation))
		except (Full):
			publish_queue.get()
			publish_queue.put((occgrid,orientation))
def publish_work():
	pose = (0,0,0,0,0,0)
	while True:
		occ_grid,orientation = publish_queue.get()
		msg = convert_to_occupancy_grid_msg(occ_grid,cell_size_in_m,occ_grid_width,occ_grid_height,pose)
		sub_object[orientation].publish(msg)
bev_tool =bev_transform_tools.fromJSON("bugcar_image_segmentation/calibration_data.json")

pp_thread =  threading.Thread(preprocess_work)
predict_thread = threading.Thread(predict_work)
bev_thread = threading.Thread(bev_work)
publish_thread = threading.Thread(publish_work)

r = rospy.Rate(30)
while not rospy.is_shutdown():
	front_img =sub_object["front"].camera.get_bgr_frame()
	rear_img = sub_object["rear"].camera.get_bgr_frame()
	left_img = sub_object["left"].camera.get_bgr_frame()
	right_img = sub_object["right"].camera.get_bgr_frame()
	if (front_img is not None and rear_img is not None \
		and left_img is not None and right_img is not None):
		pass
	else:
		continue 
	# cv2.imshow("img",right_img)
	t0 = time()

	#=======================================================================================================
	# Pre-process before feeding into deep learning model
	# while True:
	front_pp =  model.preprocess(front_img)
	rear_pp  = model.preprocess(rear_img)
	left_pp  = model.preprocess(left_img)
	right_pp =  model.preprocess(right_img)
	
	#=======================================================================================================
	# Running deep learning inference
	# case = []
	# try:
	# 	while (True):
	# 		front_segmented = model.predict(front_pp)
	# 		t = 1/(time()-t0)
	# 		print("fps:" ,t)
	# 		case.append(1/(time()-t0))
	# finally:
	# 	case_np = np.asarray(case)
	# 	print(np.mean(case_np))
	
	front_segmented = model.predict(front_pp)
	rear_segmented  = model.predict(rear_pp)
	left_segmented  = model.predict(left_pp)
	right_segmented = model.predict(right_pp)
	# front_segmented = np.where(front_segmented==0,1.0,0)
	t = rospy.Time.now()
	pose = (0,0,0,0,0,0)
	print(front_segmented.dtype)
	# cv2.imshow("front",front_segmented.astype(np.float32))

	occ_grid_front = convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(front_segmented,occ_grid_width,occ_grid_height,cell_size_in_m)\
	,cell_size_in_m,occ_grid_width,occ_grid_height,t,pose)
	occ_grid_rear= convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(rear_segmented,occ_grid_width,occ_grid_height,cell_size_in_m)\
	,cell_size_in_m,occ_grid_width,occ_grid_height,t,pose)
	occ_grid_left= convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(left_segmented,occ_grid_width,occ_grid_height,cell_size_in_m)\
	,cell_size_in_m,occ_grid_width,occ_grid_height,t,pose)
	occ_grid_right= convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(right_segmented,occ_grid_width,occ_grid_height,cell_size_in_m)\
	,cell_size_in_m,occ_grid_width,occ_grid_height,t,pose)

	sub_object["front"].publish(occ_grid_front)
	sub_object["rear"].publish(occ_grid_rear)
	sub_object["left"].publish(occ_grid_left)
	sub_object["right"].publish(occ_grid_right)
	t = 1/(time()-t0)
	print("fps:" ,t)
	r.sleep()
	#=======================================================================================================
	# Contour noise removal

	# front_pp = model.predict(pinhole_stitched["front"])
	# rear_pp  = model.predict(pinhole_stitched["rear"])
	# left_pp  = model.predict(pinhole_stitched["left"])
	# right_pp = model.predict(pinhole_stitched["right"])

	#=======================================================================================================
	# Update camera new_image flag back to False

	# (sub_object["front"]["front_lens"]).new_image = False
	# (sub_object["rear"]["rear_lens"]).new_image   = False
	# (sub_object["left"]["front_lens"]).new_image  = False
	# (sub_object["left"]["rear_lens"]).new_image   = False
	# (sub_object["right"]["front_lens"]).new_image = False
	# (sub_object["left"]["rear_lens"]).new_image   = False 