#!/home/bugcar/miniconda3/envs/tf2.2/bin/python
import logging
import os
import time
import yaml
from bugcar_image_segmentation.bev import bev_transform_tools
from bugcar_image_segmentation.image_processing_utils import contour_noise_removal
from bugcar_image_segmentation.models import ENET
from bugcar_image_segmentation.occgrid_to_ros import convert_to_occupancy_grid_msg
from cameraType.baseCamera import BaseCamera
from cameraType.ros_camera import ROSCamera, ROSStitchCamera
import tensorflow as tf
import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from stitcher import Stitcher


logger = logging.Logger('bff')
logger.setLevel(logging.INFO)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=550)])
    except RuntimeError as e:
        print(e)
class ROS_handler:
	def __init__(self, publishing_topic,camera:BaseCamera):
		self.publishing_topic  = publishing_topic
		self.Publisher = rospy.Publisher(self.publishing_topic, OccupancyGrid, queue_size=1)
		self.camera = camera

	def publish(self, occupancy_grid):
		self.Publisher.publish(occupancy_grid)

# print(os.getcwd())
rospy.init_node('camera360', anonymous=True,disable_signals=True)
params = rospy.get_param('camera360_insta')
orientations = rospy.get_param('~orientation')
model_path = rospy.get_param('model_path')
camera_file = rospy.get_param('camera_intrinsic_file_path')
model = ENET(model_path)
sub_object            = {}
cell_size_in_m = 0.1
#===========================================================================================================
# Create objects to subscribe into image topics

lens =params[orientations]["sub_topics"]
pub_topic_name = params[orientations]['pub_topics']

stitch_file = params[orientations]['stitch_calib_file']
stitcher = None

assert (len(lens)==2) == (stitch_file !=""), "either 2 topic and 1 stitch file or 1 topic and 0 stitch file"
if(len(lens)==2):
	with open(stitch_file,"r") as f:
		obj = yaml.safe_load(f)
		homography_matrix = np.reshape(np.array(obj["homography_matrix"]),(3,3))
		crop_x = obj["crop_area_x"]
		crop_y = obj["crop_area_y"]
		stitcher = Stitcher(homography_matrix,(*crop_x,*crop_y))
	sub_object.update({orientations: ROS_handler(pub_topic_name,ROSStitchCamera(lens[0],lens[1],camera_file,stitcher=stitcher))})
else:
	sub_object.update({orientations: ROS_handler(pub_topic_name,ROSCamera(lens[0],camera_file))})

bev_file = params[orientations]['bev_calib_file']
frame_id = params[orientations]['base_frame']
pose = params[orientations]['pose_to_base_frame']
occ_grid_width,occ_grid_height=params[orientations]['occ_grid_size']

bev_tool =bev_transform_tools.fromJSON(bev_file)

FPS_LIMIT = 20

while not rospy.is_shutdown():
	t0 = time.time()
	img =sub_object[orientations].camera.get_bgr_frame()
	if(logger.level <= logging.INFO):
		cv2.imshow("img "+orientations,img)
		cv2.waitKey(1)
	logger.info("image shape",img.shape)
	if (img is not None):
		pass
	else:
		continue 

	# #=======================================================================================================
	# # Pre-process before feeding into deep learning model
	front_pp =  model.preprocess(img)
	#======================================================================================================
	front_segmented = model.predict(front_pp)
	if(logger.level <= logging.INFO):
		cv2.imshow("segmented "+orientations,(front_segmented*100).astype(np.uint8))
	front_segmented = contour_noise_removal(front_segmented)
	logger.info(front_segmented.shape,img.shape)
	t = rospy.Time.now()
	resized_img = cv2.resize(front_segmented,(img.shape[1],img.shape[0]))
	occ_grid_front = convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(resized_img,occ_grid_width,occ_grid_height,cell_size_in_m),\
												   cell_size_in_m,occ_grid_width,occ_grid_height,t,frame_id,pose)
	
	sub_object[orientations].publish(occ_grid_front)
	fps = 1/(time.time()-t0)

	if (fps>FPS_LIMIT):
		time.sleep(1/FPS_LIMIT-1/fps)
		fps = 1/(time.time()-t0)
	# print("fps:" ,fps)
	#=======================================================================================================