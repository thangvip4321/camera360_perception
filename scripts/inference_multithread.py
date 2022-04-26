#!/home/bugcar/anaconda3/envs/t2.2/bin/python
import logging
import os
import signal
import traceback
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from time import time
import numpy as np
import yaml
from bugcar_image_segmentation.models import ENET
import tensorflow as tf
import rospy
# rospy.init_node('damn_som')
# print("hi")
# print(rospy.get_param("ss"))
from cameraType.baseCamera import BaseCamera
from nav_msgs.msg import OccupancyGrid
from queue import Empty, Full 
from multiprocessing import  Process
from faster_fifo import Queue
import faster_fifo_reduction
# print("hi mom")
# logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
class ROS_handler:
	def __init__(self, publishing_topic,camera:BaseCamera):
		self.publishing_topic  = publishing_topic
		self.Publisher = rospy.Publisher(self.publishing_topic, OccupancyGrid, queue_size=8)
		self.camera = camera

	def publish(self, occupancy_grid):
		self.Publisher.publish(occupancy_grid)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_memory_growth(gpus[0], True)
		tf.config.experimental.set_virtual_device_configuration(gpus[0],
																[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
	except RuntimeError as e:
		print(e)



# print("hi dad")
params = rospy.get_param('camera360_insta')
sub_object       = {}
inference_queues = {}

# print("hi mom")


#===========================================================================================================
# Create objects to subscribe into image topics
def interrupt_handler(a,b):
	rospy.signal_shutdown("haha")
	for job in all_processes:
		job.kill()
	exit(0)
def interrupt_child_handler(a,b):
	print("ok")
	rospy.signal_shutdown("haha")
	exit(0)



def read_and_process(orientation):
	from bugcar_image_segmentation.bev import bev_transform_tools
	from bugcar_image_segmentation.occgrid_to_ros import convert_to_occupancy_grid_msg
	from cameraType.ros_camera import ROSCamera, ROSStitchCamera
	from bugcar_image_segmentation.image_processing_utils import contour_noise_removal
	from stitcher import Stitcher
	import cv2
	logger = logging.getLogger(__name__)
	logger.debug("orientation 97 %s",orientation)
	rospy.init_node('read_camera_{}'.format(orientation), anonymous=True,disable_signals=True)
	bev_file = params[orientation]['bev_calib_file']
	frame_id = params[orientation]['base_frame']
	pose = params[orientation]['pose_to_base_frame']
	occ_grid_width,occ_grid_height=params[orientation]['occ_grid_size']
	cell_size_in_m = 0.1
	bev_tool =bev_transform_tools.fromJSON(bev_file)
	q = inference_queues[orientation]

	pub_topic_name = params[orientation]["pub_topics"]
	lens =params[orientation]["sub_topics"]
	stitch_file = params[orientation]['stitch_calib_file']
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
	signal.signal(signal.SIGINT,interrupt_child_handler)
	while not rospy.is_shutdown():
		logger.debug("getting the camera")
		img =sub_object[orientation].camera.get_bgr_frame()
		t0 = time()
		# cv2.imshow("read img {}".format(orientation),img)
		# cv2.waitKey(1)
		location = (orientation,t0)
		# #=======================================================================================================
		# # Pre-process before feeding into deep learning model
		front_pp =  ENET.preprocess(img)
		#======================================================================================================
		t1 = time()
		logger.debug("time for preprocessing %f",t1-t0)

		try:
			predict_queue.put((front_pp,location))
			# t2 = time()
			# print("time for putting",t2-t1)	
			segmap,location = q.get_many(max_messages_to_get=5)[0]
			# t3 = time()
		except (Full,Empty):
			traceback.print_exc()
			continue
		t2 = time()
		logger.debug("time for getting %f",t2-t1)
		segmap = contour_noise_removal(segmap)
		t = rospy.Time.now()
		resized_img = cv2.resize(segmap,(bev_tool.input_height,bev_tool.input_width))
		occ_grid_front = convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(resized_img,occ_grid_width,occ_grid_height,cell_size_in_m),\
													cell_size_in_m,occ_grid_width,occ_grid_height,t,frame_id,pose)
		sub_object[orientation].publish(occ_grid_front)
		logger.debug("time for publishing %f",time()-t2)

		# print("FPS: ",orientation,1/(time()-location[1]))
		r.sleep()


if __name__ == "__main__":
	print("hi mom")
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	formatter = logging.Formatter('%(process)s-%(threadName)s-%(funcName)s_%(lineno)d: %(message)s')
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	camera_file = rospy.get_param("camera_intrinsic_file_path")


	
	
	predict_queue = Queue(2000000*10)
	all_processes = []
	for orientation in params:
		inference_queues[orientation] = Queue(2000000*10)
		read_and_preprocess_subproc = Process(target=read_and_process,args=[orientation])
		all_processes.append(read_and_preprocess_subproc)
	for i in all_processes:
		i.start()






	model_path = rospy.get_param('model_path')
	model = ENET(model_path)
	rospy.init_node('clgt', anonymous=True,disable_signals=True)
	rate = rospy.Rate(25)
	while not rospy.is_shutdown():
		t1 = time()
		try:
			arr_img_and_loc = predict_queue.get_many(max_messages_to_get=5)
		except Empty:
			traceback.print_exc()
			rate.sleep()
			continue
		# print(arr_img_and_loc)
		arr_img = []
		for img,loc in arr_img_and_loc:
			arr_img.append(img)
		batch = np.concatenate(arr_img)
		logger.debug("batch shape {}".format(batch.shape))
		t2 = time()
		# the time for collecting image is negligible
		# print("collect images for prediction time",t2-t1)
		segmaps = model.predict(batch)
		t3 = time()
		logger.debug("inference time %f",(t3-t2))
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
		# print("time until the last can send",time()-t3)
		rate.sleep()
	signal.signal(signal.SIGINT,interrupt_handler)


