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
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
    except RuntimeError as e:
        print(e)
class ROS_handler:
	def __init__(self, publishing_topic,camera:BaseCamera):
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
def process_work(**kwargs):
	orientation = kwargs.pop('orientation')
	case = []
	try:
		while True:
			img =sub_object[orientation].camera.get_bgr_frame()
			if(img is not None):
				t0 = time()
				pp_img =  model.preprocess(img)
				segmap = model.predict(pp_img)
				t = rospy.Time.now()
				pose = (0,0,0,0,0,0)
				occ_grid_front = convert_to_occupancy_grid_msg(bev_tool.create_occupancy_grid(segmap,occ_grid_width,occ_grid_height,cell_size_in_m)\
				,cell_size_in_m,occ_grid_width,occ_grid_height,t,pose)
				sub_object[orientation].publish(occ_grid_front)
				t = 1/(time()-t0)
				case.append(t)	
				print("fps for {} camera: {}".format(orientation,t))
	finally:
		case_np = np.asarray(case)
		print("the mean fps of {} camera is {}".format(orientation,np.mean(case_np)))
		sub_object[orientation].camera.stop()

bev_tool =bev_transform_tools.fromJSON("bugcar_image_segmentation/calibration_data.json")
list_thread = []
for orientations in params:
	print("haha",orientations)
	pub_topic_name = params[orientations]["pub_topics"]
	lens =params[orientations]["sub_topics"]
	num_topics = len(lens)
	if(num_topics==2):
		sub_object.update({orientations: ROS_handler(pub_topic_name,ROSStitchCamera(lens[0],lens[1]))})
	elif(num_topics==1): # this mean there is only 1 camera
		sub_object.update({orientations: ROS_handler(pub_topic_name,ROSCamera(lens[0]))})
	else:
		pass
	list_thread.append(threading.Thread(target=process_work,kwargs={"orientation":orientations}))
for thr in list_thread:
	print("a")
	thr.start()
rospy.spin()
# list_thread[0].join()

