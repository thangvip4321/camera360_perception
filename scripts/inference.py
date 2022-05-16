#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
#!/home/bugcar/miniconda3/envs/tf2.4/bin/python
import logging
import os
import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from camera360_perception.bugcar_image_segmentation.bev import bev_transform_tools
from camera360_perception.bugcar_image_segmentation.image_processing_utils import contour_noise_removal
from camera360_perception.bugcar_image_segmentation.models import ENET
from camera360_perception.bugcar_image_segmentation.occgrid_to_ros import convert_to_occupancy_grid_msg
from camera360_perception.camera_handlers.insta360 import Insta360Camera, Insta360StitchCamera
import tensorflow as tf
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage
import cv2
import argparse
# should we create a factory to create stitcher
from camera360_perception.stitcher import Stitcher
import cv_bridge

# parsing any arguments coming in the program
parser = argparse.ArgumentParser()
parser.add_argument("-v","--verbose",
                    help="enable debugging information", type=bool, default=False)
args, unknowns = parser.parse_known_args()
print(args)

logger = logging.getLogger(__name__)
if(args.verbose):
	log_level = logging.DEBUG
else:
	log_level = logging.WARNING	
logger.setLevel(log_level)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(process)s-%(threadName)s-%(funcName)s_%(lineno)d: %(message)s')
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)
class ROS_handler:
	def __init__(self, publishing_topic,camera):
		self.bridge = cv_bridge.CvBridge()
		self.publishing_topic  = publishing_topic
		self.occ_grid_publisher = rospy.Publisher(self.publishing_topic, OccupancyGrid, queue_size=1)
		self.compressed_img_publisher = rospy.Publisher("image_"+orientations+"/compressed", CompressedImage, queue_size=1)
		self.camera = camera

	def publish_bev(self, occupancy_grid):
		self.occ_grid_publisher.publish(occupancy_grid)
	def publish_camera(self, img):
		img_msg = self.bridge.cv2_to_compressed_imgmsg(img)
		self.compressed_img_publisher.publish(img_msg)

rospy.init_node('camera360', anonymous=True,disable_signals=True)
params = rospy.get_param('camera360_insta')
orientations = rospy.get_param('~orientation')
model_path = rospy.get_param('model_path')
model = ENET(model_path)
sub_object            = {}
cell_size_in_m = 0.1
#===========================================================================================================
# Create objects to subscribe into image topics
logger.debug("{}".format(params))
lens =params[orientations]["sub_topics"]
pub_topic_name = params[orientations]['pub_topics']
stitch_file = params[orientations]['stitch_calib_file']
camera_file = params[orientations]['camera_intrinsic_file_path']
stitcher = None



assert (len(lens)==2) == (stitch_file !=""), "either 2 topic and 1 stitch file or 1 topic and 0 stitch file"
if(len(lens)==2):
	stitcher = Stitcher.createFromYaml(stitch_file=stitch_file)
	sub_object.update({orientations: ROS_handler(pub_topic_name,Insta360StitchCamera(lens[0],lens[1],camera_file,stitcher=stitcher))})
else:
	sub_object.update({orientations: ROS_handler(pub_topic_name,Insta360Camera(lens[0],camera_file))})

bev_file = params[orientations]['bev_calib_file']
frame_id = params[orientations]['base_frame']
pose = params[orientations]['pose_to_base_frame']
occ_grid_width,occ_grid_height=params[orientations]['occ_grid_size']

bev_tool =bev_transform_tools.fromJSON(bev_file)
FPS_LIMIT = 30
# stitch cost 4% more cpu
# publish and convert to occupancy grid cost 5% more
r = rospy.Rate(15)
while not rospy.is_shutdown():
	t0 = time.time()
	img =sub_object[orientations].camera.get_bgr_frame()
	if(logger.level <= logging.DEBUG):
		sub_object[orientations].publish_camera(img)
		cv2.imshow("img "+orientations,img)
		cv2.waitKey(1)
	logger.debug("image shape {}".format(img.shape))
	if (img is not None):
		pass
	else:
		continue 
	t1 = time.time()
	logger.debug("get time {}".format(t1-t0))

	# #=======================================================================================================
	# # Pre-process before feeding into deep learning model
	front_pp =  model.preprocess(img)

	#======================================================================================================
	t1 = time.time()
	front_segmented = model.predict(front_pp)[0]
	t2 = time.time()
	logger.info("inference time: {}".format(t2-t1))
	if(logger.level <= logging.DEBUG):
		cv2.imshow("segmented "+orientations,(front_segmented*100))
	front_segmented = contour_noise_removal(front_segmented)
	# cost 1% more cpu, contour
	logger.debug("shape of segmented image {}, {}".format(front_segmented.shape,img.shape))
	t = rospy.Time.now()
	resized_img = cv2.resize(front_segmented,(bev_tool.input_height,bev_tool.input_width))
	# cv2 resize cost  1-2% cpu

	og = bev_tool.create_occupancy_grid(resized_img,occ_grid_width,occ_grid_height,cell_size_in_m)
	occ_grid_front = convert_to_occupancy_grid_msg(og,cell_size_in_m,occ_grid_width,occ_grid_height,t,frame_id,pose)
	sub_object[orientations].publish_bev(occ_grid_front)
	fps = 1/(time.time()-t0)
	logger.info("fps: {}".format(fps))
	logger.info("publish time {}".format(time.time()-t2))

	r.sleep()
	# if (fps>FPS_LIMIT):
	# 	time.sleep(1/FPS_LIMIT-1/fps)
	# 	fps = 1/(time.time()-t0)
	#=======================================================================================================
