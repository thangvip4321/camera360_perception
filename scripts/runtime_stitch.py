#!/home/bugcar/miniconda3/envs/tf2.4/bin/python
#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
import cv2
import numpy as np
import time

import yaml
# import calibrate_camera_with_checkerboard
# from stitcher import Stitcher
from camera360_perception.camera_handlers.insta360 import Insta360StitchCamera
import rospy
from camera360_perception.stitcher import Stitcher
from sensor_msgs.msg import CompressedImage 
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

#===========================================================================


#===========================================================================
print("module name is: ",__name__)
if (__name__=="__main__"):
	bridge = CvBridge()
	rospy.init_node("image_stitching", anonymous=True,disable_signals=True)
	left_img_input = rospy.get_param("~left_img_input")
	homography_matrix = np.reshape(np.array(rospy.get_param("~homography_matrix")),(3,3))
	crop_x = rospy.get_param("~crop_area_x")
	crop_y = rospy.get_param("~crop_area_y")
	right_img_input  = rospy.get_param("~right_img_input")
	camera_info_file = rospy.get_param("~camera_info_file")
	stitching_output = rospy.get_param("~stitching_output","~/output")
	pub = rospy.Publisher(stitching_output,CompressedImage,queue_size=3)
	cam = Insta360StitchCamera(left_img_input, right_img_input,camera_info_file,Stitcher(homography_matrix,(*crop_x,*crop_y)))
	# print("anybody here?")
	while(not rospy.is_shutdown()):
		t0 = time.time()
		# print("anybody here no 2?")
		stitched_img = cam.get_bgr_frame()
		# print("homo matrix",cam.stitcher.homo)
		if (stitched_img is None):
			continue
		
		# stitched_img,_,_ = data_pack
		# print("shape",stitched_img.shape)
		if (stitched_img.all() != None):
			msg = bridge.cv2_to_compressed_imgmsg(stitched_img)
			pub.publish(msg)
			# cv2.imshow("Stitched",  cv2.resize(stitched_img, None, fx=1.0, fy=1.0))
			# cv2.imshow("resized Stitched",  cv2.resize(stitched_img,(512,256)))

		key =cv2.waitKey(1)
		if(key== ord('q')):
			rospy.signal_shutdown("command")
		# print("Stitching FPS:  ", str(1/(time.time() - t0)))
	# cv2.destroyAllWindows()
