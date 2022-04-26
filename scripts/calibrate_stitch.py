#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
#!/home/thang/miniconda3/envs/bugcar/bin/python
import signal
import cv2
import numpy as np
import time

import yaml

from cameraType.ros_camera import ROSStitchCamera
import rospy
from stitcher import Stitcher
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

cam = None
H = None
#===========================================================================
def keyboardInterruptHandler(signal, frame):
	global cam
	H = cam.stitcher.calcHomographyMatrix(4.0)
	warped_part = (cam.imgRight.shape[0],cam.imgRight.shape[1]*0.4,3)
	shape_left_img = (cam.imgLeft.shape[0],cam.imgRight.shape[1]*5.8,3)
	lower_y_crop,upper_y_crop,x_crop = findCropArea(warped_part,shape_left_img)
	data = dict(homography_matrix=H.tolist(),crop_area_x = [0,x_crop],crop_area_y = [upper_y_crop,lower_y_crop])
	if(H is not None):
		with open(output_config_file, 'w+') as outfile:
			yaml.dump(data, outfile, default_flow_style=False)
	print(H)
	cv2.destroyAllWindows()

	exit(0)
    #print("call your function here".format(signal))exit(0)
def findIntersection(line1, line2):
		# each line include 2 points, which have shape (2,nd), where nd is the dimension
		# convert it to homogenous coordinates, and the problem become finding the intersection of 2 planes?
		# add 1 to the back of the line
		# https://code-examples.net/en/q/719e6e
		line1=cv2.convertPointsToHomogeneous(line1)
		# print(line1,line2)
		homo_coord_line1 = np.cross(line1[0], line1[1])
		line2=cv2.convertPointsToHomogeneous(line2)
		homo_coord_line2 = np.cross(line2[0], line2[1])
		intersection = np.cross(homo_coord_line2,homo_coord_line1)
		intersection = cv2.convertPointsFromHomogeneous(intersection)
		return intersection[0][0]
		# line2 = np.concatenate(line2,np.array(1,1))
def findCropArea(shape_left_img):
	w = cam.imgLeft.shape[1]
	imageB = cam.imgLeft[:, 0:int(w*5.8/10)]
	imageA = cam.imgRight[:, int(w*4/10):w] 
	h,w,_ = imageA.shape
	result = cv2.warpPerspective(imageA, H,
			(1500,1000))
	result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
	# when stitching 2 image, denote imageA the one we are warping (which is on the right), and imageB the one on the left
	# cv2.imshow("image A",imageA)
	# print(warped_part)		
	imageA_corners = np.transpose(np.array([[0,0,1],[0,h,1],[w,h,1],[w,0,1]]))
	print("corner",imageA_corners)
	warped_corners = cv2.convertPointsFromHomogeneous(np.transpose(np.matmul(H,imageA_corners)))[:,0]
	for i in warped_corners:
		result = cv2.circle(result,(int(i[0]),int(i[1])),3,(0,0,255),-1)
	cv2.imshow("hehe",result)
	print(warped_corners)
	seam_line = np.array([[shape_left_img[1],0],[shape_left_img[1],h]])	
	top_edge_A = warped_corners[[0,3],:]
	bottom_edge_A = warped_corners[[1,2],:]
	right_edge_A = warped_corners[[2,3],:]
	
	x = findIntersection(top_edge_A,seam_line)
	y = findIntersection(bottom_edge_A,seam_line)
	top_right_corner_A = warped_corners[3]
	bottom_right_corner_A = warped_corners[2]
	imageA_new_corners = [x,y,top_right_corner_A,bottom_right_corner_A]

	# sort by y

	imageA_new_corners.sort(key=lambda pts : pts[1])
	# take the middle 2 point and take their y
	imageA_new_corners[1][1] = np.clip(imageA_new_corners[1][1],0,np.Inf)
	p1= imageA_new_corners[1]
	for i in imageA_new_corners:
		result = cv2.circle(result,(int(i[0]),int(i[1])),3,(0,0,255),-1)
	p11 = (0,imageA_new_corners[1][1])
	line_p1 = np.array([p1,p11])
	p2= imageA_new_corners[2]
	p21 = (0,imageA_new_corners[2][1])
	line_p2 = np.array([p2,p21])
	intersection_p1 = findIntersection(line_p1,right_edge_A)
	intersection_p2 = findIntersection(line_p2,right_edge_A)
	result = cv2.circle(result,(int(intersection_p1[0]),int(intersection_p1[1])),3,(0,255,0),-1)
	result = cv2.circle(result,(int(intersection_p2[0]),int(intersection_p2[1])),3,(0,255,0),-1)
	x_crop = int(min(intersection_p1[0],intersection_p2[0]))
	cv2.imshow("result",result)
	cv2.waitKey(0)
	# upper_y_crop = np.clip(int(p1[1]),0,np.Inf).astype(np.int32)
	upper_y_crop =int(p1[1])
	lower_y_crop = int(p2[1])
	return lower_y_crop,upper_y_crop,x_crop

#===========================================================================
print("module name is: ",__name__)
if (__name__=="__main__"):
	signal.signal(signal.SIGINT, keyboardInterruptHandler)
	rospy.init_node("image_stitching", anonymous=True,disable_signals=True)
	left_img_input = rospy.get_param("~left_img_input")
	print(left_img_input)
	right_img_input 	= rospy.get_param("~right_img_input")
	output_config_file = rospy.get_param("~output_config_file", \
										"/camera360/left/right_img_input")
	camera_info_file = rospy.get_param("~camera_info_file", \
										"?")
	cam = ROSStitchCamera(left_img_input, right_img_input,camera_info_file,Stitcher())
	while(not rospy.is_shutdown()):
		t0 = time.time()
		print("is everythin ok")
		data_pack = cam.get_stitch_image_with_matching_features()
		if (data_pack is None):
			print("no data")
			continue
		stitched_img,matchpoints_left_img,matchpoints_right_img = data_pack

		if (stitched_img.all() != None):

			stitched_norm = stitched_img / 255
			# print(stitched_norm.shape)
			# print(stitched_gray.dtype)
			sobelx_0 = cv2.Sobel(stitched_norm[:,:,0], cv2.CV_64F, 1,0,ksize=3)
			sobelx_1 = cv2.Sobel(stitched_norm[:,:,1], cv2.CV_64F, 1,0,ksize=3)
			sobelx_2 = cv2.Sobel(stitched_norm[:,:,2], cv2.CV_64F, 1,0,ksize=3)
			sobelx  = np.abs(sobelx_0 + sobelx_1 + sobelx_2)/3

			h_stitched, w_stitched,_ = stitched_norm.shape
			print(stitched_norm.shape)
			left_img_cropped_width = int(cam.imgLeft.shape[1]*5.8/10)
			print(cam.imgLeft.shape)

			crop_sobel = sobelx[:int(h_stitched*0.9),left_img_cropped_width-1]
			cv2.imshow("stitched_img",stitched_img)
			# print("sobel is",len(crop_sobel[crop_sobel<np.max(crop_sobel)/3]))
			print("sobel is",np.sum(crop_sobel),h_stitched)

#============================ beside sobel we also consider the rotation between 2 camera. Ideally it should be around (0,0,0)
			possible_relative_pose_solution = cv2.decomposeHomographyMat(cam.stitcher.homo,cam.get_intrinsic_matrix())
			rotations = possible_relative_pose_solution[1]
			satisfying_orientation=False
			for rotation in rotations:
				r = R.from_matrix(rotation).as_euler("xyz",degrees=True)
				print(r)
				if(np.linalg.norm(r-np.array([0,0,0])) <10):
					satisfying_orientation=True
					break

			if 	(np.sum(crop_sobel) < h_stitched*0.1*0.9 and satisfying_orientation):
					# print("good image quality")
					cam.stitcher.list_ptsA.append(matchpoints_left_img)
					cam.stitcher.list_ptsB.append(matchpoints_right_img)
					cv2.imshow("good stitch",stitched_img)
					cv2.imshow("sobel", 	cv2.resize(sobelx, None, fx=0.5, fy=0.5))
					cv2.imshow("crop_sobel", crop_sobel)
		key =cv2.waitKey(1)
		if(key== ord('q')):
			rospy.signal_shutdown("command")
		# print("Stitching FPS:  ", str(1/(time.time() - t0)))
	H = cam.stitcher.calcHomographyMatrix(5.0)

	
	#=========================find crop_area code here=========================================================
	warped_part = (cam.imgRight.shape[0],int(cam.imgRight.shape[1]*0.6),3)
	shape_left_img = (cam.imgLeft.shape[0],int(cam.imgRight.shape[1]*0.58),3)
	lower_y_crop,upper_y_crop,x_crop = findCropArea(shape_left_img)
	data = dict(homography_matrix=H.tolist(),crop_area_x = [0,x_crop],crop_area_y = [upper_y_crop,lower_y_crop])
	if(H is not None):
		with open(output_config_file, 'w+') as outfile:
			yaml.dump(data, outfile, default_flow_style=False)
	print(H)
	cv2.destroyAllWindows()
