#!/home/bugcar/miniconda3/envs/tf2.4/bin/python
#!/home/tranquockhue/anaconda3/envs/deep_planner/bin/python
#!/home/thang/miniconda3/envs/bugcar/bin/python

from distutils.debug import DEBUG
import signal
import cv2
import numpy as np
import time


from camera360_perception.camera_handlers.insta360 import createInsta360Stream
import rospy

from scipy.spatial.transform import Rotation as R
import logging
import tf2_ros
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(process)s-%(threadName)s-%(funcName)s_%(lineno)d: %(message)s')
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
cam = None
H = None
#===========================================================================
def keyboardInterruptHandler(signal, frame):
	cv2.destroyAllWindows()
	exit(0)
	#print("call your function here".format(signal))exit(0)
def create_4d_mat_from_R_and_t(R,t):
	Mat = np.identity(4)
	Mat[:3,:3]= R
	# print(Mat)
	# print(Mat[0:3])
	Mat[0:3,3] = t
	return Mat
def reject_outliers(data, m = 2.):
	d = np.abs(data - np.median(data))
	print(d)
	mdev = np.median(d,axis=0)
	s = d/mdev if mdev else 0.
	return data[s<m]
#===========================================================================
print("module name is: ",__name__)
if (__name__=="__main__"):
	signal.signal(signal.SIGINT, keyboardInterruptHandler)
	rospy.init_node("image_stitching", anonymous=True,disable_signals=True)
	left_img_input = rospy.get_param("~left_img_input")
	orientation_projection = left_img_input.split("/")[3]
	camera_name = left_img_input.split("/")[1]
	lens_name = left_img_input.split("/")[2]
	leftImg_projection_frame_name = "{}_{}_frame".format(orientation_projection,lens_name)
	leftImg_lens_link_frame_name = "{}_{}_link".format(camera_name,lens_name)
# left_projection_front_lens_frame camera360_front_lens_link left_projection_front_lens_optical_frame
	logger.debug(left_img_input)
	right_img_input 	= rospy.get_param("~right_img_input")
	orientation_projection = right_img_input.split("/")[3]
	camera_name = right_img_input.split("/")[1]
	lens_name = right_img_input.split("/")[2]
	rightImg_projection_frame_name = "{}_{}_frame".format(orientation_projection,lens_name)
	rightImg_lens_link_frame_name = "{}_{}_link".format(camera_name,lens_name)
	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)
	output_config_file = rospy.get_param("~output_config_file", \
										"/camera360/left/right_img_input")
	camera_info_file = rospy.get_param("~camera_info_file", \
										None)
	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
	aruco_params = cv2.aruco.DetectorParameters_create()
	r_array = []
	t_array = []
#           # https://docs.opencv.org/master/d9/d6a/group__aruco.html#gafce26321f39d331bc12032a72b90eda6
	MARKER_LENGTH= 0.5

	# aruco_params.adaptiveThreshWinSizeMin=8
	# aruco_params.adaptiveThreshWinSizeMax=30
	# aruco_params.adaptiveThreshWinSizeStep=2
	cam = createInsta360Stream(left_img_input, right_img_input,camera_info_file)
	logger.debug("{}, {}".format(cam.get_distortion_coeff(),cam.get_intrinsic_matrix()))

	cam.distortion_coeffs = np.array(cam.get_distortion_coeff())
	while(not rospy.is_shutdown()):
		t0 = time.time()
		# data_pack = cam.get_stitch_image_with_matching_features()
		imgLeft =cam.imgLeft
		imgRight = cam.imgRight

		if(imgLeft is None or imgRight is None):
			continue
		# imgLeft = clahe(imgLeft)
		# imgRight = clahe(imgRight)
		imgLeft = cv2.cvtColor(imgLeft,cv2.COLOR_BGR2GRAY)
		imgRight = cv2.cvtColor(imgRight,cv2.COLOR_BGR2GRAY)
		# imgLeft = cv2.fastNlMeansDenoising(imgLeft)
		# imgRight = cv2.fastNlMeansDenoising(imgRight)
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
		# imgLeft = clahe.apply(imgLeft)		
		# imgRight = clahe.apply(imgRight)
		(corners, ids,
		rejected) = cv2.aruco.detectMarkers((imgLeft),
											aruco_dict,
											parameters=aruco_params)
		criteria = 	(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.05)
		if len(corners) > 0:
			logger.debug("hehe {},{}".format(type(corners[0]),corners))
			for point in corners[0][0]:
				# print(point[0])
				point = point.astype(np.int32)
				cv2.circle(imgLeft, (point[0], point[1]), 1, (0, 255, 0), -1)	
			corners = cv2.cornerSubPix(imgLeft,corners[0],(7,7),(1,1),criteria)
		if len(rejected) > 0:
			for point in rejected[0][0]:
				# print(point[0])
				point = point.astype(np.int32)
				cv2.circle(imgLeft, (point[0], point[1]), 1, (0, 0, 255), -1)	
		result = cv2.aruco.estimatePoseSingleMarkers(
			corners,
			markerLength=MARKER_LENGTH,
			distCoeffs=cam.distortion_coeffs,
			cameraMatrix=cam.K)
		
		# rvec is the orientation of the aruco marker relative to the camera.
		# Sometimes, there will be 2 possible rotation vectors for 1 marker.
		rvec_frontcam = result[0]
		tvec_frontcam = result[1]
		# print(tvec_frontcam.shape)
		# tvec_frontcam = np.array(result,dtype=object)[1,0]
		# print("res front cam",result)
		if(rvec_frontcam is not None):
			for rot in rvec_frontcam:
				rot_mat_fid_2_front_cam = cv2.Rodrigues(rot)[0]
				logger.debug(rot_mat_fid_2_front_cam)
				logger.debug("tvec {} {}".format(tvec_frontcam,tvec_frontcam.shape))
				mat_trans_fid_2_pfr =create_4d_mat_from_R_and_t(rot_mat_fid_2_front_cam,tvec_frontcam[0])
				cv2.aruco.drawAxis(imgLeft, cam.K, cam.distortion_coeffs, rot,
								np.array(tvec_frontcam)[0][0], 0.1)
		
		
		
		(corners, ids,
		rejected) = cv2.aruco.detectMarkers((imgRight),
											aruco_dict,
											parameters=aruco_params)
		if len(corners) > 0:
			logger.debug("hehe {},{}".format(type(corners[0]),corners))
			for point in corners[0][0]:
				# print(point[0])
				point = point.astype(np.int32)
				cv2.circle(imgRight, (point[0], point[1]), 1, (0, 255, 0), -1)
			corners = cv2.cornerSubPix(imgRight,corners[0],(7,7),(1,1),criteria)
		result = cv2.aruco.estimatePoseSingleMarkers(
			corners,
			markerLength=MARKER_LENGTH,
			distCoeffs=cam.distortion_coeffs,
			cameraMatrix=cam.K)
		# rvec is the orientation of the aruco marker relative to the camera.
		# Sometimes, there will be 2 possible rotation vectors for 1 marker.
		rvec_rearcam = result[0]
		tvec_rearcam = result[1]

		
		# logger.debug("res rear cam",result)
		if(rvec_rearcam is not None):
			for rot in rvec_rearcam:
				rot_mat_fid_2_rear_cam = cv2.Rodrigues(rot)[0]
				logger.debug(rot_mat_fid_2_rear_cam)
				logger.debug("tvec {} {} ".format(tvec_rearcam,tvec_rearcam.shape))
				# mat_trans_fid_2_rear_cam =create_4d_mat_from_R_and_t(rot_mat_fid_2_rear_cam,np.array(tvec_rearcam)[0])
				mat_trans_fid_2_prl =create_4d_mat_from_R_and_t(rot_mat_fid_2_rear_cam,tvec_rearcam[0])
				cv2.aruco.drawAxis(imgRight, cam.K, cam.distortion_coeffs, rot,
								np.array(tvec_rearcam)[0][0], 0.1)
				normal_in_aruco_frame = np.array([0,0,1])
				normal_in_prl_frame = np.dot(rot_mat_fid_2_rear_cam,normal_in_aruco_frame)
				origin_in_prl_frame = tvec_rearcam[0]
				distance_to_aruco_plane = np.dot(origin_in_prl_frame,normal_in_prl_frame)



		logger.debug("rear and front orientation {}, {}".format(rvec_rearcam,rvec_frontcam))
		cv2.imshow("left",imgLeft)
		cv2.imshow("right",imgRight)
		key =cv2.waitKey(1)
		if(key== ord('q')):
			rospy.signal_shutdown("command")


		if(rvec_frontcam is None or rvec_rearcam is None):
			continue

		r_optical_frame_to_normal_frame = R.from_quat(np.array([0.5,-0.4999999999999999,0.5,-0.5000000000000001])).as_matrix()

		trans_optical_frame_to_frame	=	create_4d_mat_from_R_and_t(r_optical_frame_to_normal_frame,np.array([0,0,0]))


		mat_trans_pfr_2_prl = np.matmul(mat_trans_fid_2_prl,np.linalg.inv(mat_trans_fid_2_pfr))
		logger.debug("mat trans {}".format(mat_trans_pfr_2_prl))

		# mat_trans_prl_2_pfr = np.linalg.inv(mat_trans_pfr_2_prl)
		r = R.from_matrix(mat_trans_pfr_2_prl[:3,:3])
		t = np.array(mat_trans_pfr_2_prl[0:3,3])
		# print(t.shape,normal_in_prl_frame.shape)
		# # x = np.matmul(np.transpose(t),np.transpose(np.array(normal_in_prl_frame)))
		# # x = np.matmul(np.transpose(t),np.array(normal_in_prl_frame))
		# x = np.outer(t,np.array(normal_in_prl_frame))
		# logger.debug(x)

		
		# homo_prl_to_pfr = mat_trans_prl_2_pfr[:3,:3] - x* 1/distance_to_aruco_plane
		# logger.debug(homo_prl_to_pfr)
		# homography = cam.get_intrinsic_matrix() * homo_prl_to_pfr * np.linalg.inv(cam.get_intrinsic_matrix())
		# # homography =  homo_prl_to_pfr 
		# homography = homography / homography[2,2]
		# # homography = homography / homography[2,2]
		# # a = Stitcher(np.linalg.inv(homo_prl_to_pfr))
		# a = Stitcher(homography)
		# # a = Stitcher((homography))
		# cam.stitcher = a
		# cv2.imshow("stitched_img",cam.get_bgr_frame())
		logger.info("front2rear r and t in degree and m {}, {}".format(r.as_euler("xyz",degrees=True),t))


		# r = R.from_euler("xyz",np.array([0.2617994,0, -1.570796327]))

		trans_rightImg = tfBuffer.lookup_transform(rightImg_lens_link_frame_name,rightImg_projection_frame_name,rospy.Time())
		trans_leftImg = tfBuffer.lookup_transform(leftImg_lens_link_frame_name,leftImg_projection_frame_name,rospy.Time())
		rotation_rightImg = trans_rightImg.transform.rotation
		rotation_rightImg = R.from_quat(np.array([rotation_rightImg.x,rotation_rightImg.y,rotation_rightImg.z,rotation_rightImg.w]))
		rotation_leftImg = trans_leftImg.transform.rotation
		rotation_leftImg = R.from_quat(np.array([rotation_leftImg.x,rotation_leftImg.y,rotation_leftImg.z,rotation_leftImg.w]))
		# r = R.from_quat(np.array([0.09229595566153638,0.09229595564260615,-0.7010573847204488,0.7010573845766596]))
		Trans_from_frontRightProj2frontCamLink = create_4d_mat_from_R_and_t(rotation_leftImg.as_matrix(),np.array([0,0,0]))


		# find transformation from pinhole rear left to optical frame rear
		# r = R.from_euler("xyz",np.array([-0.2617994, 0, 1.570796327]))
		# r = R.from_quat(np.array([-0.09229595566153638,0.09229595564260615,0.7010573847204488,0.7010573845766596]))
		Trans_from_rearLeftProj2rearCamLink = create_4d_mat_from_R_and_t(rotation_rightImg.as_matrix(),np.array([0,0,0]))
		


		trans_from_frontRightProj_to_rearLeftProj = np.matmul(trans_optical_frame_to_frame,np.matmul(mat_trans_pfr_2_prl,np.linalg.inv(trans_optical_frame_to_frame)))
		trans_from_frontCamLink2rearCamLink = np.matmul(Trans_from_rearLeftProj2rearCamLink,np.matmul(trans_from_frontRightProj_to_rearLeftProj,np.linalg.inv(Trans_from_frontRightProj2frontCamLink)))
		trans_from_frontCamOptical2rearCamOptical = np.matmul(np.linalg.inv(trans_optical_frame_to_frame),np.matmul(trans_from_frontCamLink2rearCamLink,(trans_optical_frame_to_frame)))
		# a = np.matmul(r)
		# logger.info(trans_from_frontRightProj_to_rearLeftProj)
		r = R.from_matrix(trans_from_frontRightProj_to_rearLeftProj[:3,:3])
		t = trans_from_frontRightProj_to_rearLeftProj[0:3,3]
		logger.info("from left img projection to right img projection: {},{}".format(r.as_euler("xyz"),t))

		
		trans_from_rearCamOptical2frontCamOptical =np.linalg.inv(trans_from_frontCamOptical2rearCamOptical)
		# trans_from_rearCamOptical2frontCamOptical =(trans_from_frontCamOptical2rearCamOptical)
		r = R.from_matrix(trans_from_rearCamOptical2frontCamOptical[:3,:3])
		t = trans_from_rearCamOptical2frontCamOptical[0:3,3]
		logger.info("rearCamOptical2frontCamOptical r:{},t:{}".format(r.as_euler("xyz"),t))		
		r_array.append(r.as_euler("xyz"))
		t_array.append(t)
	# r_array = reject_outliers(np.asarray(r_array))
	# t_array = reject_outliers(np.asarray(t_array))	
	r_array = np.asarray(r_array)
	t_array = np.asarray(t_array)
	r_median = np.median((r_array),axis=0)
	t_median = np.median((t_array),axis=0)
	
	# r_median = np.mean(np.asarray(r_array),axis=0)
	# t_median = np.mean(np.asarray(t_array),axis=0)
	logger.info("<origin xyz=\"{} {} {}\" rpy=\"{} {} {}\"/>".format(*t_median,*r_median))		

	# H = cam.stitcher.calcHomographyMatrix(5.0)

	
	#=========================find crop_area code here=========================================================
	# shape_left_img = (cam.imgLeft.shape[0],int(cam.imgRight.shape[1]*0.58),3)

	# print(H)
	cv2.destroyAllWindows()
