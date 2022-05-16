#!/home/tranquockhue/anaconda3/envs/cvdl/bin/python
import time
import numpy as np
import cv2 as cv
import glob

import yaml
import rospy 
from cameraType.ros_camera import ROSCamera 

def keyboardInterruptHandler(signal, frame):
	print(signal,frame)
	exit(0)
    #print("call your function here".format(signal))exit(0)

if __name__ == "__main__":
# termination criteria
    internal_points_width = 8   
    internal_points_height=5
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((internal_points_width*internal_points_height,3), np.float32)
    objp[:,:2] = np.mgrid[0:internal_points_width,0:internal_points_height].T.reshape(-1,2)
    # print(objp)

    rospy.init_node("calibration_intrinsic",anonymous=True,disable_signals=True)
    camera_input_topic = rospy.get_param("~input", \
                                    "/camera360_front/right/image_rect")
    output_config_file = rospy.get_param("~output_calibration")
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    cam = ROSCamera(camera_input_topic)
    # img = cv.imread("/home/tranquockhue/image_projection_ws/src/camera360_perception/left04.jpg",cv.IMREAD_GRAYSCALE)
    try:
        while not rospy.is_shutdown():
            img = cam.get_bgr_frame()
            t0 = time.time()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # gray=np.copy(img)
            cv.imshow('img_gray', gray)

            # Find the chess board corners
            ret, corners = cv.findChessboardCornersSB(gray, (internal_points_width,internal_points_height))
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                cv.drawChessboardCorners(gray, (internal_points_width,internal_points_height), corners2, ret)
                cv.imshow('img', gray)
            print("fps:",1/(time.time()-t0))
            key =cv.waitKey(0)
            if(key==ord('q')):
                break
    finally:
        print("calculating ")      
        cv.destroyAllWindows()
        print(len(objpoints))
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("done! ")      
        # print(dist,mtx,rvecs, tvecs)
        # img = cv.imread('left09.jpg')
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
        print(newcameramtx)
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # cv.imwrite('calibresult.png', dst)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)))
        data = dict(
            intrinsic_matrix = newcameramtx.tolist(),
            distortion= dist.tolist(),
            resolution= [w,h]
        )
        with open(output_config_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)