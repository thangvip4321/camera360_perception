print(__name__)
import logging
import threading
import cv2
import numpy as np
import yaml
import rospy
from sensor_msgs.msg import CompressedImage
import cv_bridge
import traceback
from camera360_perception.stitcher import Stitcher



class Insta360Camera():
    # Generate the ROS camera object, but this Object assume that the program calling this object has already
    # initialized a node
    def __init__(self,topic_name,camera_info_file= None):
        self.logger = logging.getLogger("__main__")
        self.ev = threading.Event()
        self.img= None
        self.logger.info("%s",topic_name)
        self.sub = rospy.Subscriber(topic_name, CompressedImage, self.getImage,queue_size=1)
        self.bridge = cv_bridge.core.CvBridge()
        
        if(camera_info_file is not None):
            with open(camera_info_file,"r") as f: 
                obj = yaml.safe_load(f)
                self.K =  np.reshape(np.array(obj["intrinsic_matrix"]),(3,3))
                self.distortion_coeffs = obj["distortion"]
        else:
            self.K = None
            self.distortion_coeffs = None
    def get_bgr_frame(self):
        self.ev.wait()
        self.ev.clear()
        return self.img
    def get_intrinsic_matrix(self):
        return self.K
    def getImage(self,data):
        np_arr = np.fromstring(data.data, np.uint8)
        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # self.logger.info("%f",time.time())
        self.ev.set()
    def get_distortion_coeff(self):
        return self.distortion_coeffs
    def stop(self):
        self.sub.unregister()


class Insta360StitchCamera():
    # Generate the ROS camera object, but this Object assume that the program calling this object has already
    # initialized a node
    left_area_crop_to = 0.55
    right_area_crop_from = 0.4

    def __init__(self,left_img_topic,right_img_topic,camera_info_file:str,stitcher: Stitcher=Stitcher() ):
        self.imgLeft= None
        self.imgRight = None
        self.left_new_image = False
        self.right_new_image = False
        self.stitcher = stitcher
        self.ev = threading.Event()

        if(camera_info_file is None):
            self.K = None
            self.distortion_coeffs = None
        else:
            with open(camera_info_file,"r") as f: 
                obj = yaml.safe_load(f)
                print(obj)
                self.K =  np.reshape(np.array(obj["intrinsic_matrix"]),(3,3))
                self.distortion_coeffs = obj["distortion"]
                crop_ratio = obj["crop_ratio"]
                self.left_area_crop_to = crop_ratio["left"]
                self.right_area_crop_from = crop_ratio["right"]
        self.sub_left = rospy.Subscriber(left_img_topic, CompressedImage, self.__getImageLeft, queue_size=1)
        self.sub_right = rospy.Subscriber(right_img_topic, CompressedImage, self.__getImageRight, queue_size=1)
        self.bridge = cv_bridge.core.CvBridge()



    # stitch the left image and the right image and the return the whole frame
    def get_bgr_frame(self):
        self.ev.wait()
        self.ev.clear()
        self.left_new_image = False
        self.right_new_image = False
        h, w, _ = self.imgLeft.shape
        # print(self.imgLeft.shape)
        leftImgCropped = self.imgLeft[:, 0:int(w*self.left_area_crop_to)]
        rightImgCropped  = self.imgRight[:, int(w*self.right_area_crop_from):w]
        # rightImgCropped  = self.imgRight

        # cv2.imshow("leftImgCropped",leftImgCropped)
        # cv2.imshow("rightImgCropped",rightImgCropped)
        stitched = None
        try:
            stitched = self.stitcher.stitch([leftImgCropped, rightImgCropped])
        except TypeError as e:
            traceback.print_exc()
        return stitched[0]
    def get_stitch_image_with_matching_features(self):
            self.ev.wait()
            self.ev.clear()
        # if (self.left_new_image == True) & (self.right_new_image == True):
            self.left_new_image = False
            self.right_new_image = False
            h, w, _ = self.imgLeft.shape
            leftImgCropped = self.imgLeft[:, 0:int(w*self.left_area_crop_to)]
            rightImgCropped  = self.imgRight[:, int(w*self.right_area_crop_from):w]
            leftImgCropped = cv2.fastNlMeansDenoisingColored(leftImgCropped)
            rightImgCropped = cv2.fastNlMeansDenoisingColored(rightImgCropped)
            cv2.imshow("left",leftImgCropped)
            cv2.imshow("right",rightImgCropped)
            stitched = None
            
            try:
                stitched = self.stitcher.stitch([leftImgCropped, rightImgCropped])
            except TypeError as e:
                traceback.print_exc()
            if(stitched is not None):
                return stitched
            return stitched
        # return None
    def get_intrinsic_matrix(self):
        if self.K is None:
            raise NotImplementedError("should have initialized with camera_file_info")
        return self.K
        
    def __getImageLeft(self,data):
        np_arr = np.fromstring(data.data, np.uint8)
        self.imgLeft = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # print(self.imgLeft is None)
        self.left_new_image = True
        if(self.right_new_image):
            self.ev.set()
        # the callback still run normally, but the ev variable cannot be set
        # print("is called stithced left")

     
    def __getImageRight(self,data):
        np_arr = np.fromstring(data.data, np.uint8)
        self.imgRight = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.right_new_image = True
        if(self.left_new_image):
            self.ev.set()

    def get_distortion_coeff(self):
        if self.distortion_coeffs is None:
            raise NotImplementedError("should have initialized with camera_file_info")
        return self.distortion_coeffs
    def stop(self):
        self.sub_left.unregister()
        self.sub_right.unregister()

def createInsta360Stream(topic_name1: str,topic_name2:str ="", camera_file:str =None,stitcher=None):
    logger = logging.getLogger("__main__")
    if(camera_file is None):
        logger.warning("configuring camera with no intrinsic and distortion parameters!")
    if(stitcher is None):
        logger.warning("creating default stitcher which will be very slow")
        stitcher = Stitcher()
    if topic_name2 != "":
        return Insta360StitchCamera(topic_name1,topic_name2,camera_file,stitcher)
    else:
        return Insta360Camera(topic_name1,camera_file)