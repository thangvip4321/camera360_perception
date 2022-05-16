import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import array
import cv2
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(process)s-%(threadName)s-%(funcName)s_%(lineno)d: %(message)s')
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
# a =  ((array([[ 0.9569086 , -0.01468521,  0.29001771],
#        [ 0.01202199,  0.99986764,  0.01096254],
#        [-0.29014031, -0.00700356,  0.95695849]]), array([[ 0.99983041,  0.00395102,  0.01798718],
#        [-0.00479699,  0.99887232,  0.04723424],
#        [-0.01778028, -0.04731252,  0.99872188]]), array([[ 0.99983041,  0.00395102,  0.01798718],
#        [-0.00479699,  0.99887232,  0.04723424],
#        [-0.01778028, -0.04731252,  0.99872188]])), (array([[ 0.05047321],
#        [ 0.00429495],
#        [-0.28491306]]), array([[-0.05047321],
#        [-0.00429495],
#        [ 0.28491306]]), array([[ 0.28530321],
#        [-0.03862281],
#        [-0.02918562]]), array([[-0.28530321],
#        [ 0.03862281],
#        [ 0.02918562]])), (array([[-0.98927966],
#        [ 0.13727456],
#        [-0.04981423]]), array([[ 0.98927966],
#        [-0.13727456],
#        [ 0.04981423]]), array([[-0.325457  ],
#        [-0.04103545],
#        [ 0.94466599]]), array([[ 0.325457  ],
#        [ 0.04103545],
#        [-0.94466599]])))

intrinsics = np.array([ 
  [429.6190111627674,0,512.5],
  [0                 ,429.6190111627674 ,342.0],
  [  0                ,0                 ,1  ]
])
# homography_mat = np.array([[0.7799533461922264,0.03252953257609461,315.75709983418005]
# ,[-0.18619678412147578,0.9604013233667958,18.733284733531068]
# ,[-0.0003305184012708828,4.738582881079115e-05,1.0]
# ])
# homography_mat = np.array([[0.9016356131706094, -0.007938292128232001, 346.0047373204196],
#                             [-0.00715139367737085, 1.001756253357512, -5.077363732447504],
#                             [-8.765185506167562e-05, 3.1676416067843366e-05, 1.0]])
homography_mat=np.array([[1.051353959655354,0.01911344123241793,302.8337119241751],
[0.02868471688109597,1.0465875830118332,-29.705713607082643],
[5.144532741729318e-05,4.887857566466745e-05,1.0]])
a = cv2.decomposeHomographyMat(homography_mat,intrinsics)
print(a)
# R_from_prl_to_pfr = np.array([[ 0.9569086 , -0.01468521,  0.29001771],
#                                                 [ 0.01202199,  0.99986764,  0.01096254],
#                                                 [-0.29014031, -0.00700356,  0.95695849]])    
# t_from_prl_to_pfr = np.array([-0.05047321,
#                                                  -0.00429495,
#                                                   0.28491306])
R_from_prl_to_pfr = a[1][0] 
t_from_prl_to_pfr = a[2][0][:,0]
normal_to_a_calculated_plane_in_pfr_frame = a[3][0]
# print("hwehe",t_from_prl_to_pfr)

def create_4d_mat_from_R_and_t(R,t):
    Mat = np.identity(4)
    Mat[:3,:3]= R
    # print(Mat)
    # print(Mat[0:3])
    Mat[0:3,3] = t
    return Mat
trans_from_prl_to_pfr = np.identity(4)
trans_from_prl_to_pfr[:3,:3]= R_from_prl_to_pfr
trans_from_prl_to_pfr[0:3,3] = t_from_prl_to_pfr

# print(trans_from_prl_to_pfr)                                              


# find transformation from pinhole front right to optical frame front
# [0, 0, 0, 0, 0.26179938783, -1.570796327]
# r = R.from_euler("xyz",np.array([0, 0.26179938783, -1.570796327]))
# Trans_from_pfr_to_off = create_4d_mat_from_R_and_t(r.as_matrix(),np.array([0,0,0]))


# # find transformation from pinhole rear left to optical frame rear
# r = R.from_euler("xyz",np.array([0, 0.26179938783, 1.570796327]))
# Trans_from_prl_to_ofr = create_4d_mat_from_R_and_t(r.as_matrix(),np.array([0,0,0]))


# trans_from_ofr_to_off = np.matmul(Trans_from_pfr_to_off,np.matmul(trans_from_prl_to_pfr,np.linalg.inv(Trans_from_prl_to_ofr)))
# print(trans_from_ofr_to_off)
r_from_prl_to_pfr = R.from_matrix(trans_from_prl_to_pfr[:3,:3])
t = trans_from_prl_to_pfr[0:3,3]
print(r_from_prl_to_pfr.as_euler("xyz"),t)


r_optical_frame_to_normal_frame = R.from_quat(np.array([0.5,-0.4999999999999999,0.5,-0.5000000000000001])).as_matrix()
trans_optical_frame_to_frame	=	create_4d_mat_from_R_and_t(r_optical_frame_to_normal_frame,np.array([0,0,0]))

# trans_rightImg = tfBuffer.lookup_transform(rightImg_lens_link_frame_name,rightImg_projection_frame_name,rospy.Time())
# 		trans_leftImg = tfBuffer.lookup_transform(leftImg_lens_link_frame_name,leftImg_projection_frame_name,rospy.Time())
# 		rotation_rightImg = trans_rightImg.transform.rotation
# 		rotation_rightImg = R.from_quat(np.array([rotation_rightImg.x,rotation_rightImg.y,rotation_rightImg.z,rotation_rightImg.w]))
# 		rotation_leftImg = trans_leftImg.transform.rotation
# 		rotation_leftImg = R.from_quat(np.array([rotation_leftImg.x,rotation_leftImg.y,rotation_leftImg.z,rotation_leftImg.w]))
r_from_frontRightProj2frontCamLink = R.from_quat(np.array([0.09229595566153638,0.09229595564260615,-0.7010573847204488,0.7010573845766596]))
Trans_from_frontRightProj2frontCamLink = create_4d_mat_from_R_and_t(r_from_frontRightProj2frontCamLink.as_matrix(),np.array([0,0,0]))
# find transformation from pinhole rear left to optical frame rear
# r = R.from_euler("xyz",np.array([-0.2617994, 0, 1.570796327]))

r_from_rearLeftProj2rearCamLink = R.from_quat(np.array([-0.09229595566153638,0.09229595564260615,0.7010573847204488,0.7010573845766596]))
Trans_from_rearLeftProj2rearCamLink = create_4d_mat_from_R_and_t(r_from_rearLeftProj2rearCamLink.as_matrix(),np.array([0,0,0]))
		
rot_from_frontRightProj_to_rearLeftProj = np.matmul(r_optical_frame_to_normal_frame,np.matmul(r_from_prl_to_pfr.as_matrix(),np.linalg.inv(r_optical_frame_to_normal_frame)))
rot_from_frontCamLink2rearCamLink = np.matmul(r_from_rearLeftProj2rearCamLink.as_matrix(),np.matmul(rot_from_frontRightProj_to_rearLeftProj,np.linalg.inv(r_from_frontRightProj2frontCamLink.as_matrix())))
rot_from_frontCamOptical2rearCamOptical = np.matmul(np.linalg.inv(r_optical_frame_to_normal_frame),np.matmul(rot_from_frontCamLink2rearCamLink,(r_optical_frame_to_normal_frame)))

print("rot_from hehe",R.from_matrix(np.linalg.inv(rot_from_frontCamOptical2rearCamOptical)).as_euler("xyz"))

trans_from_frontRightProj_to_rearLeftProj = np.matmul(trans_optical_frame_to_frame,np.matmul(trans_from_prl_to_pfr,np.linalg.inv(trans_optical_frame_to_frame)))
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

