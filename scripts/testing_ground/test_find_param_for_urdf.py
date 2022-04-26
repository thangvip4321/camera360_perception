import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import array
a =  ((array([[ 0.9569086 , -0.01468521,  0.29001771],
       [ 0.01202199,  0.99986764,  0.01096254],
       [-0.29014031, -0.00700356,  0.95695849]]), array([[ 0.99983041,  0.00395102,  0.01798718],
       [-0.00479699,  0.99887232,  0.04723424],
       [-0.01778028, -0.04731252,  0.99872188]]), array([[ 0.99983041,  0.00395102,  0.01798718],
       [-0.00479699,  0.99887232,  0.04723424],
       [-0.01778028, -0.04731252,  0.99872188]])), (array([[ 0.05047321],
       [ 0.00429495],
       [-0.28491306]]), array([[-0.05047321],
       [-0.00429495],
       [ 0.28491306]]), array([[ 0.28530321],
       [-0.03862281],
       [-0.02918562]]), array([[-0.28530321],
       [ 0.03862281],
       [ 0.02918562]])), (array([[-0.98927966],
       [ 0.13727456],
       [-0.04981423]]), array([[ 0.98927966],
       [-0.13727456],
       [ 0.04981423]]), array([[-0.325457  ],
       [-0.04103545],
       [ 0.94466599]]), array([[ 0.325457  ],
       [ 0.04103545],
       [-0.94466599]])))
R_from_prl_to_pfr = np.array([[ 0.9569086 , -0.01468521,  0.29001771],
                                                [ 0.01202199,  0.99986764,  0.01096254],
                                                [-0.29014031, -0.00700356,  0.95695849]])    
t_from_prl_to_pfr = np.array([-0.05047321,
                                                 -0.00429495,
                                                  0.28491306])
# this is K
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
r = R.from_euler("xyz",np.array([0, 0.26179938783, -1.570796327]))
Trans_from_pfr_to_off = create_4d_mat_from_R_and_t(r.as_matrix(),np.array([0,0,0]))


# find transformation from pinhole rear left to optical frame rear
r = R.from_euler("xyz",np.array([0, 0.26179938783, 1.570796327]))
Trans_from_prl_to_ofr = create_4d_mat_from_R_and_t(r.as_matrix(),np.array([0,0,0]))


trans_from_ofr_to_off = np.matmul(Trans_from_pfr_to_off,np.matmul(trans_from_prl_to_pfr,np.linalg.inv(Trans_from_prl_to_ofr)))
print(trans_from_ofr_to_off)
r = R.from_matrix(trans_from_ofr_to_off[:3,:3])
t = trans_from_ofr_to_off[0:3,3]
print(r.as_euler("xyz"),t)






