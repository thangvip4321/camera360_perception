# import the necessary packages
from math import degrees
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R

class Stitcher:
	def __init__(self,precalculated_homo_matrix: np.ndarray=None,crop_area=None):
		self.rotations_from_fcam_to_rcam = []
		self.map_x = None
		self.map_y = None
		self._expected_rotation_from_frontview_to_rearview = np.array([0,0,0])
		self.list_ptsA = []
		self.list_ptsB = []
		self.matches_with_distances = []
		self.stitch_shape = (0,0)
		if precalculated_homo_matrix is None:
			self.isRunTime=False
		else:
			self.homo = precalculated_homo_matrix
			self.crop_area = crop_area
			self.isRunTime=True


		# self.homo = np.array([[ 1.09565584e+00, -1.48540125e-02 , 4.22960298e+02],
 		# 					  [ 2.73629648e-01,  9.80374830e-01 ,-1.32677131e+01],
 		# 					  [ 4.35535822e-04, -1.00907330e-04 , 1.00000000e+00]])
	def stitch(self, images, ratio=0.75, reprojThresh=4.0):
		(imageB, imageA) = images

		if(not self.isRunTime):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
			(kpsA, featuresA) = self.detectAndDescribe(imageA)
			(kpsB, featuresB) = self.detectAndDescribe(imageB)

			# match features between the two images
			M = self.matchKeypoints(kpsA, kpsB,
				featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
			if M is None:
				print("not enough matching features")
				return None
			(matches, H, status,ptsA,ptsB) = M
			if(H is None):
				print("not enough inlier")
				return None
			# print("M is",M)
			self.homo = H
			# print(np.linalg.det(H))
			# print(H)
			cv2.imshow("matches",self.drawMatches(imageA,imageB,kpsA,kpsB,matches,status))
			# print("r euler is:",e)
			# print(np.linalg.inv(r.as_matrix()))

			# self.rotation_from_fcam_to_rcam.append(r.as_euler("xyz",degrees=True))
			# print(self.homo)

		result = cv2.warpPerspective(imageA, self.homo,
				(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		if(self.isRunTime):
			if(self.crop_area is not None):
				left_x_crop,right_x_crop,upper_y_crop,lower_y_crop = self.crop_area
				result = result[upper_y_crop:lower_y_crop,left_x_crop:right_x_crop]
				# cv2.imshow("res",result)
				# cv2.waitKey(1)
		if not self.isRunTime:
			return result,ptsA,ptsB
		else:
			return result,None,None



	def calcHomographyMatrix(self,reprojThresh=4.0):
		# array_matches = np.asarray(self.matches_with_distances)
		# print(array_matches.shape)
		# distance = array_matches[:,2]
		# threshold = np.quantile(distance,0.5)
		# print(threshold,np.min(array_matches[:,2]))
		# array_matches = array_matches[np.where(distance < threshold)]
		# print(array_matches.shape)
		# ptsA = np.stack(array_matches[:,0])
		# print("after filter",ptsA.shape)
		# print(ptsA)
		# ptsB = np.stack(array_matches[:,1])

		
		ptsA = np.concatenate(self.list_ptsA)
		print(ptsA.shape)
		ptsB = np.concatenate(self.list_ptsB)

		(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
		return H
	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
			# detect and extract features from the image
		descriptor = cv2.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)


		

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
		matches_with_distances = []

		# print(dir(rawMatches[0][0]))
		# loop over the raw matches
		for m in rawMatches:
			# for i in m:
			# 	print(m[0].distance,m[1].distance)
			# 	print(m[0].queryIdx,m[1].queryIdx)
			# 	print(m[0].trainIdx,m[1].trainIdx)
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			# print(m[0].distance,m[1].distance)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
				matches_with_distances.append((m[0].trainIdx, m[0].queryIdx,m[0].distance))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			distance = np.float32([matches_with_distances[i][2] for i in range(len(matches_with_distances))])
			a = [(ptsA[i],ptsB[i],distance[i]) for i in range(len(matches))]
			self.matches_with_distances.extend(a)
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status,ptsA,ptsB)

		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
			if s == 0:
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (255, 0, 0), 1)
		# return the visualization
		return vis
	def calculate_mean_homography_mat(self):
		self.rotations_from_fcam_to_rcam = np.asarray(self.rotations_from_fcam_to_rcam)
		print(self.rotations_from_fcam_to_rcam.shape)
		average_rotation = np.mean(self.rotations_from_fcam_to_rcam,axis=0)
		print(average_rotation)
	@classmethod
	def createFromYaml(cls,stitch_file):
		with open(stitch_file,"r") as f:
			obj = yaml.safe_load(f)
			homography_matrix = np.reshape(np.array(obj["homography_matrix"]),(3,3))
			crop_x = obj["crop_area_x"]
			crop_y = obj["crop_area_y"]
			stitcher = Stitcher(homography_matrix,(*crop_x,*crop_y))
			return stitcher

class ArucoStitcher:
	def __init__(self,marker_length):
		self.marker_length = marker_length
		
	def stitch(self, images, ratio=0.75, reprojThresh=4.0):
		(imageB, imageA) = images

		if(not self.isRunTime):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
			(kpsA, featuresA) = self.detectAndDescribe(imageA)
			(kpsB, featuresB) = self.detectAndDescribe(imageB)

			# match features between the two images
			M = self.matchKeypoints(kpsA, kpsB,
				featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
			if M is None:
				print("not enough matching features")
				return None
			(matches, H, status,ptsA,ptsB) = M
			if(H is None):
				print("not enough inlier")
				return None
			# print("M is",M)
			self.homo = H
			# print(np.linalg.det(H))
			# print(H)
			
			# print("r euler is:",e)
			# print(np.linalg.inv(r.as_matrix()))

			# self.rotation_from_fcam_to_rcam.append(r.as_euler("xyz",degrees=True))
			# print(self.homo)

		result = cv2.warpPerspective(imageA, self.homo,
				(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		if(self.isRunTime):
			if(self.crop_area is not None):
				left_x_crop,right_x_crop,upper_y_crop,lower_y_crop = self.crop_area
				result = result[upper_y_crop:lower_y_crop,left_x_crop:right_x_crop]
				# cv2.imshow("res",result)
				# cv2.waitKey(1)
		if not self.isRunTime:
			return result,ptsA,ptsB
		else:
			return result,None,None



	def calcHomographyMatrix(self,reprojThresh=4.0):
		ptsA = np.concatenate(self.list_ptsA)
		print(ptsA.shape)
		ptsB = np.concatenate(self.list_ptsB)
		(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
		return H
	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
			# detect and extract features from the image
		descriptor = cv2.SIFT_create()
		(kps, features) = descriptor.detectAndCompute(image, None)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)


		

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
		# print(rawMatches)
		# loop over the raw matches
		for m in rawMatches:
			# for i in m:
			# 	print(m[0].distance,m[1].distance)
			# 	print(m[0].queryIdx,m[1].queryIdx)
			# 	print(m[0].trainIdx,m[1].trainIdx)
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status,ptsA,ptsB)

		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis
	def calculate_mean_homography_mat(self):
		self.rotations_from_fcam_to_rcam = np.asarray(self.rotations_from_fcam_to_rcam)
		print(self.rotations_from_fcam_to_rcam.shape)
		average_rotation = np.mean(self.rotations_from_fcam_to_rcam,axis=0)
		print(average_rotation)
	@classmethod
	def createFromYaml(cls,stitch_file):
		with open(stitch_file,"r") as f:
			obj = yaml.safe_load(f)
			homography_matrix = np.reshape(np.array(obj["homography_matrix"]),(3,3))
			crop_x = obj["crop_area_x"]
			crop_y = obj["crop_area_y"]
			stitcher = Stitcher(homography_matrix,(*crop_x,*crop_y))
			return stitcher	