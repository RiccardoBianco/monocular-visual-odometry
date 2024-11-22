import cv2
import numpy

def ext_pose(landmarks1, landmarks2, K, threshold, desiredconfidence):
'Assumiamo di avere le corrispondenze (landmarks1, landmarks2) [N,2]'
E, mask = cv2.findEssentialMat(landmarks1, landmarks2, K, method=cv2.RANSAC, threshold=1.0, prob=0.95)

"""Threshold è quanti pixel di distanza considera a partire dalla epipolar
line prima di considerarlo outlier. Default 1 """

R, T, good, mask, triangulatedPoints = cv2.recoverPose(E, landmarks1, landmarks2, K)

"""tra gli output c'è anche good ossia il numero di inliers che passano 
il cheirality check, mask ossia la mask degli inlier che passano il cheirality check
e infine triangulatedPoints ossia i punti 3d ricostruiti a partire dalla triangolazione"