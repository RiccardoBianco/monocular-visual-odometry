import cv2
import os
import numpy as np


# Step 1: Load Images
# frame_1_relative_folder = "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/img_CAMERA1_1261229981.580023_left.jpg"
# frame_2_relative_folder = "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/img_CAMERA1_1261229981.680019_left.jpg"

# frame_1_relative_folder = "/datasets/kitti/05/image_0/000000.png"
# frame_2_relative_folder = "datasets/kitti/05/image_0/000002.png"

frame_1_relative_folder = "/datasets/parking/images/img_00000.png"
frame_2_relative_folder = "/datasets/parking/images/img_00002.png"

frame1_folder = os.path.join(os.path.dirname(__file__) + frame_1_relative_folder)
fram2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

# import pdb; pdb.set_trace() # used for debugging

frame1 = cv2.imread(frame1_folder)
frame2 = cv2.imread(fram2_folder)


# Step 2: Convert to Grayscale
gray1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Step 3: Detect Good Features to Track in the First Image
features = cv2.goodFeaturesToTrack(
    gray1,
    maxCorners=300,  # Maximum number of corners to detect
    qualityLevel=0.1,  # Minimum quality of corners
    minDistance=10,  # Minimum distance between corners
)

# Step 4: Track Features Using Lucas-Kanade Optical Flow
features_next, status, error = cv2.calcOpticalFlowPyrLK(
    gray1,
    gray2,
    features,
    None,
    winSize=(15, 15),  # Window size for Lucas-Kanade
    maxLevel=2,  # Number of pyramid levels
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Filter only valid points
valid_features = features[status == 1]
valid_features_next = features_next[status == 1]

# Step 5: Draw Matches
img_matches = frame1.copy()
for pt1, pt2 in zip(valid_features, valid_features_next):
    x1, y1 = pt1.ravel()
    x2, y2 = pt2.ravel()
    cv2.circle(img_matches, (int(x2), int(y2)), 5, (0, 255, 0), -1)  # Green circles
    cv2.line(img_matches, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue lines

cv2.imshow("Tracked Features", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Filter only valid points
valid_features = features[status == 1]
valid_features_next = features_next[status == 1]

# Step 6: Estimate Essential Matrix with RANSAC
K = np.array([[718.856, 0, 607.1928],
            [0, 718.856, 185.2157],
            [0, 0, 1]])

essential_mat, inliers = cv2.findEssentialMat(
    valid_features,
    valid_features_next,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=0.4,
)

# Step 7: Recover Pose
_, R, t, mask = cv2.recoverPose(essential_mat, valid_features, valid_features_next, K)

print("R: ", R)
print("t: ", t)
# Step 8: Triangulate Points
# Define projection matrices for the two views
proj_matrix1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First camera
proj_matrix2 = np.dot(K, np.hstack((R, t)))  # Second camera

# Convert points to valid_features coordinates for triangulation
valid_features = valid_features[mask.ravel() == 255].astype(np.float32)
valid_features_next = valid_features_next[mask.ravel() == 255].astype(np.float32)

# import pdb; pdb.set_trace() # used for debugging


points_4d_homogeneous = cv2.triangulatePoints(proj_matrix1, proj_matrix2, valid_features.T, valid_features_next.T)

# Convert from homogeneous coordinates to 3D
points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]

valid = np.sum(points_3d[2,:] > 0)

# Step 8: Visualize or Save the 3D Points
print("Triangulated 3D points:")
print(points_3d.T)  # Transpose to get points in rows


