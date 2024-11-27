import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VisualOdometry:
    def __init__(self, frame1_path, frame2_path, K):
        self.frame1_path = frame1_path
        self.frame2_path = frame2_path
        self.K = K
        self.frame1_color = None
        self.frame2_color = None
        self.frame1 = None
        self.frame2 = None
        self.corners = None
        self.tracked_points = None
        self.status = None
        self.error = None
        self.R = None
        self.t = None
        self.inlier_corners = None
        self.inlier_tracked_points = None
        self.pts3D = None

    def load_frames(self):
        self.frame1_color = cv2.imread(self.frame1_path, cv2.IMREAD_COLOR)
        self.frame2_color = cv2.imread(self.frame2_path, cv2.IMREAD_COLOR)
        self.frame1 = cv2.imread(self.frame1_path, cv2.IMREAD_GRAYSCALE)
        self.frame2 = cv2.imread(self.frame2_path, cv2.IMREAD_GRAYSCALE)

    def detect_keypoints(self):
        self.corners = cv2.goodFeaturesToTrack(self.frame1, maxCorners=1000, qualityLevel=0.01, minDistance=7)

    def track_keypoints(self):
        self.tracked_points, self.status, self.error = cv2.calcOpticalFlowPyrLK(self.frame1, self.frame2, self.corners, None)

    def draw_tracked_points(self):
        valid_corners = self.corners[self.status == 1]
        valid_tracked_points = self.tracked_points[self.status == 1]

        for (new, old) in zip(valid_tracked_points, valid_corners):
            a, b = new.ravel()
            c, d = old.ravel()
            self.frame2_color = cv2.circle(self.frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)
            self.frame2_color = cv2.line(self.frame2_color, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)

        plt.figure()
        plt.imshow(cv2.cvtColor(self.frame2_color, cv2.COLOR_BGR2RGB))
        plt.title('Tracked Points')

    def compute_pose(self):
        valid_corners = np.float32(self.corners[self.status == 1])
        valid_tracked_points = np.float32(self.tracked_points[self.status == 1])

        E, mask = cv2.findEssentialMat(valid_corners, valid_tracked_points, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.R, self.t, mask_pose = cv2.recoverPose(E, valid_corners, valid_tracked_points, self.K)
        # self.t[0] = -self.t[0]

        print("Rotation matrix: \n", self.R)
        print("Translation vector: \n", self.t)

        self.draw_inliers_outliers(valid_corners, valid_tracked_points, mask_pose)

    def draw_inliers_outliers(self, valid_corners, valid_tracked_points, mask_pose):
        for i, (new, old) in enumerate(zip(valid_tracked_points, valid_corners)):
            a, b = new.ravel()
            c, d = old.ravel()
            if mask_pose[i]:
                self.frame2_color = cv2.circle(self.frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)
                self.frame2_color = cv2.line(self.frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            else:
                self.frame2_color = cv2.circle(self.frame2_color, (int(a), int(b)), 5, (0, 0, 255), -1)
                self.frame2_color = cv2.line(self.frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)

        # Filter inliers based on RANSAC mask
        self.inlier_corners = valid_corners[mask_pose.ravel() == 255]
        self.inlier_tracked_points = valid_tracked_points[mask_pose.ravel() == 255]

        plt.figure()
        plt.imshow(cv2.cvtColor(self.frame2_color, cv2.COLOR_BGR2RGB))
        plt.title('Inliers and Outliers')
    
    def triangulate_3D_points(self):
        # Step 5: Triangulate 3D Landmarks
        # Projection matrices for the two frames
        P1 = np.dot(self.K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First frame (identity pose)
        P2 = np.dot(self.K, np.hstack((self.R, self.t)))  # Second frame (relative pose)

        # Triangulate points
        pts4D = cv2.triangulatePoints(P1, P2, 
                                    self.inlier_corners.T, 
                                    self.inlier_tracked_points.T)

        # Convert to 3D (homogeneous to Euclidean)
        pts3D = pts4D[:3] / pts4D[3]
        self.pts3D = pts3D.T
        self.plot_landmarks()

    def plot_landmarks(self):
        # Plot 3D landmarks
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        ax.scatter(self.pts3D[:, 0], self.pts3D[:, 1], self.pts3D[:, 2], c='r', marker='o', label='3D Points')
        
        # Plot camera positions
        # First camera at origin
        camera_center1 = np.array([0, 0, 0])
        ax.scatter(*camera_center1, c='blue', marker='^', s=100, label='Camera 1')
        
        # Second camera position
        # To analyze the translation in the world frame, you must transform t from the camera frame to the world frame using the rotation matrix R_world_camera
        # t_world = - t_camera * R_world_camera
        # TODO(Andrea): reading cv2.recoverPose documentation, it seems that t has the origin in the camera frame. So why is it [-1,0,0] if we're moving right???
        # 1 solution would be to self.t[0] = -self.t[0] in compute_pose
        # but if you do this, then all the landmarks are essentially mirrored in the x-axis. Why?
        camera_center2 = -np.dot(self.R.T, self.t).ravel() 
        ax.scatter(*camera_center2, c='green', marker='^', s=100, label='Camera 2')
        
        # Set labels and show plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    frame_1_relative_folder = "/datasets/parking/images/img_00000.png"
    frame_2_relative_folder = "/datasets/parking/images/img_00002.png"

    frame1_folder = os.path.join(os.path.dirname(__file__) + frame_1_relative_folder)
    frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

    K = np.array([[331.37, 0, 320],
                  [0, 369.568, 240],
                  [0, 0, 1]])

    vo = VisualOdometry(frame1_folder, frame2_folder, K)
    vo.load_frames()
    vo.detect_keypoints()
    vo.track_keypoints()
    vo.draw_tracked_points()
    vo.compute_pose() # --> automatically draws inliers and outliers
    vo.triangulate_3D_points() # --> automatically plots 3D landmarks
    plt.show()  # Show all plots at once
    
