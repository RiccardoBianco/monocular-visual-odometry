# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def vo_bootstrap(frame1_path, frame2_path, K):
#     # Load frames
#     frame1_color = cv2.imread(frame1_path, cv2.IMREAD_COLOR)
#     frame2_color = cv2.imread(frame2_path, cv2.IMREAD_COLOR)
#     frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
#     frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

#     # Detect keypoints
#     corners = cv2.goodFeaturesToTrack(frame1, maxCorners=1000, qualityLevel=0.01, minDistance=7)

#     # Track keypoints
#     tracked_points, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, corners, None)

#     # Draw tracked points
#     valid_corners = corners[status == 1]
#     valid_tracked_corners = tracked_points[status == 1]

#     for (new, old) in zip(valid_tracked_corners, valid_corners):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1) # BGR
#         frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2) # BGR

#     plt.figure()
#     plt.imshow(cv2.cvtColor(frame2_color, cv2.COLOR_BGR2RGB))
#     plt.title('Tracked Points')

#     # Compute pose
#     valid_corners = np.float32(valid_corners)
#     valid_tracked_corners = np.float32(valid_tracked_corners)

#     E, mask = cv2.findEssentialMat(valid_corners, valid_tracked_corners, K, method=cv2.RANSAC, prob=0.999, threshold=1)
#     _, R, t, mask_pose = cv2.recoverPose(E, valid_corners, valid_tracked_corners, K)

#     print("Rotation matrix: \n", R)
#     print("Translation vector: \n", t)

#     # Draw inliers and outliers
#     for i, (new, old) in enumerate(zip(valid_tracked_corners, valid_corners)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         if mask_pose[i]:
#             frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)
#             frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#         else:
#             frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 0, 255), -1)
#             frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)

#     inlier_corners = valid_corners[mask_pose.ravel() == 255]
#     inlier_tracked_corners = valid_tracked_corners[mask_pose.ravel() == 255]

#     plt.figure()
#     plt.imshow(cv2.cvtColor(frame2_color, cv2.COLOR_BGR2RGB))
#     plt.title('Inliers and Outliers')

#     # Triangulate 3D points
#     P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First frame (identity pose)
#     P2 = np.dot(K, np.hstack((R, t)))  # Second frame (relative pose)

#     pts4D = cv2.triangulatePoints(P1, P2, inlier_corners.T, inlier_tracked_corners.T)
#     pts3D = pts4D[:3] / pts4D[3]


#     # Plot 3D landmarks
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pts3D[0], pts3D[1], pts3D[2], c='r', marker='o', label='3D Points')

#     camera_center1 = np.array([0, 0, 0])
#     ax.scatter(*camera_center1, c='blue', marker='^', s=100, label='Camera 1')

#     camera_center2 = camera_center1 + t
#     ax.scatter(*camera_center2, c='green', marker='^', s=100, label='Camera 2')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()

#     return frame2, inlier_tracked_corners, pts3D, R, t


# def vo_continuous(frame2_path, K, db_image, db_keypoints, db_landmarks, R, t, min_landmarks=10):
#     """
#     Process the new frame for continuous visual odometry.

#     Inputs:
#     - frame1_path, frame2_path: paths to consecutive frames
#     - K: camera intrinsics
#     - db_image: previous frame (grayscale)
#     - db_keypoints: np.array of shape (N,1,2) representing the 2D keypoints in the db_image
#     - db_landmarks: np.array of shape (N,3), 3D points corresponding to db_keypoints
#     - R, t: previous pose of the camera
#     - min_landmarks: threshold for minimum number of landmarks required

#     Output:
#     - updated_frame: current frame (grayscale)
#     - updated_keypoints: 2D keypoints tracked in the current frame
#     - updated_landmarks: corresponding 3D landmarks of the tracked keypoints
#     - R_new, t_new: new camera pose
#     """

#     # Load frames
#     frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

#     # Track keypoints from db_image to current frame2
#     # Note: db_keypointnumbers should be in shape (N,1,2) for calcOpticalFlowPyrLK
#     tracked_points, status, error = cv2.calcOpticalFlowPyrLK(db_image, frame2, db_keypoints, None)

    
#     # Filter for valid tracks
#     valid_idx = (status == 1).ravel()
#     valid_db_keypoints = db_keypoints[valid_idx]
#     valid_tracked_points = tracked_points[valid_idx]
#     valid_landmarks = db_landmarks[valid_idx]

#     # If we have at least 6 points (PnP requires at least some correspondences), run PnP
#     if len(valid_landmarks) > 6:
#         # Solve PnP
#         # We need objectPoints: Nx3, imagePoints: Nx2
#         objectPoints = valid_landmarks.reshape(-1,3)
#         imagePoints = valid_tracked_points.reshape(-1,2)

#         # Distortion assumed zero for simplicity; adjust if needed
#         distCoeffs = np.zeros((4,1))

#         # run solvePnPRansac
#         # Flags could be: cv2.SOLVEPNP_ITERATIVE or cv2.SOLVEPNP_AP3P, etc.
#         # R and t were previously from bootstrap: we can use them as initial guess if needed
#         success, rvec, tvec, inliers = cv2.solvePnPRansac(
#             objectPoints, imagePoints, K, distCoeffs,
#             reprojectionError=3.0, # Adjust threshold as needed
#             flags=cv2.SOLVEPNP_ITERATIVE
#         )

#         if success and len(inliers) > 0:
#             # Filter by inliers
#             inlier_mask = np.zeros(len(objectPoints), dtype=bool)
#             inlier_mask[inliers.flatten()] = True

#             # Keep only inliers
#             valid_landmarks = valid_landmarks[inlier_mask]
#             valid_tracked_points = valid_tracked_points[inlier_mask]

#             # Convert rvec, tvec to R, t
#             R_new, _ = cv2.Rodrigues(rvec)
#             t_new = tvec

#             print("Rotation matrix: \n", R_new)
#             print("Translation vector: \n", t_new)
#         else:
#             # If PnP fails, fallback to previous pose or handle error
#             R_new, t_new = R, t
#     else:
#         # Not enough points to run PnP.
#         # Could handle by just keeping previous pose, or try another method:
#         R_new, t_new = R, t

#     # Now we have updated R_new, t_new. We must update the db_image and db_keypoints, db_landmarks
#     updated_frame = frame2
#     updated_keypoints = valid_tracked_points.reshape(-1,1,2)
#     updated_landmarks = valid_landmarks

#     # Remove landmarks that were lost (we already did by filtering valid_idx)
#     # If the number of landmarks is too low, re-triangulate new landmarks
#     if updated_landmarks.shape[0] < min_landmarks:
#         print("Re-triangulating new landmarks...")
#         # Re-triangulate new landmarks:
#         # 1. Detect new features in current frame that are not in updated_keypoints
#         # 2. Match or track these new features back to previous frame or another keyframe
#         #    so that we can triangulate
#         #
#         # As a simple example, let's just detect new corners in the current frame:
#         new_corners = cv2.goodFeaturesToTrack(updated_frame, maxCorners=500, qualityLevel=0.01, minDistance=7)

#         # Filter out corners that are too close to existing updated_keypoints
#         # A simple heuristic: Remove any corner that is within some pixel distance of an existing keypoint
#         if new_corners is not None:
#             existing_points = updated_keypoints.reshape(-1, 2)
#             dist_threshold = 5
#             keep_idx = []
#             for i, c in enumerate(new_corners):
#                 c_pt = c.ravel()
#                 # Check distance to existing points
#                 dists = np.sqrt(np.sum((existing_points - c_pt)**2, axis=1))
#                 if np.all(dists > dist_threshold):
#                     keep_idx.append(i)
#             new_corners = new_corners[keep_idx]

#             # Now we have some new corners.
#             # We need a second view (the previous frame and its pose) to triangulate them.
#             # Let's try tracking them back to the previous db_image:
#             # NOTE: This is simplistic. In reality, you'd want a keyframe or a frame that differs.
#             # Here we just assume we can track them back to the previous frame for demonstration.
#             back_tracked_points, st, err = cv2.calcOpticalFlowPyrLK(updated_frame, db_image, new_corners, None)

#             # Filter valid back tracks
#             valid_back = (st == 1).ravel()
#             re_tri_corners = new_corners[valid_back]
#             re_tri_back = back_tracked_points[valid_back]

#             # Triangulate using previous pose (R, t) and current pose (R_new, t_new)
#             P1 = K @ np.hstack((R, t))
#             P2 = K @ np.hstack((R_new, t_new))
#             pts4D = cv2.triangulatePoints(P1, P2, re_tri_back.T, re_tri_corners.T)
#             pts3D_new = (pts4D[:3] / pts4D[3]).T
#             pts3D_new = pts3D_new.reshape(-1,3)

#             # Add these new landmarks to our updated sets
#             updated_keypoints = np.vstack((updated_keypoints, re_tri_corners))
#             updated_landmarks = np.vstack((updated_landmarks, pts3D_new))

#     return updated_frame, updated_keypoints, updated_landmarks, R_new, t_new

# # def vo_continuos(frame1_path, frame2_path, K, db_image, db_keypoints, db_landmarks, R, t):
# #     # Load frames
# #     frame1_color = cv2.imread(frame1_path, cv2.IMREAD_COLOR)
# #     frame2_color = cv2.imread(frame2_path, cv2.IMREAD_COLOR)
# #     frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
# #     frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

# #     # Track keypoints
# #     tracked_points, status, error = cv2.calcOpticalFlowPyrLK(db_image, frame2, db_keypoints, None)

# #     # Filter valid points
# #     valid_db_keypoints = db_keypoints[status == 1]
# #     valid_tracked_points = tracked_points[status == 1]

# #     # Estimate pose
# #     E, mask = cv2.findEssentialMat(valid_db_keypoints, valid_tracked_points, K, method=cv2.RANSAC, prob=0.999, threshold=1)
# #     _, R_new, t_new, mask_pose = cv2.recoverPose(E, valid_db_keypoints, valid_tracked_points, K)

# #     # Update state
# #     inlier_db_keypoints = valid_db_keypoints[mask_pose.ravel() == 255]
# #     inlier_tracked_points = valid_tracked_points[mask_pose.ravel() == 255]

# #     # Triangulate new landmarks
# #     P1 = np.dot(K, np.hstack((R, t)))
# #     P2 = np.dot(K, np.hstack((R_new, t_new)))

# #     pts4D = cv2.triangulatePoints(P1, P2, inlier_db_keypoints.T, inlier_tracked_points.T)
# #     pts3D_new = pts4D[:3] / pts4D[3]

# #     # Update database
# #     db_image = frame2
# #     db_keypoints = tracked_points
# #     db_landmarks = pts3D_new

# #     print("Rotation matrix: \n", R_new)
# #     print("Translation vector: \n", t_new)


# #     return frame2, inlier_tracked_points, db_landmarks, R_new, t_new
    
    
    

# if __name__ == "__main__":
#     frame_1_relative_folder = "/datasets/parking/images/img_00000.png"
#     frame_2_relative_folder = "/datasets/parking/images/img_00003.png"

#     frame1_folder = os.path.join(os.path.dirname(__file__) + frame_1_relative_folder)
#     frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

#     K = np.array([[331.37, 0, 320],
#                   [0, 369.568, 240],
#                   [0, 0, 1]])

#     db_image, db_keypoints, db_landmarks, R, t, = vo_bootstrap(frame1_folder, frame2_folder, K)
    
#     db_landmarks = db_landmarks.T ## needed for vo_continuos
    
#     for i in range(4, 598):  # Assuming you want to process frames from img_00004.png to img_00597.png
#         if i < 10:
#             frame_2_relative_folder = f"/datasets/parking/images/img_0000{i}.png"
#         elif i < 100:
#             frame_2_relative_folder = f"/datasets/parking/images/img_000{i}.png"
#         else:
#             frame_2_relative_folder = f"/datasets/parking/images/img_00{i}.png"

#         frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)
#         db_image, db_keypoints, db_landmarks, R, t = vo_continuous(frame2_folder, K, db_image, db_keypoints, db_landmarks, R, t, min_landmarks=10)

#     plt.show()  # Show all plots at once



import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotation_matrix_to_angle(R):
    # Extract rotation angle from a rotation matrix using trace
    # angle = arccos((trace(R) - 1) / 2)
    angle = np.arccos((np.trace(R) - 1) / 2.0)
    return angle

def vo_bootstrap(frame1_path, frame2_path, K):
    # Load frames
    frame1_color = cv2.imread(frame1_path, cv2.IMREAD_COLOR)
    frame2_color = cv2.imread(frame2_path, cv2.IMREAD_COLOR)
    frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints
    corners = cv2.goodFeaturesToTrack(frame1, maxCorners=1000, qualityLevel=0.01, minDistance=7)

    # Track keypoints
    tracked_points, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, corners, None)

    # Draw tracked points
    valid_corners = corners[status == 1]
    valid_tracked_corners = tracked_points[status == 1]

    # Compute pose
    valid_corners = np.float32(valid_corners)
    valid_tracked_corners = np.float32(valid_tracked_corners)

    E, mask = cv2.findEssentialMat(valid_corners, valid_tracked_corners, K, method=cv2.RANSAC, prob=0.999, threshold=1)
    _, R, t, mask_pose = cv2.recoverPose(E, valid_corners, valid_tracked_corners, K)

    print("Rotation matrix: \n", R)
    print("Translation vector: \n", t)

    inlier_corners = valid_corners[mask_pose.ravel() == 255]
    inlier_tracked_corners = valid_tracked_corners[mask_pose.ravel() == 255]

    # Triangulate 3D points
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First frame (identity pose)
    P2 = np.dot(K, np.hstack((R, t)))  # Second frame (relative pose)

    pts4D = cv2.triangulatePoints(P1, P2, inlier_corners.T, inlier_tracked_corners.T)
    pts3D = pts4D[:3] / pts4D[3]

    return frame2, inlier_tracked_corners.reshape(-1,1,2), pts3D.T, R, t


def vo_continuous(frame2_path, K,
                  db_image, db_keypoints, db_landmarks, R, t,
                  keyframe_image, keyframe_keypoints, keyframe_landmarks, keyframe_R, keyframe_t,
                  min_landmarks=50,
                  BASELINE_TRANSLATION_THRESHOLD=0.5,
                  BASELINE_ROTATION_THRESHOLD=np.deg2rad(5)): 
    """
    Process the new frame for continuous visual odometry with keyframe mechanism and baseline check.

    Inputs:
    - frame2_path: path to the current frame
    - K: camera intrinsics
    - db_image: previous frame (grayscale)
    - db_keypoints: np.array of shape (N,1,2) representing the 2D keypoints in the db_image
    - db_landmarks: np.array of shape (N,3), 3D points corresponding to db_keypoints
    - R, t: previous pose of the camera
    - keyframe_image, keyframe_keypoints, keyframe_landmarks: data of the keyframe
    - keyframe_R, keyframe_t: pose of the keyframe
    - min_landmarks: threshold for minimum number of landmarks required
    - BASELINE_TRANSLATION_THRESHOLD, BASELINE_ROTATION_THRESHOLD: thresholds for deciding when to re-triangulate

    Output:
    - updated_frame: current frame (grayscale)
    - updated_keypoints: 2D keypoints tracked in the current frame
    - updated_landmarks: corresponding 3D landmarks of the tracked keypoints
    - R_new, t_new: new camera pose
    - keyframe_image, keyframe_keypoints, keyframe_landmarks, keyframe_R, keyframe_t: possibly updated keyframe data
    """

    # Load current frame
    frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

    # Track keypoints from db_image to current frame2
    tracked_points, status, error = cv2.calcOpticalFlowPyrLK(db_image, frame2, db_keypoints, None)

    # Filter for valid tracks
    valid_idx = (status == 1).ravel()
    valid_db_keypoints = db_keypoints[valid_idx]
    valid_tracked_points = tracked_points[valid_idx]
    valid_landmarks = db_landmarks[valid_idx]

    # If we have enough points, run PnP
    if len(valid_landmarks) > 6:
        objectPoints = valid_landmarks.reshape(-1,3)
        imagePoints = valid_tracked_points.reshape(-1,2)
        distCoeffs = np.zeros((4,1))

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints, imagePoints, K, distCoeffs,
            reprojectionError=3.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success and inliers is not None and len(inliers) > 0:
            inlier_mask = np.zeros(len(objectPoints), dtype=bool)
            inlier_mask[inliers.flatten()] = True

            valid_landmarks = valid_landmarks[inlier_mask]
            valid_tracked_points = valid_tracked_points[inlier_mask]

            R_new, _ = cv2.Rodrigues(rvec)
            t_new = tvec
        else:
            # Fallback if PnP fails
            R_new, t_new = R, t
    else:
        R_new, t_new = R, t

    updated_frame = frame2
    updated_keypoints = valid_tracked_points.reshape(-1,1,2)
    updated_landmarks = valid_landmarks

    # If we don't have enough landmarks, consider re-triangulation
    if updated_landmarks.shape[0] < min_landmarks:
        # Check baseline with keyframe
        baseline_translation = np.linalg.norm(t_new - keyframe_t)
        # Compute rotation difference angle between new frame and keyframe
        R_diff = R_new @ keyframe_R.T
        rotation_angle = rotation_matrix_to_angle(R_diff)

        if baseline_translation > BASELINE_TRANSLATION_THRESHOLD or rotation_angle > BASELINE_ROTATION_THRESHOLD:
            print("Re-triangulating new landmarks...")

            # Detect new corners in current frame
            new_corners = cv2.goodFeaturesToTrack(updated_frame, maxCorners=500, qualityLevel=0.01, minDistance=7)

            if new_corners is not None and len(new_corners) > 0:
                existing_points = updated_keypoints.reshape(-1, 2)
                dist_threshold = 5
                keep_idx = []
                for i, c in enumerate(new_corners):
                    c_pt = c.ravel()
                    # Check distance to existing points
                    dists = np.sqrt(np.sum((existing_points - c_pt)**2, axis=1))
                    if np.all(dists > dist_threshold):
                        keep_idx.append(i)
                new_corners = new_corners[keep_idx]

                # Track them back to the keyframe image (NOT just previous frame)
                # This ensures a baseline between keyframe and current frame
                if new_corners is not None and len(new_corners) > 0:
                    back_tracked_points, st, err = cv2.calcOpticalFlowPyrLK(updated_frame, keyframe_image, new_corners, None)

                    valid_back = (st == 1).ravel()
                    re_tri_corners = new_corners[valid_back]
                    re_tri_back = back_tracked_points[valid_back]

                    if len(re_tri_corners) > 0:
                        P1 = K @ np.hstack((keyframe_R, keyframe_t))
                        P2 = K @ np.hstack((R_new, t_new))

                        pts4D = cv2.triangulatePoints(P1, P2, re_tri_back.T, re_tri_corners.T)
                        # Avoid division by zero
                        mask_valid = (pts4D[3] != 0)
                        pts4D = pts4D[:, mask_valid]
                        if pts4D.shape[1] > 0:
                            pts3D_new = (pts4D[:3] / pts4D[3]).T
                            # Add these new landmarks
                            re_tri_corners = re_tri_corners[mask_valid]
                            updated_keypoints = np.vstack((updated_keypoints, re_tri_corners.reshape(-1,1,2)))
                            updated_landmarks = np.vstack((updated_landmarks, pts3D_new))

            # Update the keyframe if desired
            # We can set the current frame as the new keyframe if it provides a good baseline
            keyframe_image = updated_frame.copy()
            keyframe_keypoints = updated_keypoints.copy()
            keyframe_landmarks = updated_landmarks.copy()
            keyframe_R = R_new.copy()
            keyframe_t = t_new.copy()

    return updated_frame, updated_keypoints, updated_landmarks, R_new, t_new, keyframe_image, keyframe_keypoints, keyframe_landmarks, keyframe_R, keyframe_t


if __name__ == "__main__":
    frame_1_relative_folder = "/datasets/parking/images/img_00000.png"
    frame_2_relative_folder = "/datasets/parking/images/img_00003.png"

    frame1_folder = os.path.join(os.path.dirname(__file__) + frame_1_relative_folder)
    frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

    K = np.array([[331.37, 0, 320],
                  [0, 369.568, 240],
                  [0, 0, 1]])

    db_image, db_keypoints, db_landmarks, R, t = vo_bootstrap(frame1_folder, frame2_folder, K)

    # Set the initial keyframe as the bootstrap frame2
    db_landmarks = db_landmarks.T
    keyframe_image = db_image.copy()
    keyframe_keypoints = db_keypoints.copy()
    keyframe_landmarks = db_landmarks.T.copy()
    keyframe_R = R.copy()
    keyframe_t = t.copy()

    for i in range(4, 598):
        if i < 10:
            frame_2_relative_folder = f"/datasets/parking/images/img_0000{i}.png"
        elif i < 100:
            frame_2_relative_folder = f"/datasets/parking/images/img_000{i}.png"
        else:
            frame_2_relative_folder = f"/datasets/parking/images/img_00{i}.png"

        frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

        db_image, db_keypoints, db_landmarks, R, t, keyframe_image, keyframe_keypoints, keyframe_landmarks, keyframe_R, keyframe_t = vo_continuous(
            frame2_folder, K, 
            db_image, db_keypoints, db_landmarks, R, t,
            keyframe_image, keyframe_keypoints, keyframe_landmarks, keyframe_R, keyframe_t,
            min_landmarks=10
        )

    plt.show()  # Show all plots at once
