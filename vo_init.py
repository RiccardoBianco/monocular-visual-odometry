import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dashboard import create_dashboard, update_dashboard

def vo_bootstrap(frame1_path, frame2_path, K):
    # Load frames
    frame1_color = cv2.imread(frame1_path, cv2.IMREAD_COLOR)
    frame2_color = cv2.imread(frame2_path, cv2.IMREAD_COLOR)
    frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints in the first frame
    corners = cv2.goodFeaturesToTrack(frame1, maxCorners=1000, qualityLevel=0.01, minDistance=7)

    # Track keypoints to the second frame
    tracked_points, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, corners, None)

    valid_corners = corners[status == 1]
    valid_tracked_corners = tracked_points[status == 1]

    # Visualize tracked points
    for (new, old) in zip(valid_tracked_corners, valid_corners):
        a, b = new.ravel()
        c, d = old.ravel()
        frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)
        frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)

    plt.figure()
    plt.imshow(cv2.cvtColor(frame2_color, cv2.COLOR_BGR2RGB))
    plt.title('Tracked Points')

    valid_corners = np.float32(valid_corners)
    valid_tracked_corners = np.float32(valid_tracked_corners)

    # Compute essential matrix and pose
    E, mask = cv2.findEssentialMat(valid_corners, valid_tracked_corners, K, method=cv2.RANSAC, prob=0.999, threshold=1)
    _, R, t, mask_pose = cv2.recoverPose(E, valid_corners, valid_tracked_corners, K)

    print("Rotation matrix: \n", R)
    print("Translation vector: \n", t)

    # Visualize inliers and outliers
    for i, (new, old) in enumerate(zip(valid_tracked_corners, valid_corners)):
        a, b = new.ravel()
        c, d = old.ravel()
        if mask_pose[i]:
            frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)
            frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        else:
            frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 0, 255), -1)
            frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)

    inlier_corners = valid_corners[mask_pose.ravel() == 255]
    inlier_tracked_corners = valid_tracked_corners[mask_pose.ravel() == 255]

    plt.figure()
    plt.imshow(cv2.cvtColor(frame2_color, cv2.COLOR_BGR2RGB))
    plt.title('Inliers and Outliers')

    # Triangulate 3D points
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First frame pose: Identity
    P2 = np.dot(K, np.hstack((R, t)))  # Second frame pose

    pts4D = cv2.triangulatePoints(P1, P2, inlier_corners.T, inlier_tracked_corners.T)
    pts3D = pts4D[:3] / pts4D[3]  # shape (3, N)

    # Plot 3D landmarks
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3D[0], pts3D[1], pts3D[2], c='r', marker='o', label='3D Points')

    camera_center1 = np.array([0, 0, 0])
    ax.scatter(*camera_center1, c='blue', marker='^', s=100, label='Camera 1')

    camera_center2 = camera_center1 - t.ravel()
    ax.scatter(*camera_center2, c='green', marker='^', s=100, label='Camera 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Return the second frame and the tracked keypoints as initialization
    return frame2, inlier_tracked_corners.reshape(-1,1,2), pts3D, R, t


def vo_continuous(new_frame_path, K, state, min_landmarks=50, min_baseline_angle=1.0):
    """
    Process a new frame for continuous visual odometry using a stateful approach.
    
    Parameters:
    - new_frame_path: path to the next image frame
    - K: Camera intrinsic matrix
    - state: dictionary containing
        {
          'db_image': previous grayscale frame,
          'P': array of shape (N,1,2) known landmark keypoints
          'X': array of shape (N,3) landmarks
          'R', 't': previous camera pose
          'C': candidate keypoints (M,1,2)
          'F': first observation of candidate keypoints (M,1,2)
          'T': list of (R_f, t_f) poses for first observation of each candidate
        }
    - min_landmarks: minimum number of landmarks required
    - min_baseline_angle: angle threshold for triangulation [radians]

    Returns:
    Updated state dictionary.
    """
    # Load new frame
    current_frame = cv2.imread(new_frame_path, cv2.IMREAD_GRAYSCALE)
    prev_frame = state['db_image']

    P = state['P']
    X = state['X']
    C = state['C']
    F_first = state['F']
    T_first = state['T']
    R_prev = state['R']
    t_prev = state['t']

    # 1. Track db_keypoints
    if P is not None and P.shape[0] > 0:
        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, P, None)
        valid_idx = status.flatten() == 1
        P_tracked = tracked_points[valid_idx]
        X_tracked = X[valid_idx]
    else:
        print("DB keypoints are empty. Resetting P and X...")
        P_tracked = np.empty((0,1,2), dtype=np.float32)
        X_tracked = np.empty((0,3), dtype=np.float32)

    # 2. Track candidate keypoints
    if C is not None and C.shape[0] > 0:
        C_tracked, status_c, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, C, None)
        valid_c_idx = status_c.flatten() == 1
        C_tracked = C_tracked[valid_c_idx]
        F_first = F_first[valid_c_idx]
        T_first = [T_first[i] for i in range(len(T_first)) if valid_c_idx[i]]
    else:
        print("Candidate keypoints are empty. Resetting C...")
        C_tracked = np.empty((0,1,2), dtype=np.float32)

    P = P_tracked
    X = X_tracked
    C = C_tracked

    # 3. Pose estimation with PnP
    R_new, t_new = R_prev, t_prev
    if X.shape[0] >= 6:
        objectPoints = X.reshape(-1,3)
        imagePoints = P.reshape(-1,2)
        distCoeffs = np.zeros((4,1))
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints, imagePoints, K, distCoeffs,
            reprojectionError=3.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success and inliers is not None and len(inliers) > 0:
            inlier_mask = np.zeros(len(objectPoints), dtype=bool)
            inlier_mask[inliers.flatten()] = True
            P = P[inlier_mask]
            X = X[inlier_mask]
            R_new, _ = cv2.Rodrigues(rvec)
            t_new = tvec
            if (t_new[0] > 30):
                print("I'm here")
            print("PnP successful...printing new pose:")
            print("Rotation matrix: \n", R_new)
            print("Translation vector: \n", t_new)

    # 4. If not enough landmarks, add new candidate keypoints
    if X.shape[0] < min_landmarks:
        new_corners = cv2.goodFeaturesToTrack(current_frame, maxCorners=200, qualityLevel=0.01, minDistance=7)
        if new_corners is not None:
            if C.shape[0] > 0:
                existing_points = np.vstack((P.reshape(-1,2), C.reshape(-1,2)))
            else:
                existing_points = P.reshape(-1,2)

            dist_threshold = 5
            keep_idx = []
            for i, cpt in enumerate(new_corners):
                c_pt = cpt.ravel()
                if existing_points.shape[0] > 0:
                    dists = np.sqrt(np.sum((existing_points - c_pt)**2, axis=1))
                    if np.all(dists > dist_threshold):
                        keep_idx.append(i)
                else:
                    keep_idx.append(i)

            if len(keep_idx) > 0:
                new_candidates = new_corners[keep_idx]

                if C.shape[0] > 0:
                    C = np.vstack((C, new_candidates))
                else:
                    C = new_candidates

                if F_first.shape[0] > 0:
                    F_first = np.vstack((F_first, new_candidates))
                else:
                    F_first = new_candidates

                for _ in range(len(new_candidates)):
                    T_first.append((R_new.copy(), t_new.copy()))

    # 5. Triangulate candidates if baseline angle is sufficient
    if C.shape[0] > 0:
        good_for_triangulation = []
        for i in range(C.shape[0]):
            c_current = C[i].reshape(1,2)
            c_first = F_first[i].reshape(1,2)
            R_f, t_f = T_first[i]

            pt_current_norm = np.linalg.inv(K).dot(np.array([c_current[0,0], c_current[0,1], 1.0]))
            pt_first_norm = np.linalg.inv(K).dot(np.array([c_first[0,0], c_first[0,1], 1.0]))

            angle = np.degrees(np.arccos(
                np.clip(np.dot(pt_current_norm/np.linalg.norm(pt_current_norm),
                       pt_first_norm/np.linalg.norm(pt_first_norm)),-1.0,1.0)
            ))

            if angle > min_baseline_angle:
                P_first = K @ np.hstack((R_f, t_f))
                P_current = K @ np.hstack((R_new, t_new))
                print("Triangulating candidate point ", i)
                pts4D = cv2.triangulatePoints(P_first, P_current, c_first.T, c_current.T)
                X_new = (pts4D[:3] / pts4D[3]).T
                P = np.vstack((P, c_current.reshape(1,1,2)))
                X = np.vstack((X, X_new))
                good_for_triangulation.append(i)

        if len(good_for_triangulation) > 0:
            mask_keep = np.ones(C.shape[0], dtype=bool)
            mask_keep[good_for_triangulation] = False
            C = C[mask_keep]
            F_first = F_first[mask_keep]
            T_first = [T_first[j] for j in range(len(T_first)) if mask_keep[j]]

    # Update the state
    state['db_image'] = current_frame
    state['P'] = P
    state['X'] = X
    state['C'] = C
    state['F'] = F_first
    state['T'] = T_first
    state['R'] = R_new
    state['t'] = t_new

    return state

######################################################################
##############################  Dashboard  ############################
######################################################################



if __name__ == "__main__":
    frame_1_relative_folder = "/datasets/parking/images/img_00000.png"
    frame_2_relative_folder = "/datasets/parking/images/img_00003.png"

    frame1_folder = os.path.join(os.path.dirname(__file__) + frame_1_relative_folder)
    frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

    K = np.array([[331.37,   0,    320],
                  [  0,    369.568, 240],
                  [  0,      0,      1]])

    # Bootstrap
    db_image, db_keypoints, db_landmarks, R, t = vo_bootstrap(frame1_folder, frame2_folder, K)
    # db_landmarks is (3, N), transpose to (N, 3)
    db_landmarks = db_landmarks.T

    # Initialize the state dictionary
    state = {
        'db_image': db_image,
        'P': db_keypoints,       # (N,1,2)
        'X': db_landmarks,       # (N,3)
        'R': R,
        't': t,
        'C': np.empty((0,1,2), dtype=np.float32), # No candidates yet
        'F': np.empty((0,1,2), dtype=np.float32), # No first observations for candidates yet
        'T': [] # No poses for candidates yet
    }
    
    camera_positions = []
    landmark_counts = []

    # Initialize dashboard
    fig, axes = create_dashboard()
    camera_positions = [t.ravel()]  # List to store trajectory points

    # Process subsequent frames
    for i in range(4, 598):  # Adjust the range to your dataset
        if i < 10:
            frame_2_relative_folder = f"/datasets/parking/images/img_0000{i}.png"
        elif i < 100:
            frame_2_relative_folder = f"/datasets/parking/images/img_000{i}.png"
        else:
            frame_2_relative_folder = f"/datasets/parking/images/img_00{i}.png"

        frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)
        state = vo_continuous(frame2_folder, K, state, min_landmarks=50, min_baseline_angle=1.0)
        # Update trajectory and landmark count
        camera_positions.append(state['t'].ravel())
        landmark_counts.append(len(state['X']))

        # Call the dashboard plot
        update_dashboard(fig, axes, state, camera_positions, frame2_folder)

    plt.show()  # Show all plots at once




# OLD VERSION: was working up until re-triangulation of new landmarks
#              it was also missing the candidate parts

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
