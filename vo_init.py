import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import threading

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

            print("PnP successful...printing new pose:")
            print("Rotation matrix: \n", R_new)
            print("Translation vector: \n", t_new)

    # 4. If not enough landmarks, add new candidate keypoints
    if X.shape[0] < min_landmarks:
        new_corners = cv2.goodFeaturesToTrack(current_frame, maxCorners=1000, qualityLevel=0.1, minDistance=5)
        if new_corners is not None:
            if C.shape[0] > 0:
                existing_points = np.vstack((P.reshape(-1,2), C.reshape(-1,2)))
            else:
                existing_points = P.reshape(-1,2)

            # Keep only the new corners that are not too close to existing points
            dist_threshold = 3
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

                # Store in F the first observation of the new candidates
                # N.B. This is bcs C is going to be updated with the new positions
                #      but we need to keep track of the first observation
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
                pts4D = cv2.triangulatePoints(P_first, P_current, c_first.T, c_current.T)
                X_new = (pts4D[:3] / pts4D[3]).T
                # Update database
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

# Function to perform bundle adjustment in a separate thread
def threaded_bundle_adjustment(K, camera_poses, landmarks, observations, result_dict, key):
    def pose_to_vector(R, t):
        rvec, _ = cv2.Rodrigues(R)
        return np.hstack((rvec.ravel(), t.ravel()))

    def vector_to_pose(vec):
        rvec, tvec = vec[:3], vec[3:]
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec

    poses_vector = np.hstack([pose_to_vector(R, t) for R, t in camera_poses])
    landmarks_vector = landmarks.ravel()

    def reprojection_error(params):
        num_cameras = len(camera_poses)
        num_landmarks = len(landmarks)

        camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
        landmark_params = params[num_cameras * 6:].reshape((num_landmarks, 3))

        errors = []

        for frame_idx, obs in enumerate(observations):
            R, t = vector_to_pose(camera_params[frame_idx])
            P = K @ np.hstack((R, t.reshape(-1, 1)))

            for i, kp in enumerate(obs['keypoints']):
                landmark_idx = obs['landmark_indices'][i]

                if landmark_idx >= num_landmarks:
                    continue

                X_world = landmark_params[landmark_idx]
                X_cam = P @ np.hstack((X_world, 1))
                X_cam /= X_cam[2]

                reproj_error = kp - X_cam[:2]
                errors.append(reproj_error)

        return np.hstack(errors)

    initial_params = np.hstack((poses_vector, landmarks_vector))
    result = least_squares(reprojection_error, initial_params, verbose=2)

    optimized_camera_params = result.x[:len(camera_poses) * 6].reshape((len(camera_poses), 6))
    optimized_landmarks = result.x[len(camera_poses) * 6:].reshape((-1, 3))
    optimized_camera_poses = [vector_to_pose(vec) for vec in optimized_camera_params]

    result_dict[key] = (optimized_camera_poses, optimized_landmarks)


# Integrate bundle adjustment into the VO pipeline with threading
if __name__ == "__main__":
    frame_1_relative_folder = "/datasets/parking/images/img_00000.png"
    frame_2_relative_folder = "/datasets/parking/images/img_00003.png"

    frame1_folder = os.path.join(os.path.dirname(__file__) + frame_1_relative_folder)
    frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

    K = np.array([[331.37,   0,    320],
                  [  0,    369.568, 240],
                  [  0,      0,      1]])

    db_image, db_keypoints, db_landmarks, R, t = vo_bootstrap(frame1_folder, frame2_folder, K)
    db_landmarks = db_landmarks.T

    state = {
        'db_image': db_image,
        'P': db_keypoints,
        'X': db_landmarks,
        'R': R,
        't': t,
        'C': np.empty((0,1,2), dtype=np.float32),
        'F': np.empty((0,1,2), dtype=np.float32),
        'T': []
    }

    camera_poses = [(R, t)]
    all_observations = []

    bundle_adjustment_results = {}

    for i in range(4, 598):
        if i < 10:
            frame_2_relative_folder = f"/datasets/parking/images/img_0000{i}.png"
        elif i < 100:
            frame_2_relative_folder = f"/datasets/parking/images/img_000{i}.png"
        else:
            frame_2_relative_folder = f"/datasets/parking/images/img_00{i}.png"

        frame2_folder = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

        state = vo_continuous(frame2_folder, K, state, min_landmarks=50, min_baseline_angle=10.0)
        camera_poses.append((state['R'], state['t']))

        if state['P'].shape[0] > 0:
            valid_landmarks = state['X'].shape[0]
            valid_indices = np.arange(valid_landmarks)

            landmark_indices = np.arange(state['P'].shape[0])
            valid_obs_mask = landmark_indices < valid_landmarks
            filtered_keypoints = state['P'][valid_obs_mask].reshape(-1, 2)
            filtered_indices = landmark_indices[valid_obs_mask]

            if filtered_keypoints.shape[0] > 0:
                observations = {
                    'keypoints': filtered_keypoints,
                    'landmark_indices': filtered_indices
                }
                all_observations.append(observations)

        if len(camera_poses) % 10 == 0:
            ba_thread = threading.Thread(
                target=threaded_bundle_adjustment,
                args=(K, camera_poses, state['X'], all_observations, bundle_adjustment_results, len(camera_poses))
            )
            ba_thread.start()
            ba_thread.join()

            if len(camera_poses) in bundle_adjustment_results:
                camera_poses, state['X'] = bundle_adjustment_results[len(camera_poses)]

    plt.show()
