import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from params_loader import load_parameters, Dataset

################  DASHBOARD UTILITY  ############################
# Define a global flag
stop_pipeline = False


def on_key_press(event):
    global stop_pipeline
    if event.key == ' ':
        print("Stop signal received.")
        stop_pipeline = True

def start_key_listener(fig):
    fig.canvas.mpl_connect('key_press_event', on_key_press)
################################################################
### Load parameters - CHANGE ONLY THIS ###
dataset = Dataset.MALAGA  # or Dataset.PARKING or Dataset.MALAGA

### Plotting options - eventually change these #######
plot_bootstrap = False
plot_dashboard = True
plot_ground_truth = False
plot_vo_continuous_inliers_outliers = False

### Utils #######
params = load_parameters(dataset)
file_relative_folder = os.path.dirname(__file__)

lk_params = dict(winSize=params['winSize'],
        maxLevel=params['maxLevel'],
        criteria=params['criteria'])
    

def vo_bootstrap(frame2_color, frame1, frame2, K):
    # Load frames --> this gives problems
    # frame1 = cv2.cvtColor(frame1_color, cv2.COLOR_BGR2GRAY)
    # frame2 = cv2.cvtColor(frame2_color, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints in the first frame
    # Harris returns the right transformation [1,0,0], Shi returns [-1,0,0], probably bcs from cam2 frame (?) although it should be the opposite
    # quality-level: higher is stricter (discard all the ones with quality < x * max_quality)
    if params['bootstrap_detector'] == 'Shi-Tomasi':
        corners = cv2.goodFeaturesToTrack(frame1, 
                                          maxCorners=params['max_corners_bootstrap'], 
                                          qualityLevel=params['quality_level_bootstrap'], 
                                          minDistance=params['min_distance_bootstrap']) # [-0.99,0.0,-0.04]
        # corners = cv2.goodFeaturesToTrack(frame1, maxCorners=150, qualityLevel=0.000011, minDistance=5) # [0.93,-0.04,0.35]
    elif params['bootstrap_detector'] == 'Harris':
        corners = cv2.goodFeaturesToTrack(frame1, 
                                          maxCorners=params['max_corners_bootstrap'], 
                                          qualityLevel=params['quality_level_bootstrap'], 
                                          minDistance=params['min_distance_bootstrap'],
                                          useHarrisDetector=True,
                                          k=params['k_bootstrap']) # [0.99,0.0,-0.12]
        # Interesting fact to prove how sensitive these params are... 
        # with minDistance=5 and k=0.03, it returns [ 0.99,0.0,-0.1]
        # with minDistance=7 and k=0.03, it returns [-0.92,0.0, 0.3]...

    # Track keypoints to the second frame
    
    tracked_points, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, corners, nextPts=None, **lk_params)
    if error is not None:
        valid_keypoints_mask = error < params['KLT_threshold']
        mask_KLT = np.logical_and(valid_keypoints_mask, status)
    else:
        mask_KLT = status

    valid_corners = corners[mask_KLT == 1]
    valid_tracked_corners = tracked_points[mask_KLT == 1]

    # Visualize tracked points
    for (new, old) in zip(valid_tracked_corners, valid_corners):
        a, b = new.ravel()
        c, d = old.ravel()
        frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)
        frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)
    
    if plot_bootstrap == True:
        plt.figure()
        plt.imshow(cv2.cvtColor(frame2_color, cv2.COLOR_BGR2RGB))
        plt.title('Tracked Points')

    valid_corners = np.float32(valid_corners)
    valid_tracked_corners = np.float32(valid_tracked_corners)

    # Compute essential matrix and pose
    E, mask = cv2.findEssentialMat(valid_corners, valid_tracked_corners, K, method=cv2.RANSAC, prob=params['RANSAC_Essential_Matrix_confidence'], threshold=1)
    _, R, t, mask_pose = cv2.recoverPose(E, valid_corners, valid_tracked_corners, K)
    # t = -t

    # print("Rotation matrix: \n", R)
    # print("Translation vector: \n", t)

    # Combine masks to determine final inliers and outliers
    final_mask = (mask.ravel() == 1) & (mask_pose.ravel() == 255)

    # Filter inliers and outliers
    inlier_corners = valid_corners[final_mask]
    inlier_tracked_corners = valid_tracked_corners[final_mask]

    outlier_corners = valid_corners[~final_mask]
    outlier_tracked_corners = valid_tracked_corners[~final_mask]

    # Visualize inliers and outliers
    for (new, old) in zip(inlier_tracked_corners, inlier_corners):
        a, b = new.ravel()
        c, d = old.ravel()
        frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)  # Green for inliers
        frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

    for (new, old) in zip(outlier_tracked_corners, outlier_corners):
        a, b = new.ravel()
        c, d = old.ravel()
        frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 0, 255), -1)  # Red for outliers
        frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)


    # Triangulate 3D points
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # First frame pose: Identity
    P2 = np.dot(K, np.hstack((R, t)))  # Second frame pose

    pts4D = cv2.triangulatePoints(P1, P2, inlier_corners.T, inlier_tracked_corners.T)
    pts3D = pts4D[:3] / pts4D[3]  # shape (3, N)

    # Plot 3D landmarks
    if plot_bootstrap == True:
        plt.figure()
        plt.imshow(cv2.cvtColor(frame2_color, cv2.COLOR_BGR2RGB))
        plt.title('Inliers (Green) and Outliers (Red)')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts3D[0], pts3D[1], pts3D[2], c='r', marker='o', label='3D Points')

        camera_center1 = np.array([0, 0, 0])
        ax.scatter(*camera_center1, c='blue', marker='^', s=100, label='Camera 1')

        camera_center2 = camera_center1 + t.ravel()
        ax.scatter(*camera_center2, c='green', marker='^', s=100, label='Camera 2')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    # Return the second frame and the tracked keypoints as initialization
    return frame2, inlier_tracked_corners.reshape(-1,1,2), pts3D, R, t


def vo_continuous(current_frame_color, current_frame, K, state, min_landmarks=150, min_baseline_angle=5.0):
    """
    Process a new frame for continuous visual odometry using a stateful approach.
    
    Parameters:
    - new_frame_path: path to the next image frame
    - K: Camera intrinsic matrix
    - state: dictionary containing
        {
          'db_image': previous grayscale frame,
          'P': array of shape (N,1,2) known landmark keypoints (2D)
          'X': array of shape (N,3) landmarks (3D)
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
    # current_frame_color = cv2.imread(new_frame_path, cv2.IMREAD_COLOR)
    # current_frame = cv2.imread(new_frame_path, cv2.IMREAD_GRAYSCALE)
    
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
        tracked_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, P, nextPts=None, **lk_params)
        if error is not None:
            valid_keypoints_mask = error < params['KLT_threshold']
            final_mask_db = np.logical_and(valid_keypoints_mask, status)
        else:
            final_mask_db = status
        # num_errors_over_30 = np.sum(error > 30)
        # print(f"Number of errors > 30: {num_errors_over_30}")
        
        valid_idx = final_mask_db.flatten() == 1
        
        valid_corners = P[valid_idx]
        P_tracked = tracked_points[valid_idx]
        X_tracked = X[valid_idx]
    else:
        print("DB keypoints are empty. Resetting P and X...")
        P_tracked = np.empty((0,1,2), dtype=np.float32)
        X_tracked = np.empty((0,3), dtype=np.float32)

    # 2. Track candidate keypoints
    if C is not None and C.shape[0] > 0:
        C_tracked, status_c, error = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, C, nextPts=None, **lk_params)
        if error is not None:
            valid_keypoints_mask = error < params['KLT_threshold']
            final_mask_c = np.logical_and(valid_keypoints_mask, status_c)
        else:
            final_mask_c = status_c
        
        # valid_candidate_corners = C[final_mask_c == 1]

        valid_c_idx = final_mask_c.flatten() == 1
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
    inlier_points = None
    inlier_tracked_points = None
    outlier_points = None
    outlier_tracked_points = None
    R_new, t_new = R_prev, t_prev
    if X.shape[0] >= params['PnP_min_landmarks']:
        objectPoints = X.reshape(-1,3)
        imagePoints = P.reshape(-1,2)
        distCoeffs = np.zeros((4,1))
        if params['PnP_method'] == cv2.SOLVEPNP_ITERATIVE:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints, imagePoints, K, distCoeffs,
                reprojectionError=params['PnP_reprojection_error'],
                flags=params['PnP_method'],
                confidence=params['RANSAC_PnP_confidence'],
            )
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints, imagePoints, K, distCoeffs,
                flags=params['PnP_method'],
                confidence=params['RANSAC_PnP_confidence'],
            )
        
        if success and inliers is not None and len(inliers) > 0:

            inlier_mask = np.zeros(len(objectPoints), dtype=bool)
            inlier_mask[inliers.flatten()] = True 
            
            # Visualization: Inliers and outliers
            inlier_points = valid_corners[inlier_mask]
            inlier_tracked_points = P[inlier_mask]

            outlier_points = valid_corners[~inlier_mask]
            outlier_tracked_points = P[~inlier_mask] if len(P) > len(inlier_tracked_points) else np.empty((0, 1, 2))

            P = P[inlier_mask]
            X = X[inlier_mask]
            R_new, _ = cv2.Rodrigues(rvec)
            t_new = tvec

            # print("PnP successful...printing new pose:")
            # print("Rotation matrix: \n", R_new)
            # print("Translation vector: \n", t_new)
        else:
            print("PnP failed...keeping previous pose.")
    else:
        print("Not enough landmarks for PnP...keeping previous pose.")

    # Visualize db_keypoints inliers and outliers
    # if inlier_tracked_points is not None and inlier_tracked_points.shape[0] > 0:
    for (new, old) in zip(inlier_tracked_points, inlier_points):
        a, b = new.ravel()
        c, d = old.ravel()
        # current_frame_color = cv2.circle(current_frame_color, (int(a), int(b)), 5, (0, 255, 0), -1)  # Green for inliers
        current_frame_color = cv2.line(current_frame_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

    # if outlier_tracked_points is not None and outlier_tracked_points.shape[0] > 0:
    for (new, old) in zip(outlier_tracked_points, outlier_points):
        a, b = new.ravel()
        c, d = old.ravel()
        # current_frame_color = cv2.circle(current_frame_color, (int(a), int(b)), 5, (0, 0, 255), -1)  # Red for outliers
        current_frame_color = cv2.line(current_frame_color, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)

    # 4. Add new candidate keypoints
    if params['vo_continuous_detector'] == 'Shi-Tomasi':
        new_corners = cv2.goodFeaturesToTrack(current_frame, 
                                              maxCorners=params['max_corners_continuous'], 
                                              qualityLevel=params['quality_level_continuous'], 
                                              minDistance=params['min_distance_continuous'])
    elif params['vo_continuous_detector'] == 'Harris':
        new_corners = cv2.goodFeaturesToTrack(current_frame, 
                                              maxCorners=params['max_corners_continuous'],
                                              qualityLevel=params['quality_level_continuous'], 
                                              minDistance=params['min_distance_continuous'], 
                                              useHarrisDetector=True, 
                                              k=params['k_continuous'])
    
    if new_corners is not None:
        if C.shape[0] > 0:
            existing_points = np.vstack((P.reshape(-1,2), C.reshape(-1,2)))
        else:
            existing_points = P.reshape(-1,2)

        # Keep only the new corners that are not too close to existing points
        dist_threshold = 8 #pixels
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

            # Convert the keypoints to normalized camera coordinates
            pt_current_norm = np.linalg.inv(K).dot(np.array([c_current[0,0], c_current[0,1], 1.0]))
            pt_first_norm = np.linalg.inv(K).dot(np.array([c_first[0,0], c_first[0,1], 1.0]))

            # Compute the angle between the 2 vectors (same fixed z=1, only u,v changes)
            angle = np.degrees(np.arccos(
                np.clip(np.dot(pt_current_norm/np.linalg.norm(pt_current_norm),
                       pt_first_norm/np.linalg.norm(pt_first_norm)),-1.0,1.0)
            ))

            if angle > min_baseline_angle:
                P_first = K @ np.hstack((R_f, t_f))
                P_current = K @ np.hstack((R_new, t_new))
                pts4D = cv2.triangulatePoints(P_first, P_current, c_first.T, c_current.T)
                # It looks like it outputs already in camera world frame
                # I tried multiplying by R_f.T and t_f, but it hallucinates, so probably wrong
                # X_world = R_f.T @ (X_new - t_f)
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
    
    # Visualize the frame with tracked keypoints
    if plot_vo_continuous_inliers_outliers:
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2RGB))
        plt.title('Inliers (Green) and Outliers (Red)')
        plt.show()

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

def update_dashboard(
        ax_img, ax_landmark_count, ax_trajectory_partial, ax_trajectory_full,
        current_frame_color, db_keypoints, full_trajectory, 
        landmark_counts, 
        partial_window=20
    ):
        """
        Updates the four subplots in your dashboard.

        Parameters:
        -----------
        ax_img : matplotlib Axes
            Axes object for the current image with keypoints.
        ax_landmark_count : matplotlib Axes
            Axes object for the # tracked landmarks over recent frames.
        ax_trajectory_partial : matplotlib Axes
            Axes object for the last 'partial_window' frames trajectory.
        ax_trajectory_full : matplotlib Axes
            Axes object for the full trajectory from the beginning.
        current_frame_color : np.ndarray (H,W,3)
            The current color (BGR) image with drawn keypoints.
        db_keypoints : np.ndarray (N,1,2)
            The tracked keypoints for the current frame.
        full_trajectory : list of (x, y)
            A list containing the global 2D positions (or 3D if you prefer top-down) 
            of the camera from the start to current frame.
        landmark_counts : list of int
            A list containing the number of tracked landmarks in each frame.
        partial_window : int
            Number of recent frames to display in the partial trajectory.
        """
        # 1) Current image with keypoints
        ax_img.clear()
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2RGB)
        ax_img.imshow(img_rgb)
        ax_img.set_title("Current image")
        ax_img.set_aspect('equal', adjustable='box')

        # 2) # tracked landmarks over last partial_window frames
        ax_landmark_count.clear()
        ax_landmark_count.plot(
            range(max(0, len(landmark_counts) - partial_window), len(landmark_counts)),
            landmark_counts[-partial_window:] if len(landmark_counts) > partial_window else landmark_counts,
            marker='o'
        )
        ax_landmark_count.set_title(f"# tracked landmarks (last {partial_window} frames)")
        ax_landmark_count.set_aspect('equal', adjustable='box')

        # 3) Trajectory of last partial_window frames (2D top-down view)
        ax_trajectory_partial.clear()
        if len(full_trajectory) > 0:
            # Show only last partial_window positions
            recent_trajectory = full_trajectory[-partial_window:]
            xs = [p[0] for p in recent_trajectory]
            ys = [p[1] for p in recent_trajectory]
            ax_trajectory_partial.scatter(xs, ys, c='k', s=20)
            ax_trajectory_partial.set_title(f"Trajectory of last {partial_window} frames")
        ax_trajectory_partial.set_aspect('equal', 'box')
        ax_trajectory_partial.set_aspect('equal', adjustable='box')


        # Plot ground truth trajectory
        ax_trajectory_full.clear()

        if plot_ground_truth and dataset != Dataset.MALAGA:
            ground_truth_file = file_relative_folder + params['ground_truth_path']
            if os.path.exists(ground_truth_file):
                ground_truth_data = np.loadtxt(ground_truth_file)
                gt_xs = ground_truth_data[:, 3]
                gt_zs = ground_truth_data[:, 11]
                ax_trajectory_full.plot(gt_xs, gt_zs, 'r-', label='Ground Truth Trajectory')

        # 4) Full trajectory
        if len(full_trajectory) > 0:
            xs_full = [p[0] for p in full_trajectory]
            ys_full = [p[1] for p in full_trajectory]
            ax_trajectory_full.plot(xs_full, ys_full, 'b-', label='Estimated Trajectory')



        ax_trajectory_full.set_title("Full trajectory")
        ax_trajectory_full.set_aspect('equal', adjustable='box')
        ax_trajectory_full.legend()

        ax_img.set_box_aspect(1)  # Keeps the box square
        ax_traj_partial.set_box_aspect(1)
        ax_landmark_count.set_box_aspect(1)
        ax_traj_full.set_box_aspect(1)

        # plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # short pause to allow the figure to update

if __name__ == "__main__":
    if dataset == Dataset.PARKING:
        images = sorted([f for f in os.listdir(file_relative_folder + params['relative_folder']) if f != "K.txt"])
        ground_truth = np.loadtxt(file_relative_folder + "/datasets/parking/poses.txt")
        ground_truth = ground_truth[:, [3, 11]]
        last_frame = len(images) - 1

        K = np.array([[331.37,   0,    320],
                    [  0,    369.568, 240],
                    [  0,      0,      1]])
    
    elif dataset == Dataset.MALAGA:
        images = sorted(os.listdir(file_relative_folder + params['relative_folder']))
        left_images = images[2::2]
        images = left_images
        last_frame = len(left_images) - 1

        K = np.array([[621.18428, 0, 404.0076],
                    [0, 621.18428, 309.05989],
                    [0, 0, 1]])
        
    elif dataset == Dataset.KITTI:
        ground_truth = np.loadtxt(file_relative_folder + "/datasets/kitti/poses/05.txt")
        ground_truth = ground_truth[:, [3, 11]]
        images = sorted(os.listdir(file_relative_folder + params['relative_folder']))
        last_frame = len(images) - 1

        K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])
        
    
    frame2_color = cv2.imread(file_relative_folder + params['relative_folder'] + images[params['bootstrap_frames'][1]], cv2.IMREAD_COLOR)

    frame1_gray = cv2.imread(file_relative_folder + params['relative_folder'] + images[params['bootstrap_frames'][0]], cv2.IMREAD_GRAYSCALE)
    frame2_gray = cv2.imread(file_relative_folder + params['relative_folder'] + images[params['bootstrap_frames'][1]], cv2.IMREAD_GRAYSCALE)
    
    # Bootstrap
    db_image, db_keypoints, db_landmarks, R, t = vo_bootstrap(frame2_color, frame1_gray, frame2_gray, K)
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

    # We'll keep track of the camera's *global* 2D positions (X, Z) or (X, Y) over time.
    # For simplicity, assume each new R,t is relative to the initial frame = Identity.
    # We'll accumulate poses and the number of inliers/landmarks
    full_trajectory = []
    landmark_counts = []
    
    # For the live "dashboard", create one figure with 4 subplots
    if plot_dashboard:
        fig, ((ax_img, ax_traj_partial), (ax_landmark_count, ax_traj_full)) = plt.subplots(2,2, figsize=(10,8))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Main loop
    for i in range(params['bootstrap_frames'][1]+1, last_frame):  # short range for demonstration
        # print(f"\nPROCESSING FRAME {i}")
        # cmd = input("Press Enter to continue...")

        if dataset == Dataset.PARKING:
            new_image = cv2.imread(file_relative_folder + "/datasets/parking/images/" + images[i], cv2.IMREAD_COLOR)
            new_image_gray = cv2.imread(file_relative_folder + "/datasets/parking/images/" + images[i], cv2.IMREAD_GRAYSCALE)
        elif dataset == Dataset.MALAGA:
            new_image = cv2.imread(file_relative_folder + "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/" + images[i], cv2.IMREAD_COLOR)
            new_image_gray = cv2.imread(file_relative_folder + "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/" + images[i], cv2.IMREAD_GRAYSCALE)
        elif dataset == Dataset.KITTI:
            new_image = cv2.imread(file_relative_folder + "/datasets/kitti/05/image_0/" + images[i], cv2.IMREAD_COLOR)
            new_image_gray = cv2.imread(file_relative_folder + "/datasets/kitti/05/image_0/" + images[i], cv2.IMREAD_GRAYSCALE)

        # Perform continuous VO step
        state = vo_continuous(new_image, new_image_gray, K, state, min_landmarks=150, min_baseline_angle=params['min_baseline_angle'])

        if plot_dashboard:
            # state['R'], state['t'] is the pose from the previous frame’s coordinate system
            R_new = state['R']
            t_new = state['t']

            t_dashboard = -np.matmul(R_new.T, t_new)
            
            # if len(full_trajectory) > 1:
            #     distances = [np.linalg.norm(np.array(full_trajectory[j]) - np.array(full_trajectory[j-1])) for j in range(1, len(full_trajectory))]
            #     avg_distance = np.mean(distances)
            #     last_point = np.array(full_trajectory[-1])
            #     current_point = np.array([t_dashboard[0], t_dashboard[2]])
            #     if np.linalg.norm(current_point - last_point) < 200 * avg_distance:
            #         full_trajectory.append((t_dashboard[0], t_dashboard[2]))
            #     else:
            #         print("Point not added to trajectory.")
            # else:
            full_trajectory.append((t_dashboard[0], t_dashboard[2]))
                

            # current_frame_color = cv2.imread(new_frame_path, cv2.IMREAD_COLOR)
            current_frame_color = new_image
            
            # Draw keypoints from state['P']
            if state['P'] is not None:
                for pt in state['P']:
                    x, y = pt.ravel()
                    cv2.circle(current_frame_color, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Count how many are in P (tracked db keypoints)
            tracked_landmarks_count = state['P'].shape[0] if state['P'] is not None else 0
            landmark_counts.append(tracked_landmarks_count)

            # Update the dashboard
            update_dashboard(
                ax_img, ax_landmark_count, ax_traj_partial, ax_traj_full,
                current_frame_color,
                db_keypoints=state['P'],
                full_trajectory=full_trajectory,
                landmark_counts=landmark_counts,
                partial_window=20
            )
    # plt.figure(figsize=(8, 6))
    # xs = [point[0] for point in full_trajectory]
    # ys = [point[1] for point in full_trajectory]
    # plt.plot(xs, ys, marker='o', linestyle='-', color='b')
    # plt.title('Full Trajectory')
    # plt.xlabel('X')
    # plt.ylabel('Z')
    # plt.axis('equal')
    # plt.grid(True)
    

    # Finally, show everything at the end (block=True to keep the plots open)
    plt.ioff()
    plt.show()