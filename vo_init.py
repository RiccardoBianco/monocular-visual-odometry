import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from params_loader import load_parameters, Dataset

# Set the dataset
dataset = Dataset.MALAGA  

### Plotting options - eventually change these #######
plot_bootstrap = False
plot_dashboard = True
plot_ground_truth = False
plot_vo_continuous_inliers_outliers = False

# Load parameters
params = load_parameters(dataset)

file_relative_folder = os.path.dirname(__file__)

state = {
    'P': None,       # (2xN) - keypoints
    'X': None,       # (3xN) - landmarks
    'R': None,       # (3x3) - rotation
    't': None,       # (3x1) - translation,
    'C': None,       # (2xM) - candidate keypoints
    'F': None,       # (2xM) - first observation of candidate keypoints
    'T': None,       # (12xM) - poses for first observation of each candidate
}

lk_params = dict(winSize=params['winSize'],
        maxLevel=params['maxLevel'],
        criteria=params['criteria'])

def vo_bootstrap(frame2_color, frame1, frame2, K):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # Match keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) # TODO match_per_descriptors

    good_matches = []
    for m, n in matches:
        if m.distance < params['ratio_test'] * n.distance:
            good_matches.append(m)

    # Get the keypoints from the good matches
    keypoints_frame_1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    keypoints_frame_2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(keypoints_frame_1, keypoints_frame_2, K, method=cv2.RANSAC, prob=params['RANSAC_Essential_Matrix_confidence'], threshold=1) # TODO check parameters

    inliers_frame_1 = keypoints_frame_1[mask.ravel() == 1]
    inliers_frame_2 = keypoints_frame_2[mask.ravel() == 1]
    outliers_frame_1 = keypoints_frame_1[mask.ravel() == 0]
    outliers_frame_2 = keypoints_frame_2[mask.ravel() == 0]

    # TODO dashboard
    
    # Visualize inliers and outliers
    for (new, old) in zip(inliers_frame_2, inliers_frame_1):
        a, b = new.ravel()
        c, d = old.ravel()
        frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 255, 0), -1)  # Green for inliers
        frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

    for (new, old) in zip(outliers_frame_2, outliers_frame_1):
        a, b = new.ravel()
        c, d = old.ravel()
        frame2_color = cv2.circle(frame2_color, (int(a), int(b)), 5, (0, 0, 255), -1)  # Red for outliers
        frame2_color = cv2.line(frame2_color, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
    
    _, R, t, _ = cv2.recoverPose(E, keypoints_frame_1, keypoints_frame_2, K)

    # Triangulate 3D points
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))  # First frame pose
    P2 = K @ np.hstack((R, t))  # Second frame pose

    pts4D = cv2.triangulatePoints(P1, P2, keypoints_frame_1.T, keypoints_frame_2.T)
    pts3D = pts4D[:3, :] / pts4D[3, :]  # shape (3, N)

    # TODO filter landmark behind the camera if Z < 0 for some reason

    # Update state
    state["P"] = keypoints_frame_2.T 
    state["X"] = pts3D
    state["R"] = R
    state["t"] = t

    return


def vo_continuous(new_frame_color, new_frame, prev_frame, K, state, min_landmarks=150, min_baseline_angle=5.0):

    P_prev = state['P']
    X_prev = state['X']
    C_prev = state['C']

    R_prev = state['R']
    t_prev = state['t']
    
    # 1. Track keypoints
    tracked_keypoints, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, new_frame, P_prev, nextPts=None, **lk_params)
    if error is not None:
        mask_kp = error < params['KLT_threshold']
        final_mask_kp = np.logical_and(mask_kp, status)
    else:
        final_mask_kp = status

    P_tracked = tracked_keypoints[final_mask_kp.flatten() == 1]
    X_tracked = X_prev[final_mask_kp.flatten() == 1] # TODO check dimensions

    # 2. Track candidate keypoints
    if C_prev is not None:
        C_tracked, status_c, error_c = cv2.calcOpticalFlowPyrLK(prev_frame, new_frame, C, nextPts=None, **lk_params)
        if error is not None:
            mask_c = error_c < params['KLT_threshold']
            final_mask_c = np.logical_and(mask_c, status_c)
        else:
            final_mask_c = status_c
        if C_tracked is not None:
            state["C"] = C_tracked[:, final_mask_c == 1] # TODO controllare che C_tracked restituito sia un 2xM
        


    # 3. Pose estimation with PnP
    distCoeffs = np.zeros((4,1)) # TODO oppure none al posto di questo parametro
    if params['PnP_method'] == cv2.SOLVEPNP_ITERATIVE:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            X_tracked, P_tracked, K, distCoeffs,
            reprojectionError=params['PnP_reprojection_error'],
            flags=params['PnP_method'],
            confidence=params['RANSAC_PnP_confidence'],
        )
    else:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            X_tracked, P_tracked, K, distCoeffs,
            flags=params['PnP_method'],
            confidence=params['RANSAC_PnP_confidence'],
        )
    if success:
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec # TODO tvec.flatten()
        
        if inliers is not None:
            inlier_mask = inliers.flatten() 
            
            # Visualization: Inliers and outliers # TODO se vogliamo aggiungere qualcosa qua
            inliers_keypoints = P_tracked[inlier_mask]
            inliers_landmarks = X_tracked[inlier_mask]

            outliers_keypoints = P_tracked[~inlier_mask]
            outliers_landmarks = X_tracked[~inlier_mask]

            P_tracked = P_tracked[inlier_mask]
            X_tracked = X_tracked[inlier_mask]
    else:
        print("PnP failed...keeping previous pose.")

    inlier_ration = P_tracked.shape[0] / P_prev.shape[0] # TODO non fanno esattamente così ma dovrebbero fare così
    max_inlier_ratio = 0.8 # TODO messo a caso
    # TODO inliers ration
    if X_tracked.shape[0] < min_landmarks or inlier_ration < max_inlier_ratio:
        triangulate = True 

    state["X"] = X_tracked
    state["P"] = P_tracked

    # TODO visualization --> non so come gestirlo
    '''
    # Visualize db_keypoints inliers and outliers
    if pnp_success: 
        for (new, old) in zip(inlier_tracked_points, inlier_points):
            a, b = new.ravel()
            c, d = old.ravel()
            # new_frame_color = cv2.circle(new_frame_color, (int(a), int(b)), 5, (0, 255, 0), -1)  # Green for inliers
            new_frame_color = cv2.line(new_frame_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)

        for (new, old) in zip(outlier_tracked_points, outlier_points):
            a, b = new.ravel()
            c, d = old.ravel()
            # new_frame_color = cv2.circle(new_frame_color, (int(a), int(b)), 5, (0, 0, 255), -1)  # Red for outliers
            new_frame_color = cv2.line(new_frame_color, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
    '''

    # 4. Add new candidate keypoints
    if params['vo_continuous_detector'] == 'Shi-Tomasi':
        new_corners = cv2.goodFeaturesToTrack(new_frame, 
                                              maxCorners=params['max_corners_continuous'], 
                                              qualityLevel=params['quality_level_continuous'], 
                                              minDistance=params['min_distance_continuous'])
    elif params['vo_continuous_detector'] == 'Harris':
        new_corners = cv2.goodFeaturesToTrack(new_frame, 
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

    return state, pnp_success

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

def get_dataset(dataset):
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
    
    return images, ground_truth, last_frame, K

def load_new_image(file_relative_folder, images):
    if dataset == Dataset.PARKING:
        new_image = cv2.imread(file_relative_folder + "/datasets/parking/images/" + images[i], cv2.IMREAD_COLOR)
        new_image_gray = cv2.imread(file_relative_folder + "/datasets/parking/images/" + images[i], cv2.IMREAD_GRAYSCALE)
    elif dataset == Dataset.MALAGA:
        new_image = cv2.imread(file_relative_folder + "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/" + images[i], cv2.IMREAD_COLOR)
        new_image_gray = cv2.imread(file_relative_folder + "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/" + images[i], cv2.IMREAD_GRAYSCALE)
    elif dataset == Dataset.KITTI:
        new_image = cv2.imread(file_relative_folder + "/datasets/kitti/05/image_0/" + images[i], cv2.IMREAD_COLOR)
        new_image_gray = cv2.imread(file_relative_folder + "/datasets/kitti/05/image_0/" + images[i], cv2.IMREAD_GRAYSCALE)
    return new_image, new_image_gray


if __name__ == "__main__":
    images, ground_truth, last_frame, K = get_dataset(dataset)
    
    frame2_color = cv2.imread(file_relative_folder + params['relative_folder'] + images[params['bootstrap_frames'][1]], cv2.IMREAD_COLOR)

    frame1_gray = cv2.imread(file_relative_folder + params['relative_folder'] + images[params['bootstrap_frames'][0]], cv2.IMREAD_GRAYSCALE)
    frame2_gray = cv2.imread(file_relative_folder + params['relative_folder'] + images[params['bootstrap_frames'][1]], cv2.IMREAD_GRAYSCALE)
    
    # Bootstrap 
    vo_bootstrap(frame2_color, frame1_gray, frame2_gray, K)

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
    prev_image, prev_image_gray = frame2_color, frame2_gray
    for i in range(params['bootstrap_frames'][1]+1, last_frame): 
        new_image, new_image_gray = load_new_image(file_relative_folder, images)

        # Perform continuous VO step
        state, pnp_success = vo_continuous(new_image, new_image_gray, prev_image_gray, K, state, min_landmarks=150, min_baseline_angle=params['min_baseline_angle'])
        old_image_gray = new_image_gray
        # TODO dashboard
        if plot_dashboard:
            # state['R'], state['t'] is the pose from the previous frame’s coordinate system
            R_new = state['R']
            t_new = state['t']

            t_dashboard = -np.matmul(R_new.T, t_new)

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

    # Finally, show everything at the end (block=True to keep the plots open)
    plt.ioff()
    plt.show()