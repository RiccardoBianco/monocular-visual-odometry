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
    "R": None,
    'C': None,       # (2xM) - candidate keypoints
    'F': None,       # (2xM) - first observation of candidate keypoints
    'T': None,       # (12xM) - poses for first observation of each candidate
}

lk_params = dict(winSize=params['winSize'],
        maxLevel=params['maxLevel'],
        criteria=params['criteria'])

def vo_bootstrap(frame2_color, frame1, frame2, K):
    # cv2.imshow("frame2_color", frame2_color)
    # cv2.waitKey(0)
    # cv2.imshow("frame1_gray", frame1)
    # cv2.waitKey(0)
    # cv2.imshow("frame2_gray", frame2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # Match keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) 

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # Get the keypoints from the good matches
    keypoints_frame_1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    keypoints_frame_2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(keypoints_frame_1, keypoints_frame_2, K, method=cv2.FM_RANSAC, prob=params['RANSAC_Essential_Matrix_confidence'], threshold=0.9) # TODO check parameters

    inliers_frame_1 = keypoints_frame_1[mask.ravel() == 1]
    inliers_frame_2 = keypoints_frame_2[mask.ravel() == 1]
    '''
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
    '''
    _, R, t, _ = cv2.recoverPose(E, inliers_frame_1, inliers_frame_2, K)

    # Triangulate 3D points
    #P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))  # First frame pose
    P1 = K @ np.eye(3,4)  # Second frame pose
    P2 = K @ np.hstack((R, t))  # Second frame pose

    pts4D = cv2.triangulatePoints(P1, P2, inliers_frame_1.T, inliers_frame_2.T)
    pts3D = pts4D[:3, :] / pts4D[-1, :]  # shape (3, N)

    # TODO filter landmark behind the camera if Z < 0 for some reason

    # Update state
    state["P"] = inliers_frame_2.T 
    state["X"] = pts3D

    return


def vo_continuous(new_frame_color, new_frame, prev_frame, K, state, min_landmarks):

    P_prev = state['P']
    X_prev = state['X']
    C_prev = state['C']

    triangulate = False 
    
    # 1. Track keypoints
    tracked_keypoints, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, new_frame, P_prev.T, nextPts=None, **lk_params)
    if error is not None:
        mask_kp = error < params['KLT_threshold']
        final_mask_kp = np.logical_and(mask_kp, status)
    else:
        final_mask_kp = status

    P_tracked = tracked_keypoints[final_mask_kp.flatten() == 1]
    X_tracked = X_prev[:, final_mask_kp.flatten() == 1].T

    # 2. Track candidate keypoints
    if C_prev is not None:
        C_tracked, status_c, error_c = cv2.calcOpticalFlowPyrLK(prev_frame, new_frame, state["C"].T.astype(np.float32), nextPts=None, **lk_params)
        if error_c is not None:
            mask_c = error_c < params['KLT_threshold']
            final_mask_c = np.logical_and(mask_c, status_c)
        else:
            final_mask_c = status_c
        if C_tracked is not None:
            C_tracked = C_tracked.T
            state["C"] = C_tracked[:, final_mask_c.flatten() == 1] # TODO controllare che C_tracked restituito sia un 2xM
            state["F"] = state["F"][:, final_mask_c.flatten() == 1]
            state["T"] = state["T"][:, final_mask_c.flatten() == 1]
        


    # 3. Pose estimation with PnP
    if params['PnP_method'] == cv2.SOLVEPNP_ITERATIVE:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            X_tracked, P_tracked, K, None,
            reprojectionError=params['PnP_reprojection_error'],
            flags=params['PnP_method'],
            confidence=params['RANSAC_PnP_confidence'],
        )
    else:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            X_tracked, P_tracked, K, None,
            flags=params['PnP_method'],
            confidence=params['RANSAC_PnP_confidence'],
        )
    if success:
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec.flatten()
        
        if inliers is not None:
            inlier_mask = inliers.ravel() 
            '''
            # Visualization: Inliers and outliers # TODO se vogliamo aggiungere qualcosa qua
            inliers_keypoints = P_tracked[inlier_mask]
            inliers_landmarks = X_tracked[inlier_mask]

            outliers_keypoints = P_tracked[~inlier_mask]
            outliers_landmarks = X_tracked[~inlier_mask]
            '''

            P_tracked = P_tracked[inlier_mask]
            X_tracked = X_tracked[inlier_mask]
    else:
        print("PnP failed...keeping previous pose.")

    inlier_ratio = P_tracked.shape[0] / P_prev.shape[1] 
    max_inlier_ratio = 0.3 
    if X_tracked.shape[0] < min_landmarks or inlier_ratio < max_inlier_ratio:
        triangulate = True 

    state["X"] = X_tracked.T 
    state["P"] = P_tracked.T
    

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
    # TODO: delete this eventually
    # if params['vo_continuous_detector'] == 'Shi-Tomasi':
    #     new_corners = cv2.goodFeaturesToTrack(new_frame, 
    #                                           maxCorners=params['max_corners_continuous'], 
    #                                           qualityLevel=params['quality_level_continuous'], 
    #                                           minDistance=params['min_distance_continuous'])
    if params['vo_continuous_detector'] == 'Harris':
        new_corners = cv2.cornerHarris(new_frame, 
                                       blockSize=params['block_size_continuous'], 
                                       ksize=params['ksize_continuous'], 
                                       k=params['k_continuous'])

    # select keypoints, TODO: check if it's really needed, otherwise delete
    r = params['radius_non_maxima_suppression']
    keypoints = np.zeros((params['max_corners_continuous'], 2))
    temp_scores = np.pad(new_corners, [(r, r), (r, r)], mode='constant', constant_values=0)
    for i in range(params['max_corners_continuous']):
        kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)
        keypoints[i, :] = np.array(kp)[::-1] - r # reverse the order and remove padding
        temp_scores[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)] = 0
        
    distances = np.linalg.norm(keypoints[:,None,:] - P_tracked, axis=2) # TODO tolto il trasposto
    is_close = np.any(distances < params["dist_treshold"], axis=1) # TODO settare il parametro correttamente
    keypoints_filtered = keypoints[~is_close]
    
    if state["C"] is None:
        state["C"] = keypoints_filtered.T
    else:
        state["C"] = np.concatenate((state["C"], keypoints_filtered.T), axis=1)

    if state["F"] is None:
        state["F"] = keypoints_filtered.T
    else:
        state["F"] = np.concatenate((state["F"], keypoints_filtered.T), axis=1)

    Tvec = np.hstack((R_new, t_new[:, None])).flatten()
    N = keypoints_filtered.shape[0]

    if state["T"] is None:
        state["T"] = np.tile(Tvec, (N, 1)).T
    else:
        state["T"] = np.concatenate((state["T"], np.tile(Tvec, (N, 1)).T), axis=-1)

    # T. Triangulate points
    # state, Rnew, tnew, K, triangulate
    current_pose = np.hstack((R_new, t_new[:, None]))
    poses_reshaped = state["T"].reshape(3, 4, state['C'].shape[1])
    T_reshaped = poses_reshaped[:,-1,:]
    
    distances = np.linalg.norm(T_reshaped - t_new[:, None], axis=0)

    # TODO change name of the following part fino alla fine
    avg_depth = np.mean((R_new @ state['X'] + t_new.reshape((t_new.shape[0], 1)))[2, :]) 

    thumb_mask = distances / avg_depth > params['thumb_rule']
    dist_mask = distances > params['distance_threshold']
    mask = np.logical_or(thumb_mask, dist_mask)

    possible_new_landmarks = np.sum(mask)

    if possible_new_landmarks == 0 and triangulate:
        mask = distances >= np.max(distances)

    if possible_new_landmarks > 0 or triangulate:
        prev_poses = state["T"][:,mask]
        
        unique_poses = np.unique(prev_poses, axis = 1)
        new_landmarks = None

        filter_candidates_mask = np.ones(state["T"].shape[1], dtype=bool)
        filter_keypoints_mask = np.ones(state["T"].shape[1], dtype=bool)
    
        for pose in unique_poses.T:
            indices = np.where(np.all(state["T"].T == pose, axis = 1))[0]

            M1 = pose.reshape(3,4)
            M2 = current_pose

            selected_candidates = state["C"][:, indices].astype(np.float32)
            selected_first_obs_candidates = state["F"][:, indices].astype(np.float32)
        
            points_3d_homogeneous = cv2.triangulatePoints(K @ M1, K @ M2, selected_first_obs_candidates, selected_candidates)
            points_3d = points_3d_homogeneous[:3,:] / points_3d_homogeneous[-1, :]


            # filter triangulated points
            distance_threshold_factor = params['distance_threshold_factor']
            min_reprojection_error = params['min_reprojection_error']
            
            projected_new = cv2.projectPoints(points_3d, M1[:, :3], M1[:, 3:].flatten(), K, None)[0].reshape(-1, 2)
            projected_first = cv2.projectPoints(points_3d, M2[:, :3], M2[:, 3:].flatten(), K, None)[0].reshape(-1, 2)
            reprojection_errors = (np.linalg.norm(selected_first_obs_candidates.T - projected_new, axis=1) + np.linalg.norm(selected_candidates.T - projected_first, axis=1)) / 2
            
            points_3d_camera_frame = R_new @ points_3d + t_new[:, None]
            
            threshold_x =  abs(np.mean(points_3d_camera_frame[0, :])) + distance_threshold_factor * abs(np.std(points_3d_camera_frame[0, :]))
            threshold_z = abs(np.mean(points_3d_camera_frame[2, :])) + distance_threshold_factor * abs(np.std(points_3d_camera_frame[2, :]))

            
            valid_landmark_mask = \
                (points_3d_camera_frame[0] < threshold_x) & \
                (points_3d_camera_frame[2] > 0) & \
                (points_3d_camera_frame[2] < np.min((threshold_z, 100))) & \
                (reprojection_errors < min_reprojection_error) \
                        
            points_3d = points_3d[:,valid_landmark_mask]

            if new_landmarks is None:
                new_landmarks = points_3d
            else:
                new_landmarks = np.concatenate((new_landmarks, points_3d), axis=1)

            filter_candidates_mask[indices] = False
            filter_keypoints_mask[indices[valid_landmark_mask]] = False

            # update X, P, C, F, T accordingly
            if new_landmarks is not None:
                state["X"] = np.concatenate((state["X"], new_landmarks), axis=1)
                state["P"] = np.concatenate((state["P"], state["C"][:, ~filter_keypoints_mask].astype(np.float32)), axis=1)

                state["C"] = state["C"][:, filter_candidates_mask == 1]
                state["F"] = state["F"][:, filter_candidates_mask == 1]
                state["T"] = state["T"][:, filter_candidates_mask == 1]
    
    # Visualize the frame with tracked keypoints
    if plot_vo_continuous_inliers_outliers:
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2RGB))
        plt.title('Inliers (Green) and Outliers (Red)')
        plt.show()

    
    return state, R_new, t_new

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
        return images, ground_truth, last_frame, K
    
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

    # MALAGA case
    return images, None, last_frame, K

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
    
    # cv2.imshow("frame2_color", frame2_color)
    # cv2.waitKey(0)
    # cv2.imshow("frame1_gray", frame1_gray)
    # cv2.waitKey(0)
    # cv2.imshow("frame2_gray", frame2_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Bootstrap 
    vo_bootstrap(frame2_color, frame1_gray, frame2_gray, K)

    # We'll keep track of the camera's *global* 2D positions (X, Z) or (X, Y) over time.
    # For simplicity, assume each new R,t is relative to the initial frame = Identity.
    # We'll accumulate poses and the number of inliers/landmarks
    full_trajectory = []
    landmark_counts = []
    
    # For the live "dashboard", create one figure with 4 subplots
    # if plot_dashboard:
    #     fig, ((ax_img, ax_traj_partial), (ax_landmark_count, ax_traj_full)) = plt.subplots(2,2, figsize=(10,8))
    #     plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Main loop
    prev_image, prev_image_gray = frame2_color, frame2_gray
    for i in range(params['bootstrap_frames'][1]+1, last_frame): 
        new_image, new_image_gray = load_new_image(file_relative_folder, images)

        # Perform continuous VO step
        state, R_new, t_new = vo_continuous(new_image, new_image_gray, prev_image_gray, K, state, min_landmarks=120)
        old_image_gray = new_image_gray
        # TODO dashboard
        if plot_dashboard:
            t_dashboard = -np.matmul(R_new.T, t_new)

            full_trajectory.append((t_dashboard[0], t_dashboard[2]))
                
            # current_frame_color = cv2.imread(new_frame_path, cv2.IMREAD_COLOR)
            current_frame_color = new_image
            
            # Draw keypoints from state['P']
            if state['P'] is not None:
                for pt in state['P'].T:
                    x, y = pt.ravel()
                    cv2.circle(current_frame_color, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Count how many are in P (tracked db keypoints)
            tracked_landmarks_count = state['P'].shape[0] if state['P'] is not None else 0
            landmark_counts.append(tracked_landmarks_count)

            # Update the dashboard
            # update_dashboard(
            #     ax_img, ax_landmark_count, ax_traj_partial, ax_traj_full,
            #     current_frame_color,
            #     db_keypoints=state['P'],
            #     full_trajectory=full_trajectory,
            #     landmark_counts=landmark_counts,
            #     partial_window=20
            # )

    # Finally, show everything at the end (block=True to keep the plots open)
    plt.ioff()
    plt.show()