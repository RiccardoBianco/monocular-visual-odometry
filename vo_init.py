import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    E, mask = cv2.findEssentialMat(valid_corners, valid_tracked_corners, K, method=cv2.RANSAC, prob=0.99, threshold=1)
    _, R, t, mask_pose = cv2.recoverPose(E, valid_corners, valid_tracked_corners, K)
    t = -t

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

    camera_center2 = camera_center1 + t.ravel()
    ax.scatter(*camera_center2, c='green', marker='^', s=100, label='Camera 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Return the second frame and the tracked keypoints as initialization
    return frame2, inlier_tracked_corners.reshape(-1,1,2), pts3D, R, t


def vo_continuous(new_frame_path, K, state, min_landmarks=150, min_baseline_angle=1.0):
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
    if X.shape[0] >= 20:
        objectPoints = X.reshape(-1,3)
        imagePoints = P.reshape(-1,2)
        distCoeffs = np.zeros((4,1))
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints, imagePoints, K, distCoeffs,
            reprojectionError=8.0
            # flags=cv2.SOLVEPNP_ITERATIVE
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

    # # 5. Triangulate candidates *individually* if baseline is sufficient
    # # -------------------------------------------------------------------
    # # (Professor's instruction: "only if baseline > 10% of avg scene distance,
    # #  triangulate from that first pose and current pose.")
    # if C.shape[0] > 0:
    #     # Compute average distance to the scene using current inlier X
    #     # Transform them into the current camera frame => (R_new*X_i^T + t_new)
    #     # Then average their z-values (or Euclidean norms, depending on your definition).
    #     if X.shape[0] > 0:
    #         X_in_current = (R_new @ X.T + t_new).T  # shape: (N,3)
    #         # Filter out negative depths, if any
    #         positive_depths = X_in_current[:,2][X_in_current[:,2] > 0]
    #         if len(positive_depths) > 0:
    #             average_distance = np.mean(positive_depths)
    #         else:
    #             average_distance = 1.0  # fallback
    #     else:
    #         average_distance = 1.0  # fallback if no inliers yet

    #     # We'll collect indices of candidates that we manage to triangulate
    #     good_for_triangulation = []
    #     for i in range(C.shape[0]):
    #         # Pose where candidate i was first seen
    #         R_f, t_f = T_first[i]

    #         # Baseline between the two poses
    #         baseline_dist = np.linalg.norm(t_new - t_f)
    #         # Compare with the 10% threshold
    #         if baseline_dist > 0.1 * average_distance:
    #             # OK, let's triangulate it
    #             c_current = C[i].reshape(1, 2).T  # shape (2,1)
    #             c_first   = F_first[i].reshape(1, 2).T

    #             # Camera matrices
    #             P_first   = K @ np.hstack((R_f, t_f))     # shape (3,4)
    #             P_current = K @ np.hstack((R_new, t_new)) # shape (3,4)

    #             pts4D = cv2.triangulatePoints(P_first, P_current, c_first, c_current)
    #             X_new = pts4D[:3] / pts4D[3]  # shape (3,)

    #             # If we want the 3D point in the "bootstrap/world" frame:
    #             # By default, triangulatePoints gives coords in that first camera's
    #             # reference if you used [I|0], [R|t]. But here you're using "R_f, t_f".
    #             # So X_new is in the coordinate system of the *first camera that saw it*.
    #             # If your "bootstrap" camera is the global reference, then R_f,t_f is
    #             # from world->camera, so to get world coords you do inverse transform:
    #             X_world = R_f.T @ (X_new - t_f)
    #             # But: *exact transformation depends on your chosen reference frames*.
    #             #
    #             # For simplicity, let’s store X_new in that first camera's reference
    #             # or transform to "current frame." That depends on your pipeline.
    #             # Example: let's transform it to "world" if your first camera is identity:
    #             # X_world = R_f.T.dot(X_new) - R_f.T.dot(t_f)

    #             # Add to P, X
    #             # We'll store the "current" 2D observation in P, so that the solver
    #             # can incorporate it next iteration
    #             new_pt = C[i].reshape(1,1,2)
    #             P = np.vstack((P, new_pt))
    #             # And store the 3D coordinate in X
    #             X = np.vstack((X, X_world.reshape(1,3)))

    #             # Mark this candidate as triangulated
    #             good_for_triangulation.append(i)

    #     # Remove triangulated candidates from C, F_first, T_first
    #     if len(good_for_triangulation) > 0:
    #         mask_keep = np.ones(C.shape[0], dtype=bool)
    #         mask_keep[good_for_triangulation] = False
    #         C = C[mask_keep]
    #         F_first = F_first[mask_keep]
    #         T_first = [T_first[j] for j in range(len(T_first)) if mask_keep[j]]

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
        Axes object for the last 'partial_window' frames’ trajectory.
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

    # 2) # tracked landmarks over last partial_window frames
    ax_landmark_count.clear()
    ax_landmark_count.plot(
        range(max(0, len(landmark_counts) - partial_window), len(landmark_counts)),
        landmark_counts[-partial_window:] if len(landmark_counts) > partial_window else landmark_counts,
        marker='o'
    )
    ax_landmark_count.set_title(f"# tracked landmarks (last {partial_window} frames)")

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

    # 4) Full trajectory
    ax_trajectory_full.clear()
    if len(full_trajectory) > 0:
        xs_full = [p[0] for p in full_trajectory]
        ys_full = [p[1] for p in full_trajectory]
        ax_trajectory_full.plot(xs_full, ys_full, 'b-')
        ax_trajectory_full.set_title("Full trajectory")
    ax_trajectory_full.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # short pause to allow the figure to update

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

    # We'll keep track of the camera's *global* 2D positions (X, Z) or (X, Y) over time.
    # For simplicity, assume each new R,t is relative to the initial frame = Identity.
    # We'll accumulate poses and the number of inliers/landmarks
    full_trajectory = []
    landmark_counts = []

    # For the live "dashboard", create one figure with 4 subplots
    fig, ((ax_img, ax_traj_partial), (ax_landmark_count, ax_traj_full)) = plt.subplots(2,2, figsize=(10,8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Main loop
    for i in range(4, 598):  # short range for demonstration
        if i < 10:
            frame_2_relative_folder = f"/datasets/parking/images/img_0000{i}.png"
        elif i < 100:
            frame_2_relative_folder = f"/datasets/parking/images/img_000{i}.png"
        else:
            frame_2_relative_folder = f"/datasets/parking/images/img_00{i}.png"

        new_frame_path = os.path.join(os.path.dirname(__file__) + frame_2_relative_folder)

        # Perform continuous VO step
        state = vo_continuous(new_frame_path, K, state, min_landmarks=150, min_baseline_angle=10.0)

        # state['R'], state['t'] is the pose from the previous frame’s coordinate system
        R_new = state['R']
        t_new = state['t']

        full_trajectory.append((-t_new[0], -t_new[2]))  ### HERE is with the -!!!

        current_frame_color = cv2.imread(new_frame_path, cv2.IMREAD_COLOR)
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