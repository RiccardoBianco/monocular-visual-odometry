import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_dashboard(state, camera_positions, landmark_counts, frame_path):
    """
    Display the dashboard with multiple visual odometry outputs.
    """
    current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract values for plots
    trajectory = np.array(camera_positions)
    landmarks_3d = state['X']
    keypoints = state['P']
    num_landmarks = landmark_counts

    # Initialize dashboard layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.clf()  # Clear previous plots

    # 1. Current Frame with Keypoints
    axes[0, 0].imshow(current_frame, cmap='gray')
    if keypoints is not None and keypoints.shape[0] > 0:
        axes[0, 0].scatter(keypoints[:, 0, 0], keypoints[:, 0, 1], c='green', s=10)
    axes[0, 0].set_title("Current Frame with Keypoints")
    axes[0, 0].axis("off")

    # 2. Camera Trajectory - Last 20 Frames
    if trajectory.shape[0] > 0:
        axes[0, 1].plot(trajectory[-20:, 0], trajectory[-20:, 2], '-o', markersize=4)
    axes[0, 1].set_title("Trajectory - Last 20 Frames")
    axes[0, 1].set_aspect('equal')

    # 3. Full Camera Trajectory
    if trajectory.shape[0] > 0:
        axes[0, 2].plot(trajectory[:, 0], trajectory[:, 2], '-o', markersize=2)
    axes[0, 2].set_title("Full Trajectory")
    axes[0, 2].set_aspect('equal')

    # 4. Tracked Landmarks in 3D
    if landmarks_3d.shape[0] > 0:
        ax3d = fig.add_subplot(2, 2, 3, projection='3d')
        ax3d.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], c='red', s=5)
        ax3d.set_title("Tracked Landmarks in 3D")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

    # 5. Number of Landmarks Over Time
    axes[1, 2].plot(range(-len(num_landmarks), 0), num_landmarks, '-o')
    axes[1, 2].set_title("Number of Tracked Landmarks")
    axes[1, 2].set_xlabel("Frame (last N)")
    axes[1, 2].set_ylabel("Count")

    plt.tight_layout()
    plt.pause(0.001)  # Pause to update the plot
def create_dashboard():
    """
    Initializes the fixed dashboard with predefined subplots.
    Returns:
        fig: Matplotlib figure object
        axes: Dictionary of subplot axes for easy access and updates
    """
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Visual Odometry Dashboard", fontsize=16)

    # Define the axes (subplots) in the layout
    axes = {
        'frame_keypoints': fig.add_subplot(2, 2, 1),
        'trajectory_last20': fig.add_subplot(2, 2, 2),
        'trajectory_full': fig.add_subplot(2, 2, 3),
        'landmarks_3d': fig.add_subplot(2, 2, 4, projection='3d')
    }

    # Titles for the subplots
    axes['frame_keypoints'].set_title("Current Frame with Keypoints")
    axes['trajectory_last20'].set_title("Trajectory - Last 20 Frames")
    axes['trajectory_full'].set_title("Full Trajectory")
    axes['landmarks_3d'].set_title("Tracked Landmarks in 3D")

    # Turn off axis for keypoints subplot (grayscale image)
    axes['frame_keypoints'].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust title spacing

    return fig, axes


def update_dashboard(fig, axes, state, camera_positions, frame_path):
    """
    Updates the dashboard with new data.
    """
    current_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    trajectory = np.array(camera_positions)
    landmarks_3d = state['X']
    keypoints = state['P']

    # Clear and update individual axes
    axes['frame_keypoints'].cla()
    axes['frame_keypoints'].imshow(current_frame, cmap='gray')
    if keypoints is not None and keypoints.shape[0] > 0:
        axes['frame_keypoints'].scatter(keypoints[:, 0, 0], keypoints[:, 0, 1], c='green', s=10)
    axes['frame_keypoints'].set_title("Current Frame with Keypoints")
    axes['frame_keypoints'].axis("off")

    axes['trajectory_last20'].cla()
    if trajectory.shape[0] > 0:
        axes['trajectory_last20'].plot(trajectory[-20:, 0], trajectory[-20:, 2], '-o', markersize=4)
    axes['trajectory_last20'].set_title("Trajectory - Last 20 Frames")
    axes['trajectory_last20'].set_aspect('equal')

    axes['trajectory_full'].cla()
    if trajectory.shape[0] > 0:
        axes['trajectory_full'].plot(trajectory[:, 0], trajectory[:, 2], '-o', markersize=2)
    axes['trajectory_full'].set_title("Full Trajectory")
    axes['trajectory_full'].set_aspect('equal')

    axes['landmarks_3d'].cla()
    if landmarks_3d.shape[0] > 0:
        axes['landmarks_3d'].scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], c='red', s=5)
    axes['landmarks_3d'].set_title("Tracked Landmarks in 3D")
    axes['landmarks_3d'].set_xlabel("X")
    axes['landmarks_3d'].set_ylabel("Y")
    axes['landmarks_3d'].set_zlabel("Z")

    # Refresh the canvas
    fig.canvas.draw()
    plt.pause(0.001)
