import matplotlib.pyplot as plt
import os

def read_poses(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            x, y, z = values[3], values[7], values[11]
            poses.append((x, y, z))
    return poses

def plot_poses(poses):
    x_vals = [pose[0] for pose in poses]
    y_vals = [pose[1] for pose in poses]
    z_vals = [pose[2] for pose in poses]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def read_gps_poses(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('%'):
                continue  # Skip header lines
            values = list(map(float, line.strip().split()))
            local_x, local_y, local_z = values[8], values[9], values[10]
            poses.append((local_x, local_y, local_z))
    return poses

    x_vals = [pose[0] for pose in poses]
    y_vals = [pose[1] for pose in poses]
    z_vals = [pose[2] for pose in poses]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, marker='o')
    ax.set_xlabel('Local X')
    ax.set_ylabel('Local Y')
    ax.set_zlabel('Local Z')
    plt.show()

def plot(dataset):
    parking_poses_dir = "/datasets/parking/poses.txt"
    kitti_poses_dir = "/datasets/kitti/poses/10.txt" # <-- probably 00.txt is the right file for us
    malaga_poses_dir = "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_all-sensors_GPS.txt"

    parking_poses_path = os.path.join(os.path.dirname(__file__) + parking_poses_dir)
    kitti_poses_path = os.path.join(os.path.dirname(__file__) + kitti_poses_dir)
    malaga_poses_path = os.path.join(os.path.dirname(__file__) + malaga_poses_dir)

    if dataset == 1:
        poses = read_poses(parking_poses_path)
    elif dataset == 2:
        poses = read_poses(kitti_poses_path)
    elif dataset == 3:
        poses = read_gps_poses(malaga_poses_path)
    else:
        raise ValueError("Invalid dataset number")
    
    plot_poses(poses)

if __name__ == "__main__":
    # 1. parking
    # 2. kitti
    # 3. malaga (weird GPS file)
    plot(3)
    