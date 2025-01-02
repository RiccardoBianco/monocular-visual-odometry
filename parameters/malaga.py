import cv2


malaga_params_map = {
    "relative_folder": "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/",
    "bootstrap_frames": [0, 3], # seems optimal
    "bootstrap_detector": "Shi-Tomasi", # or "Shi-Tomasi"

    # bootstrap parameters (initialization)
    "RANSAC_Essential_Matrix_confidence": 0.99, # Essential matrix Ransac
    "max_corners_bootstrap": 1000, # Harris and Shi-Tomasi
    "quality_level_bootstrap": 0.00001, # Harris and Shi-Tomasi
    "min_distance_bootstrap": 5, # Harris and Shi-Tomasi
    "k_bootstrap": 0.03, # Harris only

    # vo_continuous detector
    "vo_continuous_detector": "Harris", # or "Shi-Tomasi"
    "max_corners_continuous": 1000, # Harris and Shi-Tomasi
    "quality_level_continuous": 0.00001, # Harris and Shi-Tomasi
    "min_distance_continuous": 7, # Harris and Shi-Tomasi
    "k_continuous": 0.05, # Harris only

    # KLT (association)
    "KLT_threshold": 20, # KLT
    "winSize": (30, 30), # KLT
    "maxLevel": 3, # KLT
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001), # KLT

    # PnP (estimate_pose)
    "PnP_min_landmarks": 10, # Minimum number of landmarks to estimate pose
    "RANSAC_PnP_confidence" : 0.9999, # PnP RANSAC
    "PnP_reprojection_error" : 3, # PnP reprojection error
    "PnP_method" : cv2.SOLVEPNP_ITERATIVE, # PnP method, or cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_P3P

    # Triangulate (triangulate_candidates, process_frame)
    "min_baseline_angle": 1, # Minimum bearing angle to allow triangulation
}
