import cv2

import cv2

# TUNED
# there's another working well version with 1000 corners and 0.99 of RANSAC confidence
# Up to us to check which one is better (memory vs computational efficiency)
malaga_params_map = {
    "relative_folder": "/datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/",
    "bootstrap_frames": [0, 3], # seems optimal

    # bootstrap parameters (initialization)
    "RANSAC_Essential_Matrix_confidence": 0.9999, # Essential matrix Ransac

    # vo_continuous detector
    "vo_continuous_detector": "Harris", # or "Shi-Tomasi"
    "max_corners_continuous": 300, # Harris and Shi-Tomasi
    "min_distance_continuous": 8, # Harris and Shi-Tomasi # TODO aggiungere al posto di dist_treshold
    "k_continuous": 0.04, # Harris only
    "block_size_continuous" : 7, # Harris only
    "ksize_continuous" : 5, # Harris only
    'radius_non_maxima_suppression': 7, # Harris only

    "dist_treshold": 8, # Distance threshold for feature matching


    
    # KLT (association)
    "KLT_threshold": 10, # KLT
    "winSize": (30, 30), # KLT
    "maxLevel": 3, # KLT
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001), # KLT

    # PnP (estimate_pose)
    "RANSAC_PnP_confidence" : 0.999, # PnP RANSAC
    "PnP_reprojection_error" : 1, # PnP reprojection error
    "PnP_method" : cv2.SOLVEPNP_ITERATIVE, # PnP method, or cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_P3P

    # Triangulate (triangulate_candidates, process_frame)
    'thumb_rule': 0.2, # Thumb rule for triangulation
    'distance_threshold': 1, # Distance threshold for triangulation

    'distance_threshold_factor': 2, # Distance threshold factor for triangulation
    'min_reprojection_error': 1, # Minimum reprojection error for triangulation
}

