# VAMR Visual Odometry Pipeline

This repository contains the final implementation of our Visual Odometry (VO) pipeline for the VAMR Mini-Project, including parameter tuning for different datasets.

## Setup Instructions

1. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate VAMR_env
   ```

## Running the Test

1. To specify the dataset, modify the following line in the script:
   ```python
   dataset = Dataset.MALAGA  # or Dataset.PARKING or Dataset.KITTI or Dataset.MALAGA_ROUNDABOUT
   ```

2. Run the main script:
   ```bash
   python3 vo_init.py
   ```

## Authors

- **Riccardo Bianco:** 
- **Davide Cannone:** 
- **Tommaso Gazzini:** 
- **Andrea Zannini:** 
