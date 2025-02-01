# VAMR Visual Odometry Pipeline

This repository contains the final implementation of our Visual Odometry (VO) pipeline for the VAMR Mini-Project, including parameter tuning for different datasets.

---

## **Setup Instructions**

### 1. Download Dataset

Download the dataset from the VAMR website or your preferred source.

### 2. Organize Dataset

Place the downloaded dataset in the `datasets` directory at the root of the project.

### 3. Create Virtual Environment

#### 3.1 Conda

To create a virtual environment for the project with Conda, do the following in the root of this project:

1. Create the environment:
   ```bash
   conda env create --file=environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate VAMR_env
   ```

The command above will create the virtual environment and install all the necessary libraries.

---

### 4. Run the Code

You can execute the code by typing:

```bash
python3 vo_init.py
```

---

### 5. Change Dataset

To change the dataset or any visualization options, modify the following lines in the `vo_init.py` file:

```python
# SELECT DATASET
dataset = Dataset.PARKING  # or Dataset.MALAGA or Dataset.KITTI or Dataset.MALAGA_ROUNDABOUT

```

---

## **Authors**

- **Riccardo Bianco**: Continuous Operation, Debugging, Motion Estimation, Parameter Tuning
- **Davide Cannone**: Initialization, Continuous Operation, Debugging, Report Writing
- **Tommaso Gazzini**: 3D-3D Correspondences Problem, Visualization, Debugging, Report
- **Andrea Zannini**: Initialization, Continuous Operation, Debugging, Parameters Tuning
