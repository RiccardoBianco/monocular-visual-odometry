from parameters.kitti import kitti_params_map
from parameters.malaga import malaga_params_map
from parameters.parking import parking_params_map
from parameters.malaga_roundabout import parking_params_map_roundabout

from enum import Enum

class Dataset(Enum):
  KITTI = 0
  MALAGA = 1
  PARKING = 2
  MALAGA_ROUNDABOUT = 3


def load_parameters(dataset: Dataset):
  if dataset == Dataset.KITTI:
    params = kitti_params_map
  if dataset == Dataset.MALAGA:
    params = malaga_params_map
  if dataset == Dataset.PARKING:
    params = parking_params_map
  if dataset == Dataset.MALAGA_ROUNDABOUT:
    params = parking_params_map_roundabout
  
  return params
