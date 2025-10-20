from typing import Optional
import warnings
from platform import python_version
import numpy as np
import pandas as pd
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
from waymo_open_dataset.utils import  frame_utils

def init():
    print(python_version())
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Disable annoying warnings from PyArrow using under the hood.
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option("display.max_colwidth", None)


def load_and_merge_wod_dataframes(read, include_lidar_calibration=False):
    # Lazily read DataFrames for all components.
    association_df = read('camera_to_lidar_box_association')
    cam_box_df = read('camera_box')
    cam_img_df = read('camera_image')
    lidar_box_df = read('lidar_box')
    lidar_df = read('lidar')
    projected_lidar_box_df = read('projected_lidar_box')

    association_df = association_df[association_df['key.camera_name'] == 1]
    cam_img_df = cam_img_df[cam_img_df['key.camera_name'] == 1].compute()
    projected_lidar_box_df = projected_lidar_box_df[projected_lidar_box_df['key.camera_name'] == 1]
    lidar_df = lidar_df[lidar_df['key.laser_name'] == 1]

    lidar_box_proj = v2.merge(lidar_box_df, projected_lidar_box_df, left_nullable=True, right_nullable=True)
    lidar_box_proj_ri = v2.merge(lidar_box_proj, lidar_df, left_nullable=True, right_nullable=True)
    lidar_box_proj_ri_img = v2.merge(lidar_box_proj_ri, cam_img_df, left_group=True)

    if include_lidar_calibration:
        lidar_calibration_df = read('lidar_calibration')
        lidar_calibration_df = lidar_calibration_df[lidar_calibration_df['key.laser_name'] == 1].compute()
        return (lidar_box_proj_ri_img, lidar_calibration_df)
    else:
        return (lidar_box_proj_ri_img)

