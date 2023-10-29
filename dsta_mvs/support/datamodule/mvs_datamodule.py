import copy
import os
from typing import Any, Dict, Sequence

import numpy as np

from torch.utils.data import DataLoader

import lightning.pytorch as pl

from ...mvs_utils.camera_models import CAMERA_MODELS
from ...mvs_utils.camera_models import make_object as make_camera_model_object
from ..dataset.multi_view_camera_model_dataset import (
    MultiViewCameraModelDataset, ROSDroneImagesDataset )
from ..dataset.grid_maker_from_manifest import SurrogateCameraModelGridMaker

# For compatibility with the old code.
CAM_MODEL_INPUT = dict(
    type='Equirectangular',
    shape_struct=dict(H=512, W=2048),
    latitude_span=( -np.pi/2, 0 ),
    open_span=False,
    in_to_tensor=True, 
    out_to_numpy=False 
)

# For compatibility with the old code.
CAM_MODEL_OUTPUT = dict(
    type='Equirectangular',
    shape_struct=dict(H=160, W=640),
    latitude_span=( -np.pi/2, 0 ),
    open_span=False,
    in_to_tensor=True, 
    out_to_numpy=False
)

# For compatibility with the old code.
CONF_ADDITIONAL_CAMERA_MODEL = {
    'cv': dict(
        type='Equirectangular',
        shape_struct={'H': 80, 'W': 320},
        latitude_span=( -np.pi/2, 0 ),
        open_span=False,
        in_to_tensor=True, 
        out_to_numpy=False
    ),
}

# For compatibility with the old code.
CONF_MAP_SURROGATE_CAMERA_MODEL={
    'cam0': CAM_MODEL_INPUT,
    'cam1': CAM_MODEL_INPUT,
    'cam2': CAM_MODEL_INPUT,
    'rig':  CAM_MODEL_OUTPUT,
}

# For compatibility with the old code.
MAP_CAMERA_FRAME = {
    'cam0': 'cif0s', # The "s" suffix is for "surrogate".
    'cam1': 'cif1s',
    'cam2': 'cif2s',
    'rig':  'rifs',
    'cv':   'cv',
}

class MVSLocalDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        bf: float,
        dist_list: Sequence[float],
        data_dirs: dict,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        num_workers: int = 1,
        conf_cam_model_input: dict=CAM_MODEL_INPUT, # Right now, this is for compatibility with the old code.
        conf_cam_model_output: dict=CAM_MODEL_OUTPUT, # Right now, this is for compatibility with the old code.
        conf_additional_camera_model: dict=CONF_ADDITIONAL_CAMERA_MODEL, # Right now, this is for compatibility with the old code.
        conf_map_surrogate_camera_model: dict=CONF_MAP_SURROGATE_CAMERA_MODEL, # Right now, this is for compatibility with the old code.
        map_camera_frame: dict=MAP_CAMERA_FRAME # Right now, this is for compatibility with the old code.
    ):
        super().__init__()
        
        self.bf = bf
        self.dist_list = dist_list
        self.data_dirs = data_dirs
        self.batch_size = self.data_dirs["main"]["batch_size"]
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.num_workers = num_workers
        
        self.cam_model_input = conf_cam_model_input # Right now, this is for compatibility with the old code.
        self.cam_model_output = conf_cam_model_output # Right now, this is for compatibility with the old code.
        self.config_additional_camera_model = conf_additional_camera_model # Right now, this is for compatibility with the old code.
        self.conf_map_surrogate_camera_model = conf_map_surrogate_camera_model # Right now, this is for compatibility with the old code.
        self.map_camera_frame = map_camera_frame # Right now, this is for compatibility with the old code.

    def initialize_new_grid_maker_variables(self, dataset_path):

        self.grid_maker_builder = SurrogateCameraModelGridMaker(
            fn = os.path.join( dataset_path, 'manifest.json' ),
            conf_map_surrogate_camera_model=self.conf_map_surrogate_camera_model
        )

        self.camera_model_mvs = copy.deepcopy( self.grid_maker_builder.camera_model_map )
        
        for key, cam_model_spec in self.config_additional_camera_model.items():
            self.camera_model_mvs[key] = make_camera_model_object(CAMERA_MODELS, cam_model_spec)


    def setup(self, stage: str) -> None:
        
        self.initialize_new_grid_maker_variables(self.data_dirs["main"]["path"])

        if stage == 'fit':
            self.train_dataset = MultiViewCameraModelDataset(
                dataset_path = self.data_dirs["main"]["path"], 
                map_camera_model_raw = self.grid_maker_builder.camera_model_map_ori,
                map_camera_model = self.camera_model_mvs,
                map_grid_maker = self.grid_maker_builder.grid_maker_map,
                metadata_fn = 'metadata.json',
                frame_graph_fn = 'frame_graph.json',
                conf_dist_blend_func = dict(
                    type='BlendBy2ndOrderGradTorch',
                    threshold_scaling_factor = 0.01
                ),
                conf_dist_lab_tab = dict(
                    type='FixedDistLabelTable',
                    bf = self.bf,
                    dist_list = self.dist_list
                ),
                map_camera_frame=self.map_camera_frame,
                cam_key_cv = 'cv',
                align_corners = False,
                align_corners_nearest = False,
                data_keys = ['train'],
                mask_path = self.data_dirs["main"]["mask_path"],
                csv_input_rgb_suffix = '_rgb_fisheye',
                csv_rig_rgb_suffix='_rgb_fisheye',
                csv_rig_dist_suffix='_dist_fisheye',
                step_size=self.data_dirs["main"]["step_size"],
                max_length=self.data_dirs["main"]["max_length"]
            )

        self.val_datasets = list()
        if stage == 'fit' or stage == 'validate':
            for ds_key in self.data_dirs:
                self.initialize_new_grid_maker_variables(self.data_dirs[ds_key]["path"])

                dataset_settings = self.data_dirs[ds_key]
                dataset_path = dataset_settings["path"]
                dataset_mask_path = dataset_settings["mask_path"]
                dataset_batch_sz = dataset_settings["batch_size"]
                dataset_type = dataset_settings["type"]
                step_size = dataset_settings["step_size"]
                keep_raw_imgs = dataset_settings["keep_raw_imgs"]

                if dataset_type == "real":
                    num_samples = dataset_settings["num_samples"]
                    order = dataset_settings["order"]

                    curr_dataset = ROSDroneImagesDataset(
                        dataset_path = dataset_path, 
                        order = order,
                        map_camera_model_raw = self.grid_maker_builder.camera_model_map_ori,
                        map_camera_model = self.camera_model_mvs,
                        map_grid_maker = self.grid_maker_builder.grid_maker_map,
                        metadata_fn = 'metadata.json',
                        frame_graph_fn = 'frame_graph.json',
                        conf_dist_blend_func = dict(
                            type='BlendBy2ndOrderGradTorch',
                            threshold_scaling_factor = 0.01
                        ),
                        conf_dist_lab_tab = dict(
                            type='FixedDistLabelTable',
                            bf = self.bf,
                            dist_list = self.dist_list
                        ),
                        map_camera_frame=self.map_camera_frame,
                        cam_key_cv = 'cv',
                        align_corners = False,
                        align_corners_nearest = False,
                        data_keys = [ds_key],
                        mask_path = dataset_mask_path,
                        num_samples = num_samples,
                        step_size = step_size,
                        keep_raw_image=keep_raw_imgs
                    )

                elif dataset_type == "synthetic":
                    csv_rig_rgb_suffix  = dataset_settings['csv_rig_rgb_suffix']
                    csv_rig_dist_suffix = dataset_settings['csv_rig_dist_suffix']

                    curr_dataset = MultiViewCameraModelDataset(
                        dataset_path = dataset_path, 
                        map_camera_model_raw = self.grid_maker_builder.camera_model_map_ori,
                        map_camera_model = self.camera_model_mvs,
                        map_grid_maker = self.grid_maker_builder.grid_maker_map,
                        metadata_fn = 'metadata.json',
                        frame_graph_fn = 'frame_graph.json',
                        conf_dist_blend_func = dict(
                            type='BlendBy2ndOrderGradTorch',
                            threshold_scaling_factor = 0.01
                        ),
                        conf_dist_lab_tab = dict(
                            type='FixedDistLabelTable',
                            bf = self.bf,
                            dist_list = self.dist_list
                        ),
                        map_camera_frame=self.map_camera_frame,
                        cam_key_cv = 'cv',
                        align_corners = False,
                        align_corners_nearest = False,
                        data_keys = ['validate' if ds_key.startswith( "main" ) else ds_key],
                        mask_path = dataset_mask_path,
                        csv_input_rgb_suffix = '_rgb_fisheye',
                        csv_rig_rgb_suffix=csv_rig_rgb_suffix,
                        csv_rig_dist_suffix=csv_rig_dist_suffix,
                        step_size=step_size,
                        max_length=dataset_settings["max_length"],
                        keep_raw_image=keep_raw_imgs
                    )
                else:
                    raise ValueError(f"Invalid dataset type: {dataset_type}")

                self.val_datasets.append(
                    {"dataset": curr_dataset,
                     "batch_size": dataset_batch_sz}
                )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_train,
            num_workers=self.num_workers
        )

        return train_dataloader
    
    def val_dataloader(self):
        val_dataloaders = [ DataLoader(
            dataset=val_dict["dataset"], 
            batch_size=val_dict["batch_size"], 
            shuffle=self.shuffle_val,
            num_workers=self.num_workers
        ) for val_dict in self.val_datasets]

        return val_dataloaders
        