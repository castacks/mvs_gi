import copy
import os

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from ..mvs_utils.camera_models import CAMERA_MODELS
from ..mvs_utils.camera_models import make_object as make_camera_model_object
from ..support.dataset.multi_view_camera_model_dataset import MultiViewCameraModelDataset
from ..support.dataset.grid_maker_from_manifest import SurrogateCameraModelGridMaker
from ..model.mvs_model.torch_only import SphericalSweepStereoBase

def make_dataloader(
        data_dir,
        bf=96,
        dist_list=[0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100],
        csv_rig_rgb_suffix='_rgb_equirect', # _rgb_fisheye
        csv_rig_dist_suffix='_dist_equirect' # _dist_fisheye
):
    cam_model_input = dict(
                        type='Equirectangular',
                        shape_struct=dict(H=512, W=2048),
                        latitude_span=( -np.pi/2, 0 ),
                        open_span=False,
                        in_to_tensor=True, 
                        out_to_numpy=False 
                    )
    
    cam_model_output = dict(
                            type='Equirectangular',
                            shape_struct=dict(H=160, W=640),
                            latitude_span=( -np.pi/2, 0 ),
                            open_span=False,
                            in_to_tensor=True, 
                            out_to_numpy=False
                        )

    map_camera_frame = {
                            'cam0': 'cif0s', # The "s" suffix is for "surrogate".
                            'cam1': 'cif1s',
                            'cam2': 'cif2s',
                            'rig':  'rifs',
                            'cv':   'cv',
                        }

    config_additional_camera_model = {
                                        'cv': dict(
                                            type='Equirectangular',
                                            shape_struct={'H': 80, 'W': 320},
                                            latitude_span=( -np.pi/2, 0 ),
                                            open_span=False,
                                            in_to_tensor=True, 
                                            out_to_numpy=False
                                        ),
                                    }


    grid_maker_builder = SurrogateCameraModelGridMaker(
            fn = os.path.join( data_dir, 'manifest.json' ),
            conf_map_surrogate_camera_model={
                                                'cam0': cam_model_input,
                                                'cam1': cam_model_input,
                                                'cam2': cam_model_input,
                                                'rig': cam_model_output,
                                            }
        )

        
    camera_model_mvs = copy.deepcopy( grid_maker_builder.camera_model_map )
    
    for key, cam_model_spec in config_additional_camera_model.items():
        camera_model_mvs[key] = make_camera_model_object(CAMERA_MODELS, cam_model_spec)

    train_dataset = MultiViewCameraModelDataset(
        dataset_path = data_dir, 
        map_camera_model_raw = grid_maker_builder.camera_model_map_ori,
        map_camera_model = camera_model_mvs,
        map_grid_maker = grid_maker_builder.grid_maker_map,
        metadata_fn = 'metadata.json',
        frame_graph_fn = 'frame_graph.json',
        conf_dist_blend_func = dict(
            type='BlendBy2ndOrderGradTorch',
            threshold_scaling_factor = 0.01
        ),
        conf_dist_lab_tab = dict(
            type='FixedDistLabelTable',
            bf = bf,
            dist_list = dist_list
        ),
        map_camera_frame=map_camera_frame,
        cam_key_cv = 'cv',
        align_corners = False,
        align_corners_nearest = False,
        data_keys = ['train'],
        mask_path = 'masks.json',
        csv_input_rgb_suffix = '_rgb_fisheye',
        csv_rig_rgb_suffix=csv_rig_rgb_suffix,
        csv_rig_dist_suffix=csv_rig_dist_suffix,
    )


    dataloader = DataLoader(
                dataset=train_dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=1
            )
    
    return dataloader

def make_dataloader_4cam(
        data_dir,
        bf=96,
        dist_list=[0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100],
        csv_rig_rgb_suffix='_rgb_equirect', # _rgb_fisheye
        csv_rig_dist_suffix='_dist_equirect' # _dist_fisheye
):
    cam_model_input = dict(
                        type='Equirectangular',
                        shape_struct=dict(H=512, W=2048),
                        latitude_span=( -np.pi/2, 0 ),
                        open_span=False,
                        in_to_tensor=True, 
                        out_to_numpy=False 
                    )
    
    cam_model_output = dict(
                            type='Equirectangular',
                            shape_struct=dict(H=160, W=640),
                            latitude_span=( -np.pi/2, 0 ),
                            open_span=False,
                            in_to_tensor=True, 
                            out_to_numpy=False
                        )

    map_camera_frame = {
                            'cam0': 'cif0s', # The "s" suffix is for "surrogate".
                            'cam1': 'cif1s',
                            'cam2': 'cif2s',
                            'cam3': 'cif3s',
                            'rig':  'rifs',
                            'cv':   'cv',
                        }

    config_additional_camera_model = {
                                        'cv': dict(
                                            type='Equirectangular',
                                            shape_struct={'H': 80, 'W': 320},
                                            latitude_span=( -np.pi/2, 0 ),
                                            open_span=False,
                                            in_to_tensor=True, 
                                            out_to_numpy=False
                                        ),
                                    }


    grid_maker_builder = SurrogateCameraModelGridMaker(
            fn = os.path.join( data_dir, 'manifest.json' ),
            conf_map_surrogate_camera_model={
                                                'cam0': cam_model_input,
                                                'cam1': cam_model_input,
                                                'cam2': cam_model_input,
                                                'cam3': cam_model_input,
                                                'rig': cam_model_output,
                                            }
        )

        
    camera_model_mvs = copy.deepcopy( grid_maker_builder.camera_model_map )
    
    for key, cam_model_spec in config_additional_camera_model.items():
        camera_model_mvs[key] = make_camera_model_object(CAMERA_MODELS, cam_model_spec)

    train_dataset = MultiViewCameraModelDataset(
        dataset_path = data_dir, 
        map_camera_model_raw = grid_maker_builder.camera_model_map_ori,
        map_camera_model = camera_model_mvs,
        map_grid_maker = grid_maker_builder.grid_maker_map,
        metadata_fn = 'metadata.json',
        frame_graph_fn = 'frame_graph.json',
        conf_dist_blend_func = dict(
            type='BlendBy2ndOrderGradTorch',
            threshold_scaling_factor = 0.01
        ),
        conf_dist_lab_tab = dict(
            type='FixedDistLabelTable',
            bf = bf,
            dist_list = dist_list
        ),
        map_camera_frame=map_camera_frame,
        cam_key_cv = 'cv',
        align_corners = False,
        align_corners_nearest = False,
        data_keys = ['train'],
        mask_path = 'masks.json',
        csv_input_rgb_suffix = '_rgb_fisheye',
        csv_rig_rgb_suffix=csv_rig_rgb_suffix,
        csv_rig_dist_suffix=csv_rig_dist_suffix,
    )


    dataloader = DataLoader(
                dataset=train_dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=1
            )
    
    return dataloader

# def render_visualization(inv_dist, min_val=None, max_val=None):
#     min_val = min_val if min_val is not None else inv_dist.min()
#     max_val = max_val if max_val is not None else inv_dist.max()
#     inv_dist = (inv_dist.astype(np.float32) - min_val) / ( max_val - min_val )
#     inv_dist = np.clip(inv_dist, 0, 1) * 255
#     inv_dist = cv2.applyColorMap(inv_dist.astype(np.uint8), cv2.COLORMAP_JET)
#     return inv_dist

def load_model(chkpt_path):
    chkpt = torch.load(chkpt_path)
    hparams = chkpt['hyper_parameters']
    model = SphericalSweepStereoBase(
        feature_extractor = hparams['feature_extractor'],
        cv_builder = hparams['cv_builder'],
        cv_regulator = hparams['cv_regulator'],
        dist_regressor = hparams['dist_regressor']
    )
    return model