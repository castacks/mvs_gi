from typing import Any, Tuple, Optional, Mapping, Sequence

import copy
import cv2
import numpy as np
import time
import json
import re
import sys

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from os.path import join

import lightning.pytorch as pl

import wandb

from .dsta_sphere_stereo_cvpr2021.python.mvs_processor import MVSProcessor

sys.path.append("/workspace")
from dsta_mvs.mvs_utils.camera_models import LinearSphere, DoubleSphere
from dsta_mvs.mvs_utils.ftensor.ftensor import FTensor
from dsta_mvs.image_sampler.generic_camera_model_sampler import GenericCameraModelSampler

from dsta_mvs.visualization import (
    multi_view_tensor_2_stacked_cv2_img,
    render_visualization,
    render_with_gt )

def get_cam_idx_from_cam_key(cam_key: str):
    '''cam_key is in the form of "camN"
    '''
    m = re.search(r'cam(\d+)', cam_key)
    if m is None:
        return None
    else:
        return int(m.group(1))

def inv_pose(pose: torch.Tensor):
    '''
    Assuming pose has a shape of (B, 4, 4).
    '''

    R = pose[:, :3, :3]
    t = pose[:, :3,  3].unsqueeze(-1)

    i_pose = torch.zeros_like( pose )

    Rtr = R.transpose(1, 2)
    i_pose[:, :3, :3] = Rtr
    i_pose[:, :3,  3] = ( -Rtr @ t ).squeeze(-1)
    i_pose[:,  3,  3] = 1.0

    return i_pose

class RealTimeSphereSweepWrapper(pl.LightningModule):
    def __init__(
        self,
        val_loader_names: dict,
        dist_range: Tuple[float, float],
        candidate_count: int,
        matching_resolution: Tuple[int, int],
        reference_indices: Sequence,
        validation_metrics: Mapping[str, nn.Module],
        visualization_range: Tuple[float, float],
        sigma_i: float,
        sigma_s: float,
        val_offline_flag: bool=False,
        val_custom_step: bool=False
    ):

        super().__init__()

        self.save_hyperparameters()
        self.validation_metrics = nn.ModuleDict(validation_metrics)
        
        # Hack for re_configure()
        self.rtss_dist_range = dist_range
        self.rtss_candidate_count = candidate_count
        self.rtss_matching_resolution = matching_resolution
        self.rtss_reference_indices = reference_indices
        self.rtss_sigma_i = sigma_i
        self.rtss_sigma_s = sigma_s

        # =======================================================================================
        # The following member variables are subject to change after recovering from a 
        # checkpoint!
        # =======================================================================================
        self.val_loader_names = dict()
        self.val_depth_pred_keys = dict()
        self.val_offline_count = dict()
        self.val_offline_flag = False
        self.lab_min = 0
        self.lab_max = 192 # Only for visualization
        self.val_custom_step = False
        self.val_custom_step_name = 'val_step'
        self.re_configure( val_loader_names=val_loader_names,
                           visualization_range=visualization_range,
                           val_offline_flag=val_offline_flag,
                           val_custom_step=val_custom_step )

    def re_configure(self, 
                     val_loader_names: dict=None, 
                     visualization_range: Tuple[float, float]=None,
                     val_offline_flag: bool=False,
                     val_custom_step: bool=False):
        self.val_offline_flag = val_offline_flag # Set this to True for offline validation.

        if val_loader_names is not None:
            self.val_loader_names = dict()
            self.val_depth_pred_keys = dict()
            self.val_offline_count = dict()
            for i, data_key in enumerate(val_loader_names.keys()):
                settings = val_loader_names[data_key]
                settings.update({"name":data_key})

                depth_estimator_for_set = MVSProcessor(
                    self.rtss_dist_range[0], self.rtss_dist_range[1],
                    self.rtss_candidate_count,
                    self.rtss_matching_resolution,
                    settings["rgb_to_stitch_resolution"],
                    settings["panorama_resolution"],
                    self.rtss_reference_indices,
                    self.rtss_sigma_i,
                    self.rtss_sigma_s
                )

                rtss_calib_fn = settings["rtss_calib_fn"]
                rtss_masks_fn = settings["mask_path"]

                depth_estimator_for_set.initialize(
                    join(settings["path"], rtss_calib_fn),
                    join(settings["path"], rtss_masks_fn)
                )

                settings.update({"estimator":depth_estimator_for_set})
                calib_f = open(join(settings["path"], rtss_calib_fn), 'r')
                calib_json = json.load(calib_f)

                if settings["warp_linear_to_double"]:
                    settings.update({"samplers":self.build_samplers(calib_json)})

                # print("==============================")
                # print(f"Calibration.json for {data_key}")
                # print(calib_json)

                self.val_loader_names.update({i: settings})
                self.val_offline_count[i] = 0
                self.val_depth_pred_keys[i] = f'depth_prediction_{data_key}'

        if visualization_range is not None:
            self.lab_min = visualization_range[0]
            self.lab_max = visualization_range[1]
            
        self.val_custom_step = val_custom_step

    def build_samplers(self, calib_json):
        samplers_list = list()
        for ds_dict in calib_json["intrinsics"]:
            ds = ds_dict["intrinsics"]

            camera_model_raw = LinearSphere(
                fov_degree=195, shape_struct={"H": 1024, "W": 1024}, in_to_tensor=True, out_to_numpy=False
            )
            camera_model_target = DoubleSphere(
                ds["xi"], ds["alpha"], ds["fx"], ds["fy"], ds["cx"], ds["cy"], 195,
                shape_struct={"H": 1024, "W": 1024}, in_to_tensor=True, out_to_numpy=False
            )
            R_raw_fisheye = FTensor(torch.eye(3), f0=None, f1="cif0")

            samplers_list.append(
                GenericCameraModelSampler(
                    camera_model_raw, camera_model_target, R_raw_fisheye,
                    postprocessing=lambda x,y:(x,y)
                )
            )

        return samplers_list
    
    def _register_validation_metrics(self):
        # Get the wandb logger.
        wb = self.logger.experiment
        
        # Register the validation metrics.
        wb.define_metric(self.val_custom_step_name)
        
        # Register the depth prediction images as metrics.
        for _, v in self.val_depth_pred_keys.items():
            wb.define_metric(v, step_metric=self.val_custom_step_name)

    def assign_dataset_indexed_cam_models(self, dataset_indexed_cam_models: list):
        '''dataset_indexed_cam_models is a list of dicts of camera models. The keys of the dict 
        are "cam0" to "camN". And there is a "rig" camera.
        '''
        self.dataset_indexed_cam_models = []
        assert isinstance(dataset_indexed_cam_models, list)
        for i, cam_models in enumerate( dataset_indexed_cam_models ):
            assert isinstance(cam_models, dict)
            assert 'rig' in cam_models.keys(), \
                f'i={i}, cam_models does not have a "rig" key. keys = {cam_models.keys()}'
            self.dataset_indexed_cam_models.append( cam_models )

    def validation_step(self, batch, batch_idx, dataloader_idx:int = 0):
        if self.val_offline_flag:
            if batch_idx == 0:
                self._register_validation_metrics()
        
        imgs    = batch['imgs_raw']
        imgs_ml = batch['imgs']

        if 'inv_dist_idx' in batch:
            labels  = batch['inv_dist_idx']
        else:
            labels  = None

        val_set_settings = self.val_loader_names[dataloader_idx]
        val_set_name = val_set_settings["name"]
        depth_pred_key = self.val_depth_pred_keys[dataloader_idx]
        compute_metrics = val_set_settings["compute_metrics"]

        print_imgs = val_set_settings["print_imgs"]
        print_warped_inputs = val_set_settings["print_warped_inputs"]
        print_imgs_freq = val_set_settings["print_imgs_batch_freq"]

        estimator = val_set_settings["estimator"]

        # Real Time Sphere Sweeping Depth Estimation

        # Warp from Linear Sphere to Double Sphere if Needed
        if "samplers" in val_set_settings:
            samplers = val_set_settings["samplers"]
            
            images = list()
            # print("########################")
            for i, img in enumerate(imgs):
                # TODO: Might be a bug. img is already a tensor in shape of [1, H, W, C].
                img_permuted = img.permute((0,3,1,2))
                samplers[i].device = img.device
                img_sampled, valid_mask = samplers[i](img_permuted.float())
                img_sampled = img_sampled.permute(0,2,3,1)
                valid_mask = torch.tensor(valid_mask, device=img_sampled.device).unsqueeze(0).unsqueeze(-1)

                # print(valid_mask.shape)
                # print(img_sampled.shape)

                img_sampled = torch.where(valid_mask, img_sampled, torch.tensor([0.0],device=img_sampled.device))

                # TODO: this might be a bug. From runtime debugging, imgs is a list of tensors that
                # are not normlized to [0, 1]. So currently the following line is fine.
                # However, to be safe, we should properly clip and re-scale to [0, 255] before 
                # converting to uint8.
                images.append(img_sampled.squeeze(0).cpu().detach().numpy().astype(np.uint8))

        else:
            images = [ img.squeeze(0).cpu().detach().numpy() for img in imgs ]

        # Memory and measurement.
        if self.val_offline_flag:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            gpu_mem_in_mb = torch.cuda.memory_allocated() / 1024.0**2 
        step_time_start = time.time()

        res_dict = estimator(images)
        
        # Memory and time measurement.
        if self.val_offline_flag:
            torch.cuda.synchronize()
        
        step_time_span = time.time() - step_time_start
        step_time_span_rtss = res_dict['time_span']
        
        if self.val_offline_flag:    
            gpu_mem_diff_mb = torch.cuda.max_memory_allocated() / 1024.0**2 - gpu_mem_in_mb # in MB.

        inv_dist = np.nan_to_num(res_dict['inv_distance'])
        inv_dist = inv_dist[:int(inv_dist.shape[0]/2),:]

        # print(1.0/inv_dist.max(), 1.0/inv_dist[inv_dist!=0].min())

        inv_dist = torch.tensor(inv_dist).unsqueeze(0).unsqueeze(0).to(imgs_ml.device)
        inv_dist = inv_dist * 96

        # validation metrics
        metrics = dict()
        if compute_metrics:

            # inv_dist_idx_min = self.dist_regressor.inv_dist_idx_min
            # inv_dist_idx_max = self.dist_regressor.inv_dist_idx_max

            # Hardcode.
            inv_dist_idx_min = 96/100.0
            inv_dist_idx_max = 96/0.5

            if batch_idx > 1:
                # Ignore the first two steps where time span tend to be longer.
                metrics[f'val_step_time_{val_set_name}'] = torch.Tensor([step_time_span])
                metrics[f'val_step_time_rtss_{val_set_name}'] = torch.Tensor([step_time_span_rtss])
                if self.val_offline_flag:
                    metrics[f'val_step_gpu_mem_in_mb_{val_set_name}'] = torch.Tensor([gpu_mem_in_mb])
                    metrics[f'val_step_gpu_mem_df_mb_{val_set_name}'] = torch.Tensor([gpu_mem_diff_mb])
                
            for metric_name, metric_func in self.validation_metrics.items():

                mask_labels = torch.logical_and( labels >= inv_dist_idx_min,
                                                 labels <= inv_dist_idx_max )

                key_name = metric_name + "_" + val_set_name
                metrics[key_name] = metric_func(
                    inv_dist, 
                    labels, 
                    mask_labels)
        
        # The following block is printing the panorama from rtss.
        # # Print Images
        # if print_imgs and batch_idx % print_imgs_freq == 0:
        #     panorama = res_dict['rgb']
        #     self.log_images_cust(images, inv_dist, panorama, labels = labels, 
        #                          presamp_imgs= imgs, suffix = f"_{val_set_name}")

        # Only execute on global rank 0.
        step_num = None
        vis = None
        vis_warped_inputs = None
        if self.global_rank == 0:
            # Print Images
            if print_imgs and ( batch_idx % print_imgs_freq == 0 ):

                if labels is not None:
                    labels = labels[0].unsqueeze(0)
                
                # Figure out the step number.
                if self.val_offline_flag:
                    if self.val_custom_step:
                        step_num = self.val_offline_count[dataloader_idx]
                        self.val_offline_count[dataloader_idx] += 1

                # print(f'eval >>> batch_idx = {batch_idx}, step_num = {step_num}')

                vis = self._create_vis(
                                imgs_ml[0].unsqueeze(0), 
                                inv_dist[0].unsqueeze(0), 
                                labels=labels )
                
                if print_warped_inputs:
                    warped_inputs = self._create_warped_inputs(
                        inv_dist,
                        batch,
                        dataloader_idx )
                    
                    vis_warped_inputs = self._create_vis(
                        warped_inputs[0].unsqueeze(0),
                        inv_dist[0].unsqueeze(0),
                        labels=labels )

        if not self.val_offline_flag:
            # This happens during training.
            # metrics and vis will be logged separately.
            self.log_dict( metrics, 
                           prog_bar=False, 
                           logger=True, 
                           on_step=True, 
                           on_epoch=True, 
                           sync_dist=True )

            if self.global_rank == 0:
                if vis is not None:
                    if self.val_custom_step:
                        self.logger.experiment.log(
                            {
                                depth_pred_key: [ wandb.Image( vis ) ],
                                self.val_custom_step_name: step_num
                            }
                        )
                    else:
                        self.logger.log_image(
                            key=depth_pred_key, 
                            images=[vis],
                            step=step_num )
        else:
            # self.val_offline_flag == True
            # This happens during offline validation.
            # Implies/assumes that we are running with a single GPU and we don't need to sync.
            if print_imgs:
                # Will log everything together.
                logged_metrics = copy.deepcopy(metrics)
                
                if vis is not None:
                    logged_metrics[ depth_pred_key ] = [ wandb.Image( vis ) ]

                if vis_warped_inputs is not None:
                    wapred_inputs_key = f'{depth_pred_key}_warped_inputs'
                    logged_metrics[ wapred_inputs_key ] = [ wandb.Image( vis_warped_inputs ) ]

                if self.val_custom_step:
                    logged_metrics[ self.val_custom_step_name ] = step_num

                self.logger.experiment.log( logged_metrics )
            else:
                self.logger.experiment.log( metrics )

        return metrics

    def _create_vis(self, 
                    imgs: Tensor, 
                    preds: Tensor, 
                    labels: Tensor=None):
        '''TODO: Need docstring.
        '''
        # bf = self.dist_regressor.bf
        bf = 96

        # Stack the input images.
        stacked_input = multi_view_tensor_2_stacked_cv2_img(imgs)[0]

        pred = preds[0, 0].detach().to('cpu').numpy() / bf

        # Figure out the resize shape.
        H_stacked_input, W_stacked_input = stacked_input.shape[:2]
        W_pred = pred.shape[1]
        H_new = int( W_pred / W_stacked_input * H_stacked_input)
        
        # Resize the stacked input images.
        stacked_input_resized = cv2.resize( 
            stacked_input, ( W_pred, H_new ), interpolation=cv2.INTER_LINEAR )

        # Normalize. TODO: fix range
        lab_min, lab_max = self.lab_min / bf, self.lab_max / bf
        if labels is not None:
            lab = labels[0, 0].detach().to('cpu').numpy() / bf
            stacked_vis = render_with_gt(lab, pred, stacked_input_resized, lab_min, lab_max)
        else:
            pred_vis = render_visualization(pred, lab_min, lab_max )
            stacked_vis = np.concatenate( ( pred_vis, stacked_input_resized ), axis=0 )

        stacked_vis = cv2.cvtColor(stacked_vis, cv2.COLOR_BGR2RGB)
        
        return stacked_vis

    def _create_warped_inputs(self, outputs: Tensor, batch: dict, dataloader_idx: int):
        assert self.dataset_indexed_cam_models is not None

        # Convert the output to distance.
        # bf = self.dist_regressor.bf
        bf = 96
        dist = bf / outputs

        # Camera models.
        cam_models = self.dataset_indexed_cam_models[dataloader_idx]
        cam_model_rig = cam_models['rig']

        # Get the dimension of the output.
        out_H, out_W = cam_model_rig.shape
        
        # Convert the distance to xyz vectors in the output camera frame.
        rays, _ = cam_model_rig.get_rays_wrt_sensor_frame()
        rays = rays.unsqueeze(0).to( device=dist.device ) # [B, 3, N]
        dist = dist.view(-1, 1, out_H*out_W) # [B, 1, N]
        xyz = rays * dist

        warped_input_dict = dict()
        for cam_key, cam_model in cam_models.items():
            # Get the camera index.
            cam_idx = get_cam_idx_from_cam_key(cam_key)

            if cam_idx is None:
                # Not a regular camera.
                continue

            # Get the camera pose.
            cam_pose = batch['cam_poses'][:, cam_idx, ...] # [B, 4, 4]

            # Transformt the xyz vectors from the output camera frame to the input camera frame.
            cam_pose_inv = inv_pose(cam_pose)
            xyz_cam = cam_pose_inv[..., :3, :3] @ xyz + cam_pose_inv[..., :3, 3].unsqueeze(-1)

            # Project the transformed xyz vectors to the input image and get sampling locations.
            sampling_points, _ = cam_model.point_3d_2_pixel(xyz_cam, normalized=True)

            # Convert sampling_points to the format of sampling grid.
            # sampling_points has a shape of (B, 2, N), convert it to (B, H, W, 2)
            sampling_grid = sampling_points.permute(0, 2, 1).contiguous().view(-1, out_H, out_W, 2)

            # Sample the warped image.
            warped_input = F.grid_sample( 
                batch['imgs'][:, cam_idx, ...], 
                sampling_grid, 
                align_corners=False, 
                mode='bilinear' )
            
            warped_input_dict[cam_idx] = warped_input

        return torch.stack( 
            [ warped_input_dict[cam_idx] for cam_idx in sorted( warped_input_dict.keys() ) ],
            dim=1 ) # [B, X, C, H, W]
    