
from typing import Any, Dict

import copy
import cv2
from lightning import LightningModule, Trainer
import numpy as np
import re
import wandb

from lightning.pytorch.callbacks import Callback

import torch
from torch import Tensor
import torch.nn.functional as F

from ..visualization import (
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

class TrainingValidationVis(Callback):
    def __init__(self, 
                 label_vis_min: float, 
                 label_vis_max: float,
                 step_span: int=100,
                 ):
        super().__init__()
        self.label_vis_min = label_vis_min
        self.label_vis_max = label_vis_max
        self.step_span = step_span
        self.bf = None
        self.logger = None
        self.rank = None
        self.val_offline_count = None # Dict. Reference!
        self.val_offline_flag = None
        self.val_custom_step = None # Boolean.
        self.val_custom_step_name = None
        
        self.val_loader_names = None # A list dict. Settings for each val loader.
        self.val_depth_pred_keys = None # A list of strings.
        
        self.dataset_indexed_cam_models = None
        
        self.state = {
            'label_vis_min': self.label_vis_min,
            'label_vis_max': self.label_vis_max,
            'step_span': self.step_span,
        }
    
    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        
    def state_dict(self):
        return self.state.copy()
    
    def setup(self, trainer, pl_module, stage):
        self.bf = pl_module.mvs_model.dist_regressor.bf
        self.logger = pl_module.logger
        self.rank = pl_module.global_rank
        self.val_offline_count = pl_module.val_offline_count
        self.val_offline_flag = pl_module.val_offline_flag
        self.val_custom_step = pl_module.val_custom_step
        self.val_custom_step_name = pl_module.val_custom_step_name
        
        self.val_loader_names = pl_module.val_loader_names
        self.val_depth_pred_keys = pl_module.val_depth_pred_keys
    
    def setup_dataset_indexed_cam_models(self, trainer: Trainer):
        if self.dataset_indexed_cam_models is not None:
            return
        
        if trainer.val_dataloaders is None:
            return
        
        self.dataset_indexed_cam_models = []
        for i, val_dataloader in enumerate( trainer.val_dataloaders ):
            val_dataset = val_dataloader.dataset
            
            assert 'rig' in val_dataset.map_camera_model.keys(), \
                f'TrainingValidationVis: setup: '\
                f'i={i}, map_camera_model does not have a "rig" key. '\
                f'keys = {val_dataset.map_camera_model.keys()}'
            
            self.dataset_indexed_cam_models.append(
                copy.deepcopy( val_dataset.map_camera_model )
            )
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_dataset_indexed_cam_models(trainer)
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_dataset_indexed_cam_models(trainer)
    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.setup_dataset_indexed_cam_models(trainer)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        imgs     = batch['imgs']
        labels   = batch['inv_dist_idx']
        inv_dist = outputs['inv_dist']
        
        if batch_idx % self.step_span == 0:
            vis = self._create_vis(imgs, inv_dist, labels)
            self.logger.log_image(
                            key='depth_prediction_train', 
                            images=[vis] )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        imgs     = batch['imgs']
        inv_dist = outputs['inv_dist']
        
        if 'inv_dist_idx' in batch:
            labels  = batch['inv_dist_idx']
        else:
            labels  = None
        
        val_set_settings = self.val_loader_names[dataloader_idx]
        depth_pred_key = self.val_depth_pred_keys[dataloader_idx]

        print_imgs = val_set_settings["print_imgs"]
        print_warped_inputs = val_set_settings["print_warped_inputs"]
        print_imgs_freq = val_set_settings["print_imgs_batch_freq"]
        
        step_num = None
        vis = None
        vis_warped_inputs = None
        if self.rank == 0:
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
                                imgs[0].unsqueeze(0), 
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
            if self.rank == 0:
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
            metrics = outputs['metrics']
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

    def _create_vis(self, 
                    imgs: Tensor, 
                    preds: Tensor, 
                    labels: Tensor=None):
        '''TODO: Need docstring.
        '''

        # Stack the input images.
        stacked_input = multi_view_tensor_2_stacked_cv2_img(imgs)[0]

        pred = preds[0, 0].detach().to('cpu').numpy() / self.bf

        # Figure out the resize shape.
        H_stacked_input, W_stacked_input = stacked_input.shape[:2]
        W_pred = pred.shape[1]
        H_new = int( W_pred / W_stacked_input * H_stacked_input)
        
        # Resize the stacked input images.
        stacked_input_resized = cv2.resize( 
            stacked_input, ( W_pred, H_new ), interpolation=cv2.INTER_LINEAR )

        # Normalize. TODO: fix range
        lab_min, lab_max = self.label_vis_min / self.bf, self.label_vis_max / self.bf
        if labels is not None:
            lab = labels[0, 0].detach().to('cpu').numpy() / self.bf
            stacked_vis = render_with_gt(lab, pred, stacked_input_resized, lab_min, lab_max)
        else:
            pred_vis = render_visualization(pred, lab_min, lab_max )
            stacked_vis = np.concatenate( ( pred_vis, stacked_input_resized ), axis=0 )

        stacked_vis = cv2.cvtColor(stacked_vis, cv2.COLOR_BGR2RGB)
        
        return stacked_vis
    
    def _create_warped_inputs(self, 
                              outputs: Tensor, 
                              batch: dict, 
                              dataloader_idx: int):
        assert self.dataset_indexed_cam_models is not None

        # Convert the output to distance.
        bf = self.bf
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
        