from typing import Any, Tuple, Optional, Mapping

import re
import time

import torch
from torch import nn

import lightning.pytorch as pl

from .torch_only import SphericalSweepStereoBase
from ...support.loss_function.distance_loss import *

class SphericalSweepStereo2024(pl.LightningModule):
    def __init__(
        self,
        val_loader_names: dict,
        feature_extractor: nn.Module,
        cv_builder: nn.Module,
        cv_regulator: nn.Module,
        dist_regressor: nn.Module,
        augmentation: nn.Module,
        validation_metrics: Mapping[str, nn.Module],
        volume_loss: nn.Module,
        distance_loss: nn.Module,
        loss_weights: Tuple[float, float], # volume, dist
        val_offline_flag: bool=False,
        val_custom_step: bool=False
    ):
        super().__init__()

        self.save_hyperparameters()

        self.mvs_model = SphericalSweepStereoBase( feature_extractor, 
                                                   cv_builder,
                                                   cv_regulator,
                                                   dist_regressor )

        self.augmentation = augmentation
        self.validation_metrics = nn.ModuleDict(validation_metrics)

        self.volume_loss = volume_loss # VolumeCrossEntropy(bf, dist_list)
        self.distance_loss = distance_loss # nn.SmoothL1Loss()
        self.loss_weights = loss_weights

        self.last_start_time = None
        
        # =======================================================================================
        # The following member variables are subject to change after recovering from a 
        # checkpoint!
        # =======================================================================================
        self.val_loader_names = dict()
        self.val_depth_pred_keys = dict()
        self.val_offline_count = dict()
        self.val_offline_flag = False
        self.val_custom_step = False
        self.val_custom_step_name = 'val_step'
        self.re_configure( val_loader_names=val_loader_names,
                           val_offline_flag=val_offline_flag,
                           val_custom_step=val_custom_step )

        # Every element of self.dataset_indexed_cam_models will be a dict of camera models.
        # This list is indexed by the dataloader index.
        # This list is only for creating visualizations during offline validation.
        self.dataset_indexed_cam_models = None

    def re_configure(self, 
                     val_loader_names: dict=None, 
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
                self.val_loader_names.update({i: settings})
                self.val_offline_count[i] = 0
                self.val_depth_pred_keys[i] = f'depth_prediction_{data_key}'
            
        self.val_custom_step = val_custom_step

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

    def training_step(self, batch, batch_idx):
        step_start_time = time.time()
        
        imgs        = batch['imgs']
        grids       = batch['grids']
        grid_masks  = batch['grid_masks']
        labels      = batch['inv_dist_idx']
        masks       = batch['masks']

        # Data augmentation
        # TODO: Check if we need to use grid_masks.
        imgs, grids, masks = self.augmentation((imgs, grids, masks))

        # Forward.
        forward_start_time = time.time()
        inv_dist, norm_costs = self.mvs_model(imgs, grids, grid_masks, masks)
        forward_end_time = time.time()

        # Compute volume loss.
        loss = dict()
        loss['vol_loss'] = self.volume_loss(norm_costs, labels)
        
        # Compute distance loss.
        loss_mask = labels <= 191 # TODO: Hard coded.
        loss['distance_loss'] = self.distance_loss(inv_dist, labels, mask=loss_mask)

        # Combined loss
        loss['loss'] = self.loss_weights[0] * loss['vol_loss'] \
                     + self.loss_weights[1] * loss['distance_loss']
        
        self.log_dict(loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        timing=dict()
        timing["forward_duration"] = forward_end_time - forward_start_time

        if self.last_start_time is not None:
            timing["iter_duration"] = step_start_time - self.last_start_time

        self.log_dict(timing, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # Update the last start time.
        self.last_start_time = step_start_time

        return {
            'loss': loss,
            'inv_dist': inv_dist,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx:int = 0):
        if self.val_offline_flag:
            if batch_idx == 0:
                self._register_validation_metrics()
        
        imgs        = batch['imgs']
        grids       = batch['grids']
        grid_masks  = batch['grid_masks']
        masks       = batch['masks']

        if 'inv_dist_idx' in batch:
            labels  = batch['inv_dist_idx']
        else:
            labels  = None

        val_set_settings = self.val_loader_names[dataloader_idx]
        val_set_name = val_set_settings["name"]
        compute_metrics = val_set_settings["compute_metrics"]

        # Memory and measurement.
        if self.val_offline_flag:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            gpu_mem_in_mb = torch.cuda.memory_allocated() / 1024.0**2 
        step_time_start = time.time()

        inv_dist, _ = self.mvs_model(imgs, grids, grid_masks, masks)
        
        # Memory and time measurement.
        if self.val_offline_flag:
            torch.cuda.synchronize()
        
        step_time_span = time.time() - step_time_start
        
        if self.val_offline_flag:    
            gpu_mem_diff_mb = torch.cuda.max_memory_allocated() / 1024.0**2 - gpu_mem_in_mb # in MB.

        # validation metrics
        metrics = dict()
        if compute_metrics:

            inv_dist_idx_min = self.dist_regressor.inv_dist_idx_min
            inv_dist_idx_max = self.dist_regressor.inv_dist_idx_max

            if batch_idx > 1:
                # Ignore the first two steps where time span tend to be longer.
                metrics[f'val_step_time_{val_set_name}'] = torch.Tensor([step_time_span])
                if self.val_offline_flag:
                    metrics[f'val_step_gpu_mem_in_mb_{val_set_name}'] = torch.Tensor([gpu_mem_in_mb])
                    metrics[f'val_step_gpu_mem_df_mb_{val_set_name}'] = torch.Tensor([gpu_mem_diff_mb])
                
            for metric_name, metric_func in self.validation_metrics.items():

                mask_labels = torch.logical_and( labels >= inv_dist_idx_min,
                                                 labels <= inv_dist_idx_max )

                key_name = metric_name + "_" + val_set_name
                metrics[key_name] = metric_func(inv_dist, labels, mask_labels)
        
        if not self.val_offline_flag:
            # This happens during training.
            # metrics is logged here.
            self.log_dict( metrics, 
                           prog_bar=False, 
                           logger=True, 
                           on_step=True, 
                           on_epoch=True, 
                           sync_dist=True )
            # Visualizations are logged in the callback.
        else:
            # self.val_offline_flag == True
            # This happens during offline validation.
            # Implies/assumes that we are running with a single GPU and we don't need to sync.
            # All logging should be handled by the callback.
            pass

        return {
            'metrics': metrics,
            'inv_dist': inv_dist,
        }
    