from typing import Any, Tuple, Optional, Mapping, Sequence

import cv2
import numpy as np
import time
import json
import sys

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from os.path import join

import lightning.pytorch as pl

import wandb

from .dsta_sphere_stereo_cvpr2021.python.mvs_processor import MVSProcessor

sys.path.append("/workspace")
from dsta_mvs.mvs_utils.camera_models import LinearSphere, DoubleSphere
from dsta_mvs.mvs_utils.ftensor.ftensor import FTensor
from dsta_mvs.image_sampler.generic_camera_model_sampler import GenericCameraModelSampler

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
        sigma_s: float
    ):

        super().__init__()

        self.save_hyperparameters()
        self.validation_metrics = nn.ModuleDict(validation_metrics)
        
        self.val_loader_names = dict()
        for i, data_key in enumerate(val_loader_names.keys()):
            settings = val_loader_names[data_key]
            settings.update({"name":data_key})
            self.val_loader_names.update({i: settings})

            depth_estimator_for_set = MVSProcessor(
                dist_range[0], dist_range[1],
                candidate_count,
                matching_resolution,
                settings["rgb_to_stitch_resolution"],
                settings["panorama_resolution"],
                reference_indices,
                sigma_i,
                sigma_s
            )
            depth_estimator_for_set.initialize(
                join(settings["path"], "calibration.json"),
                join(settings["path"], "masks")
            )

            settings.update({"estimator":depth_estimator_for_set})
            calib_f = open(join(settings["path"], "calibration.json"), 'r')
            calib_json = json.load(calib_f)

            if settings["warp_linear_to_double"]:
                settings.update({"samplers":self.build_samplers(calib_json)})

            print("==============================")
            print(f"Calibration.json for {data_key}")
            print(calib_json)

        self.lab_min = visualization_range[0]
        self.lab_max = visualization_range[1]

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
  
    def validation_step(self, batch, batch_idx, dataloader_idx:int = 0):
        imgs        = batch['imgs_raw']

        if 'inv_dist_idx' in batch:
            labels  = batch['inv_dist_idx']
        else:
            labels  = None

        val_set_settings = self.val_loader_names[dataloader_idx]
        val_set_name = val_set_settings["name"]
        print_imgs = val_set_settings["print_imgs"]
        compute_metrics = val_set_settings["compute_metrics"]
        print_imgs_freq = val_set_settings["print_imgs_batch_freq"]
        estimator = val_set_settings["estimator"]

        # Real Time Sphere Sweeping Depth Estimation
        

        # Warp from Linear Sphere to Double Sphere if Needed
        if "samplers" in val_set_settings:
            samplers = val_set_settings["samplers"]
            
            images = list()
            print("########################")
            for i, img in enumerate(imgs):
                img_permuted = img.permute((0,3,1,2))
                samplers[i].device = img.device
                img_sampled, valid_mask = samplers[i](img_permuted.float())
                img_sampled = img_sampled.permute(0,2,3,1)
                valid_mask = torch.tensor(valid_mask, device=img_sampled.device).unsqueeze(0).unsqueeze(-1)

                print(valid_mask.shape)
                print(img_sampled.shape)

                img_sampled = torch.where(valid_mask, img_sampled, torch.tensor([0.0],device=img_sampled.device))
                images.append(img_sampled.squeeze(0).cpu().detach().numpy().astype(np.uint8))

        else:
            images = [ img.squeeze(0).cpu().detach().numpy() for img in imgs ]

        forward_start_time = time.time()
        res_dict = estimator(images)
        forward_time = time.time()

        inv_dist = np.nan_to_num(res_dict['inv_distance'])
        inv_dist = inv_dist[:int(inv_dist.shape[0]/2),:]

        print(1.0/inv_dist.max(), 1.0/inv_dist[inv_dist!=0].min())

        
        timing=dict()
        timing[f"forward_duration_{val_set_name}"] = forward_time - forward_start_time
        self.log_dict(timing, logger=True, on_step=True, on_epoch=True)

        # validation metrics
        metrics = dict()
        if compute_metrics:
            for metric_name, metric_func in self.validation_metrics.items():
                key_name = metric_name + "_" + val_set_name
                metrics[key_name] = metric_func(torch.tensor(inv_dist).unsqueeze(0).unsqueeze(0).to(labels.device), labels)

            self.log_dict(metrics, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        # Print Images
        if print_imgs and batch_idx % print_imgs_freq == 0:
            panorama = res_dict['rgb']
            self.log_images_cust(images, inv_dist, panorama, labels = labels, 
                                 presamp_imgs= imgs, suffix = f"_{val_set_name}")

        return metrics


    def log_images_cust(self, imgs: Tensor, pred: Tensor, panorama, labels: Tensor = None, presamp_imgs: Tensor = None, suffix: str = ''):
        print("Prediction Shape: ", pred.shape)

        # Normalize. TODO: fix range
        lab_min, lab_max = self.lab_min, self.lab_max
        invalid_mask = pred==0

        pred = ((1.0/pred.astype(np.float32)) - lab_min) / ( lab_max - lab_min )
        pred = (np.clip(pred, 0, 1)) * 255
        pred = cv2.applyColorMap(pred.astype(np.uint8),cv2.COLORMAP_JET)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        pred[invalid_mask] = 0

        #Crop Depth at 180deg for fairness
        #pred = pred[:int(pred.shape[0]/2),:,:]

        #panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

        if labels is not None:
            print("Label Shape: ", labels.shape)
            lab = labels[0, 0].detach().to('cpu').numpy()
            lab = (lab.astype(np.float32) - lab_min) / ( lab_max - lab_min )
            lab = np.clip(lab, 0, 1) * 255
            lab = cv2.applyColorMap(lab.astype(np.uint8),cv2.COLORMAP_JET)
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)

            outputs = np.vstack([pred, lab])
        else:
            outputs = pred

        # log manually because WandB logger is broken and increments step
        # whether you like it or not.
        self.logger.log_image(key=f'depth_prediction{suffix}', images=[outputs])

        in_imgs = np.vstack(imgs)
        in_imgs = cv2.cvtColor(in_imgs, cv2.COLOR_BGR2RGB)

        self.logger.log_image(key=f'inputs{suffix}', images=[in_imgs])

        self.logger.log_image(key=f'panorama{suffix}', images=[panorama])

        if presamp_imgs is not None:
            presamp_imgs = [ img.squeeze(0).cpu().detach().numpy() for img in presamp_imgs ]
            presamp_imgs = np.vstack(presamp_imgs)
            self.logger.log_image(key=f'presamp_inputs{suffix}', images=[presamp_imgs])