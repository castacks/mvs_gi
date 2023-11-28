
# Ordinary packages.
import numpy as np
import os
import re
import json
import csv
from typing import Sequence

# PyTorch and torchvision.
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision.transforms import ToTensor, ColorJitter

# Local.
# from ...mvs_utils.debug import (show_obj, show_sum)
from ...mvs_utils.ftensor import FTensor, f_ones, RefFrame
from ...mvs_utils.image_io import read_image, read_mask, read_compressed_float
from ...mvs_utils.metadata_reader import MetadataReader
from ...image_sampler import CameraModelRotation as ImageSampler
from ...image_sampler import NoOpSampler, INTER_BLENDED, BLEND_FUNCTIONS, BlendBy2ndOrderGradTorch
from ...image_sampler import make_object as make_sampler_object

from .dist_label_table import LinspaceDistLabelTable
from .torch_cuda_sweep import RayMaker, transform_3D_points_torch

from ..register import (
    DATASET, # Dataset.
    DIST_LAB_TAB,
    register,
    make_support_object)

def cam_idx_from_cam_key(cam_key):
    m = re.search(r'(\d+)$', cam_key)
    assert m is not None, f'cam_key {cam_key} has no trailing numbers. '
    return int(m.group(1))

@register(DATASET)
class MultiViewCameraModelDataset(Dataset):

    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            metadata_fn="metadata.json",
            frame_graph_fn="frame_graph.json",
            conf_dist_blend_func=None, # The blend function used by the sampler operating on distance images.
            conf_dist_lab_tab=None,
            map_camera_frame=None, # Camera frames for the output images. Should be a dict.
            cam_key_cv='cv', # The virtual camera for the cost volume.
            true_grid=False,
            align_corners=False,
            align_corners_nearest=False,
            keep_raw_image=False,
            data_keys=['train'],
            mask_path='masks.json', # relative to dataset root
            csv_input_rgb_suffix='_rgb_fisheye',
            csv_rig_rgb_suffix='_rgb_equirect',
            csv_rig_dist_suffix='_dist_fisheye',
            step_size=1,
            max_length=0,
            )

    def __init__(self, 
        # Not obtained from the config directly. 
        dataset_path=None, 
        map_camera_model_raw: dict=None, # Camera models for the raw images.
        map_camera_model: dict=None,     # Camera models for the output images of this dataset.
        map_grid_maker=None,
        # Obtained from the config directly. 
        metadata_fn="metadata.json",
        frame_graph_fn="frame_graph.json",
        conf_dist_blend_func=None, # The blend function used by the sampler operating on distance images.
        conf_dist_lab_tab=None,
        map_camera_frame: dict=None, # Camera frames for the output images.
        cam_key_cv='cv', # The virtual camera for the cost volume.
        true_grid=False,
        align_corners=False,
        align_corners_nearest=False,
        keep_raw_image=False,
        data_keys=['train'],
        mask_path='masks.json', # relative to dataset root
        csv_input_rgb_suffix='_rgb_fisheye',
        csv_rig_rgb_suffix='_rgb_fisheye',
        csv_rig_dist_suffix='_dist_fisheye',
        step_size=1,
        max_length=0,
        ):
        super().__init__()

        assert(dataset_path != None)

        # === Copy the arguments. ===
        self.dataset_path          = dataset_path
        self.map_camera_model_raw  = map_camera_model_raw
        self.map_camera_model      = map_camera_model
        self.map_camera_frame      = map_camera_frame
        self.map_grid_maker        = map_grid_maker
        # Cost volume frame. Must be the same with its cam_key in maps like map_camera_frame.
        self.cam_key_cv            = cam_key_cv
        self.align_corners         = align_corners
        self.align_corners_nearest = align_corners_nearest
        self.use_true_grid         = true_grid # Boolean.
        self.keep_raw_image        = keep_raw_image
        
        # === Frame of the cost volume. ===
        self.frame_cv = self.map_camera_frame[self.cam_key_cv]
        
        # === Distance blending function. ===
        if ( conf_dist_blend_func is None ):
            conf_dist_blend_func = BlendBy2ndOrderGradTorch.get_default_init_args()
        self.dist_blend_func = make_sampler_object( BLEND_FUNCTIONS, conf_dist_blend_func )
        
        # === Distance label table. ===
        if ( conf_dist_lab_tab is None ):
            conf_dist_lab_tab = LinspaceDistLabelTable.get_default_init_args()
        self.dist_lab_tab = make_support_object( DIST_LAB_TAB, conf_dist_lab_tab )

        # === Read the metadata togather with the frame graph. ===
        # self.metar, self.cam_to_camdata, and self.frame_graph will be populated.
        self.read_metadata_and_frame_graph(metadata_fn, frame_graph_fn)

        # Temporary fix for metadata keys make from mixed integers and strings.
        temp_set = set(self.cam_to_camdata.keys())
        temp_set.remove('rig')
        self.cam_int_indices = sorted( temp_set )
        
        # === Other simple initializations. ===
        # Used for resampling distance images.
        self.dist_at_infinity = 1e6
        # ToTensor.
        self.to_tens = ToTensor()

        # === Load masks. ===
        # self.masks will be populated.
        self.masks = None
        self.masks_augmented = None # Used for validation.
        self.read_masks(mask_path)

        # === Samplers. ===
        # Create a sampler if there is a frame associated with the output camera.
        # self.map_sampler will be populated.
        self.create_samplers()

        # Sample the masks, if necessary.
        self.sample_masks()

        # === Rays for building the cost volume. ===
        # Create an array of rays for use in creating the feature-to-cost volume sampling grids.
        self.ray_maker = RayMaker( camera_model=self.map_camera_model[self.cam_key_cv],
                                   frame_name=self.frame_cv )

        # [ 3, N, H, W ]
        self.rays = self.init_sweep_rays_cuda()
        
        # === Grids. ===
        # Will be a Tensor in the shape of [X, 4, 4] where X is the number of cameras.
        self.cam_poses = None
        self.grids = None
        self.grids_valid_masks = None
        # self.grids and self.grids_valid_masks will be populated.
        self.create_grids_from_rays()
        
        # === Filename list column suffix. ===
        # Need to set these values before the first call to len(self).
        self.csv_input_rgb_suffix = csv_input_rgb_suffix
        self.csv_rig_rgb_suffix   = csv_rig_rgb_suffix
        self.csv_rig_dist_suffix  = csv_rig_dist_suffix

        # === Access the filesystem. ===
        # Read data partitions split
        # self.filenames will get populated.
        self.read_filenames(data_keys)

        if step_size > 1:
            new_filenames = dict()
            for k, v in self.filenames.items():
                new_filenames[k] = v[::step_size]
            self.filenames = new_filenames

        if max_length > 0:
            if max_length < len(self):
                new_filenames = dict()
                for k, v in self.filenames.items():
                    new_filenames[k] = v[:max_length]
                self.filenames = new_filenames

                print(f'dataset >>> max_length set to {max_length}, len(self) = {len(self)}. ')
    
    def read_metadata_and_frame_graph(self, metadata_fn, frame_graph_fn):
        self.metar = MetadataReader(self.dataset_path)
        self.metar.read_metadata_and_initialize_dirs(
            os.path.join(self.dataset_path, metadata_fn),
            os.path.join(self.dataset_path, frame_graph_fn),
            create_dirs=False)
        # Convenient aliases.
        self.cam_to_camdata = self.metar.cam_to_camdata
        self.frame_graph    = self.metar.frame_graph
    
    def read_masks(self, mask_path):
        '''
        mask_path is a JSON file with the content similar to the following:

        {
            "masks": [
                {
                    "raw_camera": "cam0",
                    "mask": "valid_mask_0.png"
                },
                {
                    "raw_camera": "cam1",
                    "mask": "valid_mask_1.png"
                },
                {
                    "raw_camera": "cam2",
                    "mask": "valid_mask_2.png"
                }
            ],
            "masks_augmented": [
                {
                    "raw_camera": "cam0",
                    "mask": "aug_mask_0.png"
                },
                {
                    "raw_camera": "cam1",
                    "mask": "aug_mask_1.png"
                },
                {
                    "raw_camera": "cam2",
                    "mask": "aug_mask_2.png"
                }
            ]
        }

        Where "masks_augmented" is optional.
        '''
        mask_path = os.path.join(self.dataset_path, mask_path)
        with open(mask_path, 'r') as fp:
            mask_dict = json.load(fp)

        self.masks = dict()
        for data in mask_dict['masks']:
            fn = os.path.join(self.dataset_path, data['mask'])
            mask = read_mask(fn).astype(np.float32)
            mask = self.to_tens(mask)
            self.masks[data['raw_camera']] = mask

        self.masks_augmented = None
        if 'masks_augmented' in mask_dict:
            self.masks_augmented = dict()
            for data in mask_dict['masks_augmented']:
                fn = os.path.join(self.dataset_path, data['mask'])
                mask = read_mask(fn).astype(np.float32)
                mask = self.to_tens(mask)
                self.masks_augmented[data['raw_camera']] = mask
                            
    def add_csv_to_filenames(self, csv_fn, additional_paths = []):
        with open(csv_fn, 'r') as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                for k_ in row.keys():
                    if k_ not in self.filenames:
                        self.filenames[k_] = list()
                    fn = row[k_]

                    fn = os.path.join(*fn.split('\\')) # handle Windows backslashes
                    full_fn = os.path.join(*additional_paths, fn)
                    self.filenames[k_].append(full_fn)


    def read_filenames(self, data_keys):
        partition_fn = os.path.join(self.dataset_path, 'data_partitions.json')
        with open(partition_fn, 'r') as f:
            partition_conf = json.load(f)

        self.filenames = dict()


        for k in data_keys:
            partition = partition_conf[k]
            
            for env in partition.keys():
                for collection_name in partition[env]:

                    # The base dir of this trajectory.
                    collection_path = os.path.join(self.dataset_path, env, collection_name)

                    if os.path.isdir(collection_path):
                        # Retrieve the filename of the CSV file.
                        traj_dir = collection_path
                        traj_meta_json_fn = os.path.join(traj_dir, 'meta.json')
                        with open(traj_meta_json_fn, 'r') as f_meta:
                            traj_meta = json.load(f_meta)
                        
                        # Read the csv file.
                        csv_fn = os.path.join(traj_dir, traj_meta['selected_file_list'])
                        self.add_csv_to_filenames(csv_fn, additional_paths=[env, collection_name])
                    elif collection_path.endswith(".csv"):
                        self.add_csv_to_filenames(collection_path)
                    else:
                        raise ValueError(f"Encountered Invalid Value in data_partitions.json: {collection_name}")
                        

    def add_frame_to_frame_graph(self, T: FTensor):
        '''
        It is assumed that T.f0 is already in the graph and T.f1 is
        not in the graph yet.
        '''
        
        f1 = RefFrame(T.f1, f'{T.f1}')
        self.frame_graph.add_frame(f1)
        self.frame_graph.add_or_update_pose_edge( T )

    def find_rig_raw_frame(self):
        '''
        This function is a helper for deal with the way the rig camera is specified int the 
        metadata. 
        
        At the moment, there are two scenariios:
        1. The rig camera is one of the input cameras. In this case, there will be one of the 
        cameras that has a "is_rig" key defined in its data dict. Then the raw frame is the 
        "image_frame" key of that camera.
        2. The rig camera is not one of the input cameras. In this case, there will be a key 
        with the name "rig" in the metadata dict. Then the raw frame is the "image_frame" of this
        "rig" key.
        '''
        
        # try:
        #     raw_frame = self.cam_to_camdata['rig']['data']['image_frame']
        # except KeyError:
        #     for _, cdata in self.cam_to_camdata.items():
        #         if 'is_rig' in cdata and cdata['is_rig'] == True:
        #             raw_frame = cdata['data']['image_frame']
        #             return raw_frame
        
        # raise KeyError(
        #     f'No rig camera found in the metadata.'
        #     f'metadata is\n{self.cam_to_camdata}')
        
        # TODO: Hard coded for now.
        return 'rif'

    def create_single_sampler(self, 
            cam_key, frame_raw, frame, camera_model_raw):
        # If the output frame is None or empty string, then no sampler is needed.
        # Use the raw camera model as the output camera model.
        if frame is None or frame == '':
            if cam_key not in self.map_camera_model:
                print(f'{cam_key} is not in self.map_camera_model. Use camera_model_raw instead. ')
                self.map_camera_model[cam_key] = camera_model_raw
            
            R_raw_output = f_ones(3, f0=frame_raw, f1=frame_raw)
            self.map_sampler[cam_key] = NoOpSampler(
                camera_model_raw,
                R_raw_fisheye=R_raw_output,
                convert_output=False)
        else:
            # Query the frame graph for the transform between raw_frame and frame.
            T_raw_output = self.frame_graph.query_transform( f0=frame_raw, f1=frame )

            # Create a sampler for the output camera using the rotation.
            # Need to set convert_output=False to have better efficiency
            self.map_sampler[cam_key] = ImageSampler(
                    camera_model_raw, 
                    self.map_camera_model[cam_key], 
                    R_raw_fisheye=T_raw_output.rotation,
                    convert_output=False)
            
            # We need to update the frame graph.
            self.add_frame_to_frame_graph(T_raw_output)

        # This is required since pytorch doesn't like sharing CUDA tensors across processes.
        self.map_sampler[cam_key].device = 'cpu'

    def create_samplers(self):
        self.map_sampler = dict()
        
        # map_camera_frame should contain frames like 'rig' and 'cv'.
        for cam, frame in self.map_camera_frame.items():
            if cam == 'cv':
                continue
            
            if cam == 'rig':
                frame_raw = self.find_rig_raw_frame()
                camera_model_raw = self.map_camera_model_raw['rig']
            else:
                # Get the frame of the raw camera. Recorded in the metadata.
                cam_idx = cam_idx_from_cam_key(cam)
                frame_raw = self.cam_to_camdata[cam_idx]['data']['image_frame']
                camera_model_raw = self.map_camera_model_raw[cam]

            self.create_single_sampler( cam, frame_raw, frame, camera_model_raw )
        
        # The cv sampler for sampling the true dist.
        self.create_single_sampler( 
            'cv', 
            self.find_rig_raw_frame(), 
            self.map_camera_frame['cv'],
            self.map_camera_model_raw['rig'])

    def sample_masks(self):
        for cam, mask in self.masks.items():
            # Test if we need to sample.
            sampler = self.map_sampler[cam]
            if isinstance( sampler, NoOpSampler ):
                continue

            mask = mask*255
            
            mask, _ = sampler(mask, invalid_pixel_value=0)
            mask[mask > 0] = 1.0

            # NOTE: The sampler is configured to skip the output wrapping.
            # And the sampled mask is a torch.Tensor with the batch dimension.
            self.masks[cam] = mask.squeeze(0)

    def init_sweep_rays_cuda(self, true_dist=None):
        # Prepare memory for a [3, N, Ho, Wo] grid for the 
        # spherical rays. N is 1 for true sweep.
        if true_dist is not None:
            # true_dist is a Torch tensor with the shape of [1, H, W].
            rays, valid_mask = self.ray_maker.make_rays( grid_distance=true_dist)
        else:
            rays, valid_mask = self.ray_maker.make_rays( candidates=self.dist_lab_tab.dist() )

        # TODO: here, valid_mask is not used.
        return rays

    def update_rays_according_2_true_dist(self, true_dist):
        # Re-sample the true distance with the cv sampler.
        sampler = self.map_sampler[self.cam_key_cv]
        true_G_label, _ = sampler(
            true_dist, 
            interpolation=INTER_BLENDED,
            invalid_pixel_value=self.dist_at_infinity,
            blend_func=self.dist_blend_func)
        true_G_label = true_G_label.squeeze(0)
        
        # Sweep with thr true distance.
        self.rays = self.init_sweep_rays_cuda(true_dist=true_G_label)

    def make_sweep_grid_cuda(self, 
        grid_maker,
        pose: FTensor, 
        pose_error: FTensor=None):

        '''
        https://docs.google.com/presentation/d/1TK_iSx5yYuJqvZ2fxQuvog4-87IDOr_9mho7AGR6K9A/edit#slide=id.ge5b0b7a031_0_0

        grid_maker: A callable object that takes in an array 3D points.
        pose (FTensor): The pose of the caemra.
        pose_error (FTensor): The pose error got added to the pose. Must be measured in the panorama frame.
        '''
              
        # Apply error.
        if pose_error is not None:
            pose = pose_error @ pose

        #Testing Rotating Sampling Grid Pose 
        #************** Temporary **************
        # correction = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
        #                            [0.0, 0.0, 1.0, 0.0], 
        #                            [0.0, -1.0, 0.0, 0.0],
        #                            [0.0, 0.0, 0.0, 1.0]])

        # pose = pose @ correction

        # We need the inverse transform.
        inv_pose = pose.inverse()

        # Transform the rays.
        # The rays are in the cv frame.
        rays = transform_3D_points_torch( 
            inv_pose.unsqueeze(0), self.rays.unsqueeze(0) )

        # Make the sampling grid.
        grid, valid_mask = grid_maker.make_grid(rays)

        # Correct the dimension. 
        # TODO: Really, we need to move the grid computation out from dataset.
        grid = grid.squeeze(0).cpu() # The batch dimension is 1.
        valid_mask = valid_mask.squeeze(0).cpu()

        return grid, valid_mask

    def get_pose_from_sampler(self, cam_key):
        sampler = self.map_sampler[cam_key]
        
        # Get the frame.
        cif = sampler.R_raw_fisheye.f1
        
        # Query the transform.
        # TODO: double check self.frame_cv is the correct frame.
        return self.frame_graph.query_transform( f0=self.frame_cv, f1=cif )

    def create_grids_from_rays(self):
        cam_poses = []
        grids = []
        valid_masks = []
        for k in self.cam_int_indices:
            # if k != "rig":
            cam_key = f'cam{k}'

            # Get the pose of the image/camera w.r.t. the rig.
            cam_pose = self.get_pose_from_sampler(cam_key)

            # Save the pose.
            # Because self.cam_int_indices is sorted.
            cam_poses.append( cam_pose.tensor() )

            # Get the grid maker.
            grid_maker = self.map_grid_maker[cam_key]

            # Cost volume sampling grid.
            t_grid, t_valid_mask = self.make_sweep_grid_cuda(grid_maker, cam_pose)

            grids.append(t_grid)
            valid_masks.append(t_valid_mask)

        self.grids = torch.stack(grids)
        self.grids_valid_masks = torch.stack(valid_masks)

        self.cam_poses = torch.stack( cam_poses, dim=0 ) # First dim is number of cameras.

    def read_rgb_as_tensor(self, fn, cam_key):
        img_raw = read_image(fn)
        
        # See if we need to aument the raw image using the augmented mask.
        if self.masks_augmented is not None and cam_key in self.masks_augmented:
            img_in = self.to_tens(img_raw) * self.masks_augmented[cam_key]
        else:
            img_in = img_raw

        # Re-sample the image.
        sampler = self.map_sampler[cam_key]
        # Return value is a Tensor. 
        # NOTE: I am ignoring the valid mask.
        img_t, _ = sampler(img_in)
        
        # img_t has 4 dimensions.
        return img_t.squeeze(0), img_raw

    def compose_rgb_input_fn(self, i, cam_key):
        # Compose the file path and read the image.
        fn = os.path.join(
            self.dataset_path,
            self.filenames[f'{cam_key}{self.csv_input_rgb_suffix}'][i] )

        return fn

    def read_input_rgb(self, i, cam_key):
        # Compose the file path and read the image.
        fn = self.compose_rgb_input_fn(i, cam_key)
        return self.read_rgb_as_tensor(fn, cam_key)

    def read_true_rgb(self, i):
        # # For debugging with the panorama images generated by UE.
        # # Compose the file path and read the image.
        # fn = os.path.join(
        #     self.dataset_path,
        #     self.filenames['rig_rgb_pano'][i] )
        
        # TODO: Fix this! This is hard-coded.
        fn = os.path.join(
            self.dataset_path,
            self.filenames[f'rig{self.csv_rig_rgb_suffix}'][i] )
        
        return self.read_rgb_as_tensor(fn, 'rig')

    # TODO: This function is hard-coded for cam0.
    def read_dist(self, i, csv_header_prefix, csv_header_suffix):
        # grd_pth = os.path.join(
        #     self.dataset_path, 
        #     self.filenames[f'rig{self.csv_rig_dist_suffix}'][i] )
        # TODO: Fix this!
        grd_pth = os.path.join(
            self.dataset_path, 
            self.filenames[f'{csv_header_prefix}{csv_header_suffix}'][i] )
        true_dist = read_compressed_float(grd_pth)
        return true_dist

    def read_true_inv_dist(self, i, sampler_key, csv_header_prefix, csv_header_suffix):
        # Read the raw ground truth distance values.
        true_dist = self.read_dist(i, csv_header_prefix, csv_header_suffix) # NumPy array.

        # Resample the true distance image and convert it to a tensor.
        sampler = self.map_sampler[sampler_key]
        true_dist_t, valid_mask = sampler(
            true_dist, 
            interpolation=INTER_BLENDED, 
            invalid_pixel_value=self.dist_at_infinity,
            blend_func=self.dist_blend_func)

        # Calculate and resize inverse distance index. 
        inv_dist_idx = self.dist_lab_tab.dist_2_inv_idx( true_dist_t )
        inv_dist_idx = inv_dist_idx.squeeze(0)

        # Handle the extra dimention after the sampling procedure.
        true_dist_t = true_dist_t.squeeze(0)

        # TODO: valid_mask is NumPy array.
        valid_mask = torch.from_numpy(valid_mask)

        return inv_dist_idx, true_dist_t, valid_mask, true_dist

    def __getitem__(self, i):
        assert ( 0 <= i < len(self) ), \
            f'Index {i} is out of range [0-{len(self)-1}]. '

        # Read the individual input images.
        imgs     = []
        imgs_raw = []
        masks    = []
        # TODO: check for unordered dict bug in other places. 
        # for k in self.cam_to_camdata.keys():
        for k in self.cam_int_indices:
            # if k == "rig":
            #     continue

            cam_key = f'cam{k}'

            # Read the input image. NOTE: The valid mask is ignored.
            t_img, img_raw = self.read_input_rgb( i, cam_key )

            imgs.append(t_img)
            imgs_raw.append(img_raw)
            masks.append(self.masks[cam_key])
                
        # torch.stack() adds a new dimension.
        imgs  = torch.stack(imgs)
        masks = torch.stack(masks)

        # Read the ground truth RGB image.
        rig_rgb, rig_rgb_raw = self.read_true_rgb(i)
        # TODO: Potential bug! Assuming cam0 is the rig!
        # rig_rgb     = imgs[0]
        # rig_rgb_raw = imgs_raw[0]
        rig_mask    = masks[0]

        # Read the ground truth distance values and inverse it.
        # TODO: valid_mask_dist is ignored.
        inv_dist_idx, true_dist, valid_mask_dist, true_dist_raw = \
            self.read_true_inv_dist(i, 
                                    sampler_key='rig',
                                    csv_header_prefix='rig', 
                                    csv_header_suffix=self.csv_rig_dist_suffix)

        #If using the ground truth grid, generate the spherical rays to be used for each grid
        if self.use_true_grid:
            self.update_rays_according_2_true_dist(true_dist_raw)
            self.create_grids_from_rays()

        # TODO: debug use.
        fn_rgb_cam0 = self.compose_rgb_input_fn(i, 'cam0')

        # Put everything into a dictionary.
        data_entry = {
            'sel_id': torch.Tensor([i]).to(dtype=torch.int32),
            'fn_rgb_cam0': fn_rgb_cam0,
            'imgs': imgs,
            'masks': masks,
            'grids': self.grids,
            'grid_masks': self.grids_valid_masks,
            'cam_poses': self.cam_poses,
            'inv_dist_idx': inv_dist_idx,
            'rig_rgb': rig_rgb,
            'rig_mask': rig_mask,
        }
        
        if self.keep_raw_image:
            data_entry['imgs_raw'] = imgs_raw
            data_entry['rig_rgb_raw'] = rig_rgb_raw
            data_entry['rig_dist_raw'] = true_dist_raw
        
        return data_entry
    
    def __len__(self):
        # TODO: Fix this! The logic around rig should be consistent with other places.
        return len(self.filenames[f'rig{self.csv_rig_dist_suffix}'])
    
    def get_num_cams(self):
        return self.metar.num_cams
    
@register(DATASET)
class FullDistDataset(MultiViewCameraModelDataset):
    
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            metadata_fn="metadata.json",
            frame_graph_fn="frame_graph.json",
            conf_dist_blend_func=None, # The blend function used by the sampler operating on distance images.
            conf_dist_lab_tab=None,
            map_camera_frame=None, # Camera frames for the output images. Should be a dict.
            cam_key_cv='cv', # The virtual camera for the cost volume.
            true_grid=False,
            align_corners=False,
            align_corners_nearest=False,
            keep_raw_image=False,
            data_keys=['train'],
            mask_path='masks.json', # relative to dataset root
            csv_input_rgb_suffix='_rgb_fisheye',
            csv_rig_rgb_suffix='_rgb_fisheye',
            csv_rig_dist_suffix='_dist_fisheye',
            step_size=1,
            max_length=0, )
        
    def __init__(self,
        # Not obtained from the config directly. 
        dataset_path=None, 
        map_camera_model_raw: dict=None, # Camera models for the raw images.
        map_camera_model: dict=None,     # Camera models for the output images of this dataset.
        map_grid_maker=None,
        transforms=None,
        # Obtained from the config directly. 
        metadata_fn="metadata.json",
        frame_graph_fn="frame_graph.json",
        conf_dist_blend_func=None, # The blend function used by the sampler operating on distance images.
        conf_dist_lab_tab=None,
        map_camera_frame: dict=None, # Camera frames for the output images.
        cam_key_cv='cv', # The virtual camera for the cost volume.
        true_grid=False,
        align_corners=False,
        align_corners_nearest=False,
        keep_raw_image=False,
        data_keys=['train'],
        mask_path='masks.json', # relative to dataset root
        csv_input_rgb_suffix='_rgb_fisheye',
        csv_rig_rgb_suffix='_rgb_fisheye',
        csv_rig_dist_suffix='_dist_fisheye',
        step_size=1,
        max_length=0,):
        
        super().__init__(
            dataset_path=dataset_path, 
            map_camera_model_raw=map_camera_model_raw, # Camera models for the raw images.
            map_camera_model=map_camera_model,     # Camera models for the output images of this dataset.
            map_grid_maker=map_grid_maker,
            transforms=transforms,
            metadata_fn=metadata_fn,
            frame_graph_fn=frame_graph_fn,
            conf_dist_blend_func=conf_dist_blend_func, # The blend function used by the sampler operating on distance images.
            conf_dist_lab_tab=conf_dist_lab_tab,
            map_camera_frame=map_camera_frame, # Camera frames for the output images.
            cam_key_cv=cam_key_cv, # The virtual camera for the cost volume.
            true_grid=true_grid,
            align_corners=align_corners,
            align_corners_nearest=align_corners_nearest,
            keep_raw_image=keep_raw_image,
            data_keys=data_keys,
            mask_path=mask_path, # relative to dataset root
            csv_input_rgb_suffix=csv_input_rgb_suffix,
            csv_rig_rgb_suffix=csv_rig_rgb_suffix,
            csv_rig_dist_suffix=csv_rig_dist_suffix,
            step_size=step_size,
            max_length=max_length )
        
    def __getitem__(self, i):
        assert ( 0 <= i < len(self) ), \
            f'Index {i} is out of range [0-{len(self)-1}]. '

        # Read the individual input images.
        imgs     = []
        imgs_raw = []
        dist_inv = []
        dist_raw = []
        masks    = []
        # TODO: check for unordered dict bug in other places. 
        # for k in self.cam_to_camdata.keys():
        for k in self.cam_int_indices:
            # if k == "rig":
            #     continue

            cam_key = f'cam{k}'

            # Read the input image. NOTE: The valid mask is ignored.
            t_img, img_raw = self.read_input_rgb( i, cam_key )

            imgs.append(t_img)
            imgs_raw.append(img_raw)
            masks.append(self.masks[cam_key])
            
            # Read the ground truth distance values and inverse it.
            # TODO: valid_mask_dist is ignored.
            # TODO: self.csv_rig_dist_suffix is used for all cameras.
            inv_dist_idx, true_dist, valid_mask_dist, true_dist_raw = \
                self.read_true_inv_dist(i, 
                                        sampler_key=cam_key,
                                        csv_header_prefix=cam_key, 
                                        csv_header_suffix=self.csv_rig_dist_suffix)
                
            dist_inv.append(inv_dist_idx)
            dist_raw.append(true_dist_raw)
                
        # torch.stack() adds a new dimension.
        imgs     = torch.stack(imgs)
        masks    = torch.stack(masks)
        dist_inv = torch.stack(dist_inv)

        # Read the ground truth RGB image.
        rig_rgb, rig_rgb_raw = self.read_true_rgb(i)
        # TODO: Potential bug! Assuming cam0 is the rig!
        rig_mask = masks[0]

        # TODO: Assuming rig is cam0!
        inv_dist_idx  = dist_inv[0]
        true_dist_raw = dist_raw[0]

        #If using the ground truth grid, generate the spherical rays to be used for each grid
        if self.use_true_grid:
            self.update_rays_according_2_true_dist(true_dist_raw)
            self.create_grids_from_rays()

        fn_rgb_cam0 = self.compose_rgb_input_fn(i, 'cam0')

        # Put everything into a dictionary.
        data_entry = {
            'sel_id': torch.Tensor([i]).to(dtype=torch.int32),
            'fn_rgb_cam0': fn_rgb_cam0,
            'imgs': imgs,
            'masks': masks,
            'dist_inv': dist_inv,
            'grids': self.grids,
            'grid_masks': self.grids_valid_masks,
            'cam_poses': self.cam_poses,
            'inv_dist_idx': inv_dist_idx,
            'rig_rgb': rig_rgb,
            'rig_mask': rig_mask,
        }
        
        if self.keep_raw_image:
            data_entry['imgs_raw'] = imgs_raw
            data_entry['dist_raw'] = dist_raw
            data_entry['rig_rgb_raw'] = rig_rgb_raw
            data_entry['rig_dist_raw'] = true_dist_raw
        
        return data_entry
    

@register(DATASET)
class ROSDroneImagesDataset(MultiViewCameraModelDataset):
    
    # TODO: get_default_init_args() has inconsistent arguments with the __init__() function.
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            order=[0,1,2], #Can be any list of indices to swap the csv columns as long as the list is exactly as long as the number o columns in the read csvs
            metadata_fn="metadata.json",
            frame_graph_fn="frame_graph.json",
            conf_dist_blend_func=None, # The blend function used by the sampler operating on distance images.
            conf_dist_lab_tab=None,
            map_camera_frame=None, # Camera frames for the output images. Should be a dict.
            cam_key_cv='cv', # The virtual camera for the cost volume.
            true_grid=False,
            align_corners=False,
            align_corners_nearest=False,
            keep_raw_image=False,
            data_keys=['train'],
            mask_path='masks.json', # relative to dataset root
            csv_input_rgb_suffix='_rgb_fisheye',
            csv_rig_rgb_suffix='_rgb_fisheye',
            csv_rig_dist_suffix='_dist_fisheye',)
        
    def __init__(self,
        # Not obtained from the config directly. 
        dataset_path=None, 
        map_camera_model_raw: dict=None, # Camera models for the raw images.
        map_camera_model: dict=None,     # Camera models for the output images of this dataset.
        map_grid_maker=None,
        # Obtained from the config directly. 
        metadata_fn="metadata.json",
        frame_graph_fn="frame_graph.json",
        conf_dist_blend_func=None, # The blend function used by the sampler operating on distance images.
        conf_dist_lab_tab=None,
        map_camera_frame: dict=None, # Camera frames for the output images.
        cam_key_cv='cv', # The virtual camera for the cost volume.
        true_grid=False,
        align_corners=False,
        align_corners_nearest=False,
        keep_raw_image=False,
        data_keys=['train'],
        order: Sequence=[1, 0, 2], # TODO: default value is inconsistent with get_default_init_args().
        num_samples: int = 10, # TODO: change it to max_length. 
        step_size: int = 20,
        mask_path='masks.json', # relative to dataset root
        csv_input_rgb_suffix='',
        csv_rig_rgb_suffix=None,
        csv_rig_dist_suffix=None):

        super().__init__(
            dataset_path=dataset_path, 
            map_camera_model_raw=map_camera_model_raw, # Camera models for the raw images.
            map_camera_model=map_camera_model,     # Camera models for the output images of this dataset.
            map_grid_maker=map_grid_maker,
            metadata_fn=metadata_fn,
            frame_graph_fn=frame_graph_fn,
            conf_dist_blend_func=conf_dist_blend_func, # The blend function used by the sampler operating on distance images.
            conf_dist_lab_tab=conf_dist_lab_tab,
            map_camera_frame=map_camera_frame, # Camera frames for the output images.
            cam_key_cv=cam_key_cv, # The virtual camera for the cost volume.
            true_grid=true_grid,
            align_corners=align_corners,
            align_corners_nearest=align_corners_nearest,
            keep_raw_image=keep_raw_image,
            data_keys=data_keys,
            mask_path=mask_path, # relative to dataset root
            csv_input_rgb_suffix=csv_input_rgb_suffix,
            csv_rig_rgb_suffix=csv_rig_rgb_suffix,
            csv_rig_dist_suffix=csv_rig_dist_suffix,
            step_size=step_size,
            max_length=num_samples )
        
        self.order = order
        self.num_samples = num_samples
        self.step_size = step_size

        # print("Limiting ROSDroneImage List:")
        # for _k in self.filenames:
        #     stepped_list = self.filenames[_k][0:-1:step_size]
        #     if self.num_samples > 0:
        #         self.filenames[_k] = stepped_list[:num_samples]
        #     else:
        #         self.filenames[_k] = stepped_list
        #     print(f"{_k}: {len(self.filenames[_k])}")

    # def read_rgb_as_tensor(self, fn, cam_key):
    #     img_raw = read_image(fn)
        
    #     # Re-sample the image.
    #     sampler = self.map_sampler[cam_key]
    #     # Return value is a Tensor. 
    #     # NOTE: I am ignoring the valid mask.
    #     img_t, _ = sampler(img_raw)
        
    #     # img_t has 4 dimensions.
    #     return img_t.squeeze(0), img_raw

    def read_input_rgb(self, i, cam_key):
        # Compose the file path and read the image.

        fn = os.path.join(
            self.dataset_path,
            self.filenames[f'{cam_key}{self.csv_input_rgb_suffix}'][i] )
        
        return self.read_rgb_as_tensor(fn, cam_key)

    def add_csv_to_filenames(self, csv_fn, additional_paths = []):
        with open(csv_fn, 'r') as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                for k_ in row.keys():
                    if k_ not in self.filenames:
                        self.filenames[k_] = list()
                    fn = row[k_]

                    fn = os.path.join(*fn.split('\\')) # handle Windows backslashes
                    full_fn = os.path.join(*additional_paths, fn)
                    full_fn = full_fn[1:] if full_fn[0] == "/" else full_fn

                    self.filenames[k_].append(full_fn)

    def create_samplers(self):
        self.map_sampler = dict()
        
        # map_camera_frame should contain frames like 'rig' and 'cv'.
        for cam, frame in self.map_camera_frame.items():
            if cam == 'cv':
                continue
            
            if cam == 'rig':
                continue

            else:
                # Get the frame of the raw camera. Recorded in the metadata.
                cam_idx = cam_idx_from_cam_key(cam)
                frame_raw = self.cam_to_camdata[cam_idx]['data']['image_frame']
                camera_model_raw = self.map_camera_model_raw[cam]

            self.create_single_sampler( cam, frame_raw, frame, camera_model_raw )


    def __len__(self):
        # TODO: Fix this! The logic around rig should be consistent with other places.
        return len(self.filenames[list(self.filenames.keys())[0]])

    def __getitem__(self, i):
        assert ( 0 <= i < len(self) ), \
            f'Index {i} is out of range [0-{len(self)-1}]. '

        # Read the individual input images.
        imgs     = []
        imgs_raw = []
        masks    = []
        # TODO: check for unordered dict bug in other places. 
        # for k in self.cam_to_camdata.keys():
        for k in self.cam_int_indices:
            # if k == "rig":
            #     continue

            cam_key = f'cam{k}'

            # Read the input image. NOTE: The valid mask is ignored.
            t_img, img_raw = self.read_input_rgb( i, cam_key )

            imgs.append(t_img)
            imgs_raw.append(img_raw)
            masks.append(self.masks[cam_key])
        
        # Rearrange images according to the correct order of the real images
        imgs  = [imgs[i] for i in self.order]
        masks = [masks[i] for i in self.order]
        imgs_raw = [imgs_raw[i] for i in self.order]

        # torch.stack() adds a new dimension.
        imgs     = torch.stack(imgs)
        masks    = torch.stack(masks)

        # TODO: Potential bug! Assuming cam0 is the rig!
        rig_mask = masks[0]
        rig_rgb  = imgs[0] #NOT CORRECT

        fn_rgb_cam0 = self.compose_rgb_input_fn(i, 'cam0')

        # Put everything into a dictionary.
        data_entry = {
            'sel_id': torch.Tensor([i]).to(dtype=torch.int32),
            'fn_rgb_cam0': fn_rgb_cam0,
            'imgs': imgs,
            'masks': masks,
            'grids': self.grids,
            'grid_masks': self.grids_valid_masks,
            'cam_poses': self.cam_poses,
            'rig_mask': rig_mask
        }
        
        if self.keep_raw_image:
            data_entry['imgs_raw'] = imgs_raw
        
        return data_entry
    
    