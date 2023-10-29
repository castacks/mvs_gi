
# System.
import json
import numpy as np

# PyTorch and torchvision.
# import torch
# import torch.nn.functional as F

# Local.
from ...mvs_utils.shape_struct import ShapeStruct
from ...mvs_utils.camera_models import ( CAMERA_MODELS, Equirectangular )
from ...mvs_utils.camera_models import make_object as make_camera_model
from .torch_cuda_sweep import CameraModelGridMaker

from ..register import ( GRID_MAKER_BUILDER, register )

class GridMakerBuilder(object):
    def __init__(self):
        super().__init__()

        self.grid_maker_map = None

def read_manifest(fn):
    with open(fn, 'r') as fp:
        manifest_obj = json.load(fp)

    # Get the camera_models.
    camera_models_dict = manifest_obj['camera_models']

    # Get the samplers.
    samplers_list = manifest_obj['samplers']
    samplers_dict = dict()
    for sampler_spec in samplers_list:
        if sampler_spec['mvs_main_cam_model_for_cam']:
            samplers_dict[ sampler_spec['mvs_cam_key'] ] = sampler_spec['sampler']

    return camera_models_dict, samplers_dict

class GridMakerFromManifest(GridMakerBuilder):
    def __init__(self, fn):
        super().__init__()

        self.camera_model_map = None
        self._camera_model_map_ori = None

        self.parse_manifest(fn)

    def parse_manifest(self, fn):
        raise NotImplementedError()
    
    @property
    def camera_model_map_ori(self):
        raise NotImplementedError()

@register(GRID_MAKER_BUILDER)
class GridMakerFromManifestPlain(GridMakerFromManifest):
    def __init__(self, fn):
        super().__init__(fn)

    def parse_manifest(self, fn):
        # Read the manifest.
        camera_models_dict, samplers_dict = read_manifest(fn)

        # Make the camera models.
        camera_models = dict()
        for cam_model_key, cam_model_dict in camera_models_dict.items():
            camera_model = make_camera_model(CAMERA_MODELS, cam_model_dict)
            camera_model.in_to_tensor = False
            camera_models[cam_model_key] = camera_model

        # Make the grid makers.
        self.grid_maker_map = dict()
        self.camera_model_map = dict()
        for cam_id, sampler_dict in samplers_dict.items():
            cam_model_key = sampler_dict['cam_model_key']
            self.grid_maker_map[cam_id] = CameraModelGridMaker( camera_models[cam_model_key] )
            self.camera_model_map[cam_id] = camera_models[cam_model_key]

    @property
    def camera_model_map_ori(self):
        return self.camera_model_map

@register(GRID_MAKER_BUILDER)
class EquirectGridMakerFromManifest(GridMakerFromManifest):
    def __init__(self, fn: str, ss_equirect: ShapeStruct):
        '''
        ss_equirect: the new shape for the equirectangular version of the fisheye camera.
        '''
        self.ss = ShapeStruct.read_shape_struct(ss_equirect) \
            if isinstance(ss_equirect, dict) \
            else ss_equirect
        
        super().__init__(fn)

    def parse_manifest(self, fn):
        # Read the manifest.
        camera_models_dict, samplers_dict = read_manifest(fn)

        # Make the camera models.
        camera_models = dict()
        camera_models_ori = dict()
        for cam_model_key, cam_model_dict in camera_models_dict.items():
            # Create a camera model for the original fisheye camera.
            camera_model_ori = make_camera_model(CAMERA_MODELS, cam_model_dict)
            camera_model_ori.in_to_tensor = True
            # Prepare for creating the equirectangular verion on CPU.
            # camera_model_ori.out_to_numpy = True # True is used before changing to the new pixel coordinate definition.
            camera_model_ori.out_to_numpy = False
            camera_models_ori[cam_model_key] = camera_model_ori

            # Create a new camra model for the equirectangular version of the original camera.
            # If this camera is the rig, then this will be just a dummy object and will not 
            # be used later.
            camera_model = Equirectangular(
                self.ss, 
                latitude_span=( -np.pi/2, 0 ),
                open_span=False,
                in_to_tensor=True, 
                out_to_numpy=False)
            camera_models[cam_model_key] = camera_model

        # Make the grid makers.
        self.grid_maker_map = dict()
        self.camera_model_map = dict()
        self._camera_model_map_ori = dict()
        for cam_id, sampler_dict in samplers_dict.items():
            if cam_id == 'rig':
                continue

            cam_model_key = sampler_dict['cam_model_key']
            self.grid_maker_map[cam_id] = CameraModelGridMaker( camera_models[cam_model_key] )
            self.camera_model_map[cam_id] = camera_models[cam_model_key]
            self._camera_model_map_ori[cam_id] = camera_models_ori[cam_model_key]

    @property
    def camera_model_map_ori(self):
        return self._camera_model_map_ori
    
@register(GRID_MAKER_BUILDER)
class SurrogateCameraModelGridMaker(GridMakerFromManifest):
    def __init__(self, fn: str, conf_map_surrogate_camera_model: dict):
        '''
        self.map_surrogate_camera_model: a dict recording the mapping from a cam_model_key to a 
        surrogate camera model for creating the grid. A cam_model_key is a key in the \
        "camera_models" section of the manifest file.
        '''
        
        self.map_surrogate_camera_model = dict()
        for cam_key, camera_model_spec in conf_map_surrogate_camera_model.items():
            self.map_surrogate_camera_model[cam_key] = \
                make_camera_model(CAMERA_MODELS, camera_model_spec)
        
        super().__init__(fn)

    def parse_manifest(self, fn):
        # Read the manifest.
        camera_models_dict, samplers_dict = read_manifest(fn)

        # Collect all the camera models.
        camera_models_ori = dict()
        for cam_model_key, cam_model_dict in camera_models_dict.items():
            # Create a camera model for the original camera.
            camera_model_ori = make_camera_model(CAMERA_MODELS, cam_model_dict)
            camera_model_ori.in_to_tensor = True
            # Prepare for creating the equirectangular verion on CPU.
            # camera_model_ori.out_to_numpy = True # True is used before changing to the new pixel coordinate definition.
            camera_model_ori.out_to_numpy = False
            camera_models_ori[cam_model_key] = camera_model_ori

        # Create the grid makers.
        self.grid_maker_map = dict()
        self.camera_model_map = dict()
        self._camera_model_map_ori = dict()
        for cam_key, sampler_dict in samplers_dict.items():
            # TODO: This assumes that the rig is also pre-processed.
            # if cam_key == 'rig':
            #     continue

            cam_model_key = sampler_dict['cam_model_key']
            if cam_key in self.map_surrogate_camera_model:
                grid_maker_camera_model = self.map_surrogate_camera_model[cam_key]
            else:
                grid_maker_camera_model = camera_models_ori[cam_model_key]

            self.grid_maker_map[cam_key] = CameraModelGridMaker( grid_maker_camera_model )
            self.camera_model_map[cam_key] = grid_maker_camera_model
            self._camera_model_map_ori[cam_key] = camera_models_ori[cam_model_key]

    @property
    def camera_model_map_ori(self):
        return self._camera_model_map_ori
