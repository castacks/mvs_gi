
import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..')

if _TOP_PATH not in sys.path:
    sys.path.insert( 0, _TOP_PATH)
    for i, p in enumerate(sys.path):
        print(f'{i}: {p}')

import copy
import numpy as np

import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit # This is needed for initializing CUDA driver.

import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Local packages.
from .inference_class import InferenceProxy

class InferenceTRT(InferenceProxy):
    def __init__(self, 
                 engine_fn,
                 argv, 
                 preprocessed_config=False,
                 sample_input=False,
                 debug=False):
        super().__init__(argv=argv, preprocessed_config=preprocessed_config, debug=debug)
        
        self.sample_input = sample_input
        self.map_samplers = None
        if self.sample_input:
            self.prepare_samplers()

        self.trt_logger = trt.Logger()

        # Override parent's model member.
        self.grid_masks = self.grid_masks.to(dtype=torch.float32)
        self.masks      = self.masks.to(dtype=torch.float32)

        self.model = None
        self.context = None
        self.output_size = None
        self.output_buf = None
        self.output_mem = None
        self.get_model(model_config=engine_fn)

        self.stream = torch.cuda.Stream()

    def prepare_samplers(self):
        # Copy the samplers from the dataset.
        self.map_samplers = copy.deepcopy(self.dataloader.dataset.map_sampler)
        # Push the samplers to GPU.
        for _, sampler in self.map_samplers.items():
            sampler.device = 'cuda'

    # Required override.
    def get_model(self, model_config, **kwargs):
        engine_fn = model_config
        assert os.path.exists(engine_fn)

        print( f"Reading engine from file {engine_fn}" )
        with open(engine_fn, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.model = runtime.deserialize_cuda_engine( f.read() )

        self.context = self.model.create_execution_context()

        imgs_bind_idx       = self.model.get_binding_index('imgs')
        grids_bind_idx      = self.model.get_binding_index('grids')
        grid_masks_bind_idx = self.model.get_binding_index('grid_masks')
        masks_bind_idx      = self.model.get_binding_index('masks')

        self.context.set_binding_shape(imgs_bind_idx, self.dummy_imgs.shape)
        self.context.set_binding_shape(grids_bind_idx, self.grids.shape)
        self.context.set_binding_shape(grid_masks_bind_idx, self.grid_masks.shape)
        self.context.set_binding_shape(masks_bind_idx, self.masks.shape)

        output_bind_idx = self.model.get_binding_index('inv_dist')
        # ( 1, 1, H, W )
        self.output_size = tuple(self.context.get_binding_shape(output_bind_idx))
        self.output_buf = torch.zeros(self.output_size, dtype=torch.float32).cuda().contiguous()
        self.output_mem = self.output_buf.data_ptr()

    # Override.
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, torch.Tensor):
            # NOTE: This is the debug branch.
            t_imgs = imgs.cuda()
        else:
            if isinstance(imgs, list):
                imgs = np.stack(imgs, axis=0)
            # Convert the input from NumPy to Tensor.
            t_imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).cuda().unsqueeze(0)
            if t_imgs.dtype == torch.uint8:
                t_imgs = t_imgs.float() / 255.0

        if self.sample_input:
            sampled_imgs = []

            # NOTE: Assume batch size is always 1.
            assert t_imgs.shape[0] == 1, f'Batch size must be 1. Got {t_imgs.shape[0]} instead. '
            for i, img in enumerate(t_imgs[0, ...]):
                key = f'cam{i}'
                sampler = self.map_samplers[key]
                # NOTE: The second return value is the mask.
                sampled, _ = sampler(img)
                # NOTE: sampler(img) returns the first tensor with the batch dimension.
                sampled_imgs.append( sampled )

            t_imgs = torch.cat(sampled_imgs, dim=0).unsqueeze(0)

        return t_imgs
    
    # Override.
    def postprocess_imgs(self, inv_dist):
        with torch.no_grad():
            inv_dist = (self.output_buf / self.bf).detach().to(device='cpu').numpy()
            inv_dist = inv_dist[0, 0]
        return inv_dist

    def inference(self, imgs, input_dict=None):
        self.context.execute_async_v2(
            bindings=[
                imgs.data_ptr(),
                self.grids.data_ptr(),
                self.grid_masks.data_ptr(),
                self.masks.data_ptr(),
                self.output_mem,
            ],
            stream_handle=self.stream.cuda_stream
        )

        self.stream.synchronize()

        return self.output_buf
