import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np
import tensorrt as trt
import time
import pycuda.driver as cuda
import pycuda.autoinit # This is needed for initializing CUDA driver.

TRT_LOGGER = trt.Logger()

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():

    working_dir = 'code_release_202310_trt/artifacts_op13_gi/config103'

    sample = np.load( os.path.join( working_dir, 'test_sample.npz' ) )
    imgs       = sample['imgs']
    grids      = sample['grids']
    grid_masks = sample['grid_masks']
    masks      = sample['masks'] 

    engine = load_engine( os.path.join( working_dir, 'model_backport_x86.engine' ) )
    context = engine.create_execution_context()
    
    imgs_bind_idx = engine.get_binding_index('imgs')
    grids_bind_idx = engine.get_binding_index('grids')
    grid_masks_bind_idx = engine.get_binding_index('grid_masks')
    masks_bind_idx = engine.get_binding_index('masks') 

    # context.set_input_shape('imgs', imgs.shape)
    context.set_binding_shape(imgs_bind_idx, imgs.shape)
    imgs_buf = np.ascontiguousarray(imgs)
    imgs_mem = cuda.mem_alloc(imgs.nbytes)

    # context.set_input_shape('grids', grids.shape)
    context.set_binding_shape(grids_bind_idx, grids.shape)
    grids_buf = np.ascontiguousarray(grids)
    grids_mem = cuda.mem_alloc(grids.nbytes)

    context.set_binding_shape(grid_masks_bind_idx, grid_masks.shape)
    grid_masks_buf = np.ascontiguousarray(grid_masks)
    grid_masks_mem = cuda.mem_alloc(grid_masks.nbytes)

    # context.set_input_shape('masks', masks.shape)
    context.set_binding_shape(masks_bind_idx, masks.shape)
    masks_buf = np.ascontiguousarray(masks)
    masks_mem = cuda.mem_alloc(masks.nbytes)

    # output_size = trt.volume(context.get_tensor_shape('inv_dist'))
    # output_dtype = trt.nptype(engine.get_tensor_dtype('inv_dist'))
    output_bind_idx = engine.get_binding_index('inv_dist')
    output_size = tuple(context.get_binding_shape(output_bind_idx))
    output_dtype = engine.get_binding_dtype(output_bind_idx)
    # output_buf = cuda.pagelocked_empty(output_size, output_dtype)
    output_buf = cuda.pagelocked_empty(output_size, np.float32)
    output_mem = cuda.mem_alloc(output_buf.nbytes)
    
    # Transfer input data to the GPU.
    stream = cuda.Stream()
    cuda.memcpy_htod_async(imgs_mem, imgs_buf, stream)
    cuda.memcpy_htod_async(grids_mem, grids_buf, stream)
    cuda.memcpy_htod_async(grid_masks_mem, grid_masks_buf, stream)
    cuda.memcpy_htod_async(masks_mem, masks_buf, stream)

    # Single prediction save image
    context.execute_async_v2(
        bindings=[
            int(imgs_mem), 
            int(grids_mem), 
            int(grid_masks_mem), 
            int(masks_mem), 
            int(output_mem)],
        stream_handle=stream.handle
    )

    # Transfer prediction output from the GPU.
    cuda.memcpy_dtoh_async(output_buf, output_mem, stream)

    stream.synchronize()

    # output = np.frombuffer(output_buf, dtype=output_dtype)
    # output = output.reshape(context.get_tensor_shape('inv_dist'))
    output = np.frombuffer(output_buf, dtype=np.float32)
    output = output.reshape(output_size)

    pred = output[0, 0]

    # Normalize. TODO: fix range
    pred_min, pred_max = pred.min(), pred.max()
    pred_min = max( 0.0, pred_min )
    pred_max = min( 192, pred_max )

    pred = (pred.astype(np.float32) - pred_min) / ( pred_max - pred_min )
    pred = np.clip(pred, 0, 1) * 255
    pred = cv2.applyColorMap(pred.astype(np.uint8),cv2.COLORMAP_JET)

    cv2.imwrite( os.path.join( working_dir, 'trt_out.png'), pred )

    # Timing

    elapsed_times = []
    for _ in range(101):
            start = time.time()
            # output = session.run(['inv_dist'], {'imgs': imgs, 'grids': grids, 'masks': masks})
            context.execute_async_v2(
                bindings=[
                    int(imgs_mem), 
                    int(grids_mem), 
                    int(grid_masks_mem),
                    int(masks_mem), 
                    int(output_mem)],
                stream_handle=stream.handle
            )

            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buf, output_mem, stream)

            stream.synchronize()
            elapsed = time.time() - start
            elapsed_times.append(elapsed)

    avg_elapsed = sum(elapsed_times[1:])/(len(elapsed_times) - 1)

    print('Elapsed time:', avg_elapsed)

if __name__ == '__main__':
      main()