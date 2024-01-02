# Hardware-accelerated deployment on Nvidia Jetson

This document descirbes the deployment workflow to enable hardware acceleration via TensorRT (TRT)
on the Nvidia Jetson platform.

The general workflow is as follows:

1. Load model from a training checkpoint and export as an ONNX model.
2. Preprocess the ONNX model to improve compatibility with TensorRT by compilation.
3. Compile TensorRT engine on the target device.
4. Perform inference using TRT Engine.

This workflow has been validated on Jetpack 4.6.1 and 5.1.

## ONNX export from checkpoint

ONNX export involves tracing the PyTorch model and mapping PyTorch operations to ONNX operations.
The specific mapping depends on the version of PyTorch and the target ONNX opset (operation set).

Use `scripts/export.py` to load a checkpoint and perform ONNX export. Be sure to set the
`opset_version` to a compatible version for the target Jetson platform. 
- Jetpack 4.6.1 supports opset 13.
- Jetpack 5.1 supports opset 16.

> **Note**
>
> The input checkpoint and output ONNX model paths are currently specified in the script. This
> should be exposed as script arguments in the future.

## ONNX model preprocessing

The default PyTorch ONNX export is suboptimal and some
[problems](https://github.com/onnx/onnx-tensorrt/blob/main/docs/faq.md#common-assertion-errors) were
encounted compiling the TRT Engine from the directly-exported model. To improve compatibility, we
preprocess the ONNX model using Nvidia's [polygraphy][polygraphy_git] tool.

[polygraphy_git]: https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy

```sh
polygraphy surgeon sanitize <input ONNX> --fold-constants -o <ouput ONNX>
```

## Compile TensorRT engine

> **Note**
>
> This step needs to be done on the target hardware, as TensorRT implementation is hardware
> specific.

To utilize TensorRT, the model must first be compiled to a binary TensorRT engine. The conversion
process utilizes multiple tactics to attempt to determine the best way to distribute operations
across the available hardware. For this reason, the process may take a long time, and is hardware
configuration specific and must be performed on the target hardware.

We use [polygraphy][polygraphy_git] to perform the TRT engine compilation, with `float16` conversion
enabled. Note, `<output TRT engine>` must end with a `.engine` extension.

```sh
# For TensorRT 8.6.
polygraphy convert <input ONNX> --fp16 -o <output TRT engine>

# For TensorRT 8.2 on Jetson with JetPack 4.6.x.
polygraphy convert <input ONNX> --workspace=500M --fp16 -o <output TRT engine>
```

## Perform inference

The `scripts/predict_trt.py` is provided for perfoming inference with the resultant TensoRT engine.