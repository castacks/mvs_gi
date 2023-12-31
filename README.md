This is the official code repository for the paper: "Geometry-Informed Distance Candidate
Selection for Adaptive Lightweight Omnidirectional Stereo Vision with Fisheye Images".

# Overview

We are working on providing better details about our work, including the code, datasets,
pre-trained models , and more. Please stay tuned while things are progressing. 

At the current moment, we provide instructions on how to run our pre-trained models locally.
Please refer to the [Offline validation instructions](docs/offline_validation/README.md) for more
details.

For the developers, [here](docs/original_home_page_readme/README.md) is the original README page.

# Code structure

(Coming soon)

# Training and validation procedures and pre-trained models

(More details coming soon)

## PyTorch models ##
Pre-trained models (need associated configs):

| Model name | Link                                                                                                                            |
|------------|---------------------------------------------------------------------------------------------------------------------------------|
| E8         | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/pre_trained_models/E8/dsta_sweep_config24_WB_jbektzh2_v122.ckpt)     |
| G8         | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/pre_trained_models/G8/dsta_sweep_config25_WB_koju4sfh_v140.ckpt)     |
| E16        | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/pre_trained_models/E16/dsta_sweep_config19_WB_zdtldl4s_v96.ckpt)     |
| G16        | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/pre_trained_models/G16/dsta_sweep_config20_WB_f6tysxvk_v93.ckpt)     |
| G16V       | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/pre_trained_models/G16V/dsta_sweep_config21_WB_a7kccavm_v59.ckpt)    |
| G16VV      | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/pre_trained_models/G16VV/dsta_sweep_config103_WB_jy2dqg6r_v102.ckpt) |

__Update history__
- 2023-12-31: G16V ( two versions used in various places, need to add config29 )

## TensorRT models ##

We also provide the TensorRT version of some of our models. Please note that depending on what
hard ware platform the user is trying to deploy, the TensorRT model provided by us may not work.
For this purpose, we also provide the ONNX models associated the the TensorRT ones such that a
user can do the conversion on their own hardware platform. Pleae follow the
[instructions](docs/HardwareAcceleration.md) to do the conversion. All the ONNX models are the
sanitized version described in the instructions.

Table: TensorRT model performance.
| Model name | Target platform      | TensorRT ver. | Tested machine         | Infer. time (ms) | Infer. Mem. (MB) |
|------------|----------------------|---------------|------------------------|------------------|------------------|
| G16V       | x86_64               | >=8.6.1       | GTX1070MQ<br>RTX3080Ti | 73<br>6          | 400<br>500       |
| G16V       | Jetson JetPack 4.6.x | 8.2.x         | Jetson Xavier NX       | 160              | 2100             |
| G16VV      | x86_64               | >=8.6.1       | GTX1070MQ<br>RTX3080Ti | 210<br>11        | 800<br>710       |
| G16VV      | Jetson JetPack 4.6.x | 8.2.x         | Jetson Xavier NX       | 270              | 1800             |

Table: Optimized model links.
| Model name | Opt. Ver.               | Link                                                                                                                |
|------------|-------------------------|---------------------------------------------------------------------------------------------------------------------|
| G16V       | ONNX, Operation Set 13  | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/onnx_tensorrt/config29_WB_zslyi5q8_v130_sanitized.onnx)  |
| G16V       | TensorRT 8.2, Xavier NX | [downlaod](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/onnx_tensorrt/config29_WB_zslyi5q8_v130_jp4.6.1.engine)  |
| G16VV      | ONNX, Operation Set 13  | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/onnx_tensorrt/config103_WB_jy2dqg6r_v102_sanitized.onnx) |
| G16VV      | TensorRT 8.2, Xavier NX | [download](https://airlab-share.andrew.cmu.edu:8081/mvs_gi/onnx_tensorrt/config103_WB_jy2dqg6r_v102_jp4.6.1.engine) |

__Update history__
- 2023-12-31: G16V ( was config21 in the paper, now config29, need update paper )

[//]: # (This is a comment line, see https://stackoverflow.com/questions/4823468/comments-in-markdown)
[//]: # (The above TensorRT table is saved at https://drive.google.com/drive/folders/18TTxTwLSsJrnlKawxWXw1eybwctOKNRa?usp=drive_link)

# Datasets

(Coming soon)

# Paper (preprint)

The preprint version of the paper is available on [the AirLab's website](http://theairlab.org/img/posts/2023-10-10-dsta-depth-gicandidates/ICRA_2024__Pulling__Tan__Hu__Scherer.pdf).

# Citation

(Coming soon)

