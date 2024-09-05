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

[//]: # (This is a comment line, see https://stackoverflow.com/questions/4823468/comments-in-markdown)
[//]: # (The mvs_gi_download link.)

[mvs_gi_download_link]: https://github.com/castacks/mvs_gi_download

## PyTorch models ##
Pre-trained models (need associated configs):

Updated on 2024-09-05: Please refer to [mvs_gi_download][mvs_gi_download_link] for the download instructions.

| Model name | Link                                    |
|------------|-----------------------------------------|
| E8         | [mvs_gi_download][mvs_gi_download_link] |
| G8         | [mvs_gi_download][mvs_gi_download_link] |
| E16        | [mvs_gi_download][mvs_gi_download_link] |
| G16        | [mvs_gi_download][mvs_gi_download_link] |
| G16V       | [mvs_gi_download][mvs_gi_download_link] |
| G16VV      | [mvs_gi_download][mvs_gi_download_link] |

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
| G16V       | Jetson JetPack 5.1.2 | 8.5.2         | Jetson AGX Xavier      | 104              | 500              |
| G16VV      | x86_64               | >=8.6.1       | GTX1070MQ<br>RTX3080Ti | 210<br>11        | 800<br>710       |
| G16VV      | Jetson JetPack 4.6.x | 8.2.x         | Jetson Xavier NX       | 270              | 1800             |
| G16VV      | Jetson JetPack 5.1.2 | 8.5.2         | Jetson AGX Xavier      | 200              | 600              |
| G16VV      | Jetson Jetpack 5.0.2 | 8.4.1         | Jetson AGX Orin        | 65               | 1900             |

Table: Optimized model links. Updated on 2024-09-05: Please refer to [mvs_gi_download][mvs_gi_download_link] for the download instructions.

| Model name | Opt. Ver.                  |Filename                                   | Link                                    |
|------------|----------------------------|-------------------------------------------|-----------------------------------------|
| G16V       | ONNX, Operation Set 13     | config29_WB_zslyi5q8_v130_sanitized.onnx  | [mvs_gi_download][mvs_gi_download_link] |
| G16V       | TensorRT 8.2, Xavier NX    | config29_WB_zslyi5q8_v130_jp4.6.1.engine  | [mvs_gi_download][mvs_gi_download_link] |
| G16V       | TensorRT 8.5.2, AGX Xavier | config29_WB_zslyi5q8_v130_jp5.1.2.engine  | [mvs_gi_download][mvs_gi_download_link] |
| G16VV      | ONNX, Operation Set 13     | config103_WB_jy2dqg6r_v102_sanitized.onnx | [mvs_gi_download][mvs_gi_download_link] |
| G16VV      | TensorRT 8.2, Xavier NX    | config103_WB_jy2dqg6r_v102_jp4.6.1.engine | [mvs_gi_download][mvs_gi_download_link] |
| G16VV      | TensorRT 8.5.2, AGX Xavier | config103_WB_jy2dqg6r_v102_jp5.1.2.engine | [mvs_gi_download][mvs_gi_download_link] |
| G16VV      | TensorRT 8.4.1, AGX Orin   | config103_WB_jy2dqg6r_v102_jp5.0.2.engine | [mvs_gi_download][mvs_gi_download_link] |

__Update history__
- 2023-12-31: G16V ( was config21 in the paper, now config29, need update paper )
- 2024-09-05: Paper updated after peer review.

[//]: # (This is a comment line, see https://stackoverflow.com/questions/4823468/comments-in-markdown)
[//]: # (The above TensorRT table is saved at https://drive.google.com/drive/folders/18TTxTwLSsJrnlKawxWXw1eybwctOKNRa?usp=drive_link . However, it's outdated. )

# Datasets

We created a new synthetic dataset with 3 fisheye cameras facing to the same direction. The data
are collected by using the simulation environments provided by TartanAir V2, whichi itself is
under development and will be released soon. We have a training set and a validation set. The
total size is about 1.3T with the training set being 1.2T. There are 50 environments for the
trainig set and 21 envrionments for the validation set. We made sure that there are no overlaps
between them.

The dataset is currentl hosted by our own server and we provide simple scripts for downloading.

Updated on 2024-09-05: Please refer to [mvs_gi_download][mvs_gi_download_link] for the download instructions.

The structure of the dataset is the same with the sample dataset used in the [Offline validation instructions](docs/offline_validation/README.md). A separate documentation (coming soon) gives more details about the design of the dataset.

# Paper (preprint)

The preprint version of the paper is available on [the AirLab's website](http://theairlab.org/img/posts/2023-10-10-dsta-depth-gicandidates/ICRA_2024__Pulling__Tan__Hu__Scherer.pdf).

# Citation

(Coming soon)

