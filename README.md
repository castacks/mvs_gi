# dsta_mvs

Learning-based multi-view stereo using fisheye cameras.

## Repository organization

``` sh
dsta_mvs_lightning/
├── api_key.env                     # WandB API key, used by docker script (provide your own)
├── config.yaml                     # Lightning configuration YAML
├── Dockerfile                      # Dockerfile to build development image
├── docker_run.sh                   # Convenience script to start Docker container
├── main.py                         # Main training/validation/test script
├── README.md                       
├── requirements.txt                
└── dsta_mvs/                       
    ├── image_sampler/              # Git submodule
    ├── mvs_utils/                  # Git submodule
    ├── __init__.py
    ├── model/                      # Model-related code
    │   ├── __init__.py
    │   ├── common/
    │   ├── cost_volume_builder/
    │   ├── cost_volume_regulator/
    │   ├── distance_regressor/
    │   ├── feature_extractor/
    │   └── mvs_model/              # Models (LightningModules)
    └── support                     # Training-related code
        ├── __init__.py
        ├── register.py
        ├── augmentation/
        ├── datamodule/             # LightningDataModules
        ├── dataset/
        └── loss_function/
```

## Interactive training using Docker (local/DGX)

### 1. Build/pull Docker image

A [pre-built docker image](https://hub.docker.com/r/jehontan/mvs-lightning) is available on Docker Hub.

```sh
docker pull jehontan/mvs-lightning
```

or build your own

```sh
docker build -t <your image tag> . 
```

### 2. Enable/disable WandB

To enable WandB logging place a `api_key.env` file in the root repository directory with contents:

```sh
WANDB_API_KEY=<your_api_key>
```
Alternatively, you can login interactively in the container, or set the `WANDB_API_KEY` environment variable using other means.

### 3. Run interactive container session

A convenience script is provided.

```sh
./docker_run.sh
```

### 4. Run main script

From the `dsta_mvs_lightning/` directory:

```sh
python main.py fit -c config.yaml
```

This will run training using the configuration defined in `config.yaml`.
## Interactive training on SLURM + Singularity (Perceptron/PSC)

### 1. Start an interactive SLURM job
```sh
srun --pty -p a100-gpu-shared --gpus 4 --cpu-per-task 64 --mem 512G bash
```

### 2. Start shell session with Singularity container

```sh
singularity shell --nv <your_sif_file>
```

The SIF file may be generated using the Dockerfile by first creating a docker image then building SIF from the docker image. Building the docker image is not supported on Perceptron/PSC.

A [pre-built docker image](https://hub.docker.com/r/jehontan/mvs-lightning) is available on Docker Hub. The SIF file can be built on the cluster:

```sh
singularity build <your_sif_file> docker://jehontan/mvs-lightning
```

A pre-built SIF file is available on Perceptron at `/storage2/datasets/jehont/mvs_lightning.sif`

### 3. Enable/disable WandB

To enable logging to WandB you will need to login using your API key.

```sh
wandb login <your_api_key>
```

This should only need to be done once; the login credentials are cached.

To disable WandB (e.g. for debugging):

```sh
wandb disabled
```

The enabled/disabled state is persistent across runs. Remember to enable WandB again when done.

```sh
wandb enabled
```

### 4. Run main script

From the `dsta_mvs_lightning/` directory:

```sh
python main.py fit -c config.yaml
```

This will run training using the configuration defined in `config.yaml`.
