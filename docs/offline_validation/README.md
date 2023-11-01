This README shows how to run offline validation.

# Overview

Running mvs_gi might be a little bit more involved. We need the code, pre-trained model, 
configuration files, dataset, and the envrionment. Here we show the whole process by following 
an example.

The following procedure has been tested only on Linux system. 

# Get the code

mvs_gi has several Git submodules, so the proper way for getting the code is

```bash
cd <awesome location for codes>
git clone https://github.com/castacks/mvs_gi.git
cd mvs_gi
git submodule update --init --recursive
```

# Get the sample data

Run the following command to get the sample data from our data server.

```bash
wget -O sample_validation_data.zip https://airlab-share.andrew.cmu.edu:8081/mvs_gi/code_release_202310_data.zip 
```

Unzip the downloaded file.


```bash
unzip sample_validation_data.zip -d <awesome location for data>
```

After extraction from the zip file, the folder structure should look like the following.

```
<awesome location for data>
├── code_release_202310_data
│   ├── real_world
│   └── synthetic
```

While set up a Docker container in [a later section](#run-the-validation), the user needs to refer to `<awesome location for data>/code_release_202310_data/` as the `/dataset/` folder.

# Docker

## Get the Docker image

The following Docker image could be used for both training and validation.

```bash
docker pull theairlab/dsta_ngc_x86:22.08_11_lightning_for_mvs
```

## (Optional) Augment the Docker image

To reduce possible issues when using Docker. We recommend to augment a raw Docker image by 
adding the local user to it. We provide some automatic scripts to do this.

```bash
cd <awesome location for scripts>
git clone https://github.com/castacks/dsta_docker.git
cd dsta_docker
git submodule update --init --recursive
cd scripts
./add_user_2_image.sh theairlab/dsta_ngc_x86:22.08_11_lightning_for_mvs theairlab/dsta_ngc_x86:22.08_99_local
```

After running the above commands, there will be a new Docker image with the tag of 
`theairlab/dsta_ngc_x86:22.08_99_local`.

## Ensure proper installation of required packages
Make sure your docker container is configured to run with GPU. Refer [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.3/install-guide.html) for installation. Also install `nvidia-container-toolkit` if you haven't already

# WandB

mvs_gi utilizes WandB for logging. Please make sure the user has a valid WandB account. Please also have the API key of the user's WandB account.

# Run offline validation

## Get a sample working directory

A working directory is a folder where the user keeps case-dependent files, such as configurations
and checkpoints. Outputs that are not handled by the default logger also go to the working
directory. We provide a sample working directory for running all of our offline validations. It also
provides a pre-trained model. Use the following command to download the sample working directory.

```bash
cd <awesome location for working directorys> #can be same code folder containing this repo (mvs_gi)
wget https://airlab-share.andrew.cmu.edu:8081/mvs_gi/mvs_gi_code_release_validation_samples.zip
unzip mvs_gi_code_release_validation_samples.zip -d ./
```

After extracting files from the zip file, there are two folders. One of them, `CR_EV004`, is the
working directory. The zip file represents a new concept, that is "working root", the parent folder
of a working directory. Our code makes use of "working root" such that the user can keep machine
specific configuration files under "working root" and all the working directories for different
experiment cases can refer to them. The second folder get extracted from the above zip file contains
some shared configuration files that will be utilized by the sample working directory.

## Local configuration

After unzip the sample working directory. The user need to copy the following two files to a case 
directory. Go to the CR_EV004 folder in the sample working directory that we just downloaded. Then do

```bash
cp <path to code>/samples/offline_validation/CR_EV004/*.sh ./
```

where [path_to_code](https://github.com/castacks/mvs_gi/tree/main/docs/offline_validation#get-the-code) is the previously specified directory when the user clones this repo.

## Run the validation

In the `CR_EV004` folder, we need to modify the content of `run.sh` and `eval.sh` to correctly 
reflect the system setting. 
- `run.sh`: Change the settings of how the volumes are mounted for the Docker container. Change the tag of the Docker image.
    - [path to working root](https://github.com/castacks/mvs_gi/tree/main/docs/offline_validation#get-a-sample-working-directory): path to sample working root `mvs_gi_code_release_inference_samples`

- `eval.sh`: The WandB API key.

 Then run the `run.sh` file.

```bash
cd <path to working root>/CR_EV004
./run.sh
```

While `run.sh` is running, logs are pushed to the WandB cloud. 