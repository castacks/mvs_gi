Run inference with new custom dataset. Follow the steps in [offline_validation doc](https://github.com/castacks/mvs_gi/tree/main/docs/offline_validation) first.

The data directory has the following structure:
- mvs_val_data
  - real_world
  - synthetic

## Dataset requirement
Your dataset should be similar to the following example:

Synthetic:
```
wildfire_ordv1_20231114
├── AmusementParkExposure #name
│   ├── Pose_easy_000 #name
│      ├── cam0 #image folder
│      ├── cam1 #image folder
│      ├── cam2 #image folder
│      ├── default_file_list.csv 
│      ├── meta.json 
│      └── rig 
├── data_partitions.json
├── frame_graph.json
├── manifest.json
├── masks.json
├── metadata.json
├── valid_mask_0.png
├── valid_mask_1.png
└── valid_mask_2.png
```
Real:
```
20230328 #custom_data_parent
├── data 
│   ├── 1107_gascola
│      ├── images_0 #image folder
│      ├── images_1 #image folder
│      ├── images_2 #image folder
├── calibration.json
├── data_partitions.json
├── frame_graph.json
├── manifest.json
├── masks.json
├── metadata.json
├── per_nv_08.csv #listed in data_partitions.json
├── valid_mask_0.png
├── valid_mask_1.png
└── valid_mask_2.png
```

## Change config for custom dataset
1. Put your data folder (i.g. `wildfire_ordv1_20231114`) into corresponding type folder (real / synthetic)

2. Change data root (path to `mvs_val_data`) in `mvs_gi_code_release_inference_samples/CR_EV004/run.sh` if haven't already
    * docker ... --volume="path_to_data/:/dataset/"

3. Change the paths in `change_config.py` and run script to automatically update config files.
    * param `project_3d_point`: bool whether saving 3d point cloud to dataset
    * param `debug_purpose`: bool whether generate config for inference for only few frames (if True) or entire dataset
    * check parameters in `update_ev_yaml()` (~line 113-125) which will update the sample_workspace/config/ev_data_dirs.yaml

4. For debug purposes, check if these updated files are correct
    * sample_workspace/config/ev_data_dirs.yaml
    * sample_workspace/CR_EV004/run.sh
    * dataset/.../data_partitions.json
    * dataset/.../per_env_list.csv

5. Run as specified in offline_validation_doc

## More details on point cloud
Point cloud can be find in the inner-most data folder (where the images folder are stored):
- ex: 'mvs_val_data_gascola/real_world/20230328/data/1107_gascola/point_cloud' or 
'mvs_val_data_gascola/synthetic/DSTA_MVS_Dataset_V3/OldTownSummerExposure_dsta_data/Pose_easy_002/point_cloud'

The entrance is at `_project_3d()` in `mvs_gi/dsta_mvs/model/mvs_model/spherical_sweep_stereo.py`