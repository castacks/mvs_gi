# One run to change all configs for running inference
import json
import os
import csv
import yaml
import numpy as np

# data_root = '/home/migo/code/mvs_val_data'
# new_data_type = 'synthetic' 
# custom_data = 'wildfire_ordv1_20231114/AmusementParkExposure' #path to folder containing specific data dir
# custom_data_backslash = 'wildfire_ordv1_20231114\AmusementParkExposure'

data_root = '/home/migo/code/mvs_val_data_gascola'
new_data_type = 'real_world'
custom_data_parent = '20230328' #path to parent data folder
custom_data_path = 'data/1107_gascola'
sample_workspace = '/home/migo/code/mvs_gi_code_release_inference_samples'

debug_purpose = False #if True, generate config file to do inference only on few frames instead of full data
project_3d_point = True

#=================================================================================================
def data_partition_synthetic():
  #update data_partitions.jon
  with open(f'{data_root}/{new_data_type}/data_partitions.json', 'r') as f:
    origin = json.load(f)

  data_list = os.listdir(f'{data_root}/{new_data_type}/{custom_data}')
  if custom_data not in origin['validate']:
    origin['validate'][custom_data]  = data_list
  
  csv_path = origin['per_env'][''][0]
  print('updating', f'{data_root}/{new_data_type}/{csv_path}')
  update_csv_synthetic(f'{data_root}/{new_data_type}/{csv_path}', data_list)

def update_csv_synthetic(csv_path, data_list):
  rows = []
  for data in data_list:
    row = [f'{custom_data_backslash}\{data}\cam0\\000001_Fisheye.png',
          f'{custom_data_backslash}\{data}\cam1\\000001_Fisheye.png',
          f'{custom_data_backslash}\{data}\cam2\\000001_Fisheye.png',
          f'{custom_data_backslash}\{data}\cam0\\000001_Fisheye.png',
          f'{custom_data_backslash}\{data}\cam0\\000001_FisheyeDistance.png',
          f'{custom_data_backslash}\{data}\cam1\\000001_FisheyeDistance.png',
          f'{custom_data_backslash}\{data}\cam2\\000001_FisheyeDistance.png',
          f'{custom_data_backslash}\{data}\cam0\\000001_FisheyeDistance.png',
          'rig\\000_CubeScene.png']
    rows.append(row)
  print(rows)

  if os.path.exists(csv_path):
    with open(csv_path, mode='a', newline='') as csvfile:
      # Check if the file is empty or not
      csvfile.seek(0, os.SEEK_END)  # Move the file pointer to the end of the file
      if csvfile.tell() > 0:  
        csvfile.seek(0, os.SEEK_END)  
        if csvfile.read(1) != '\n':  
            csvfile.write('\n')  # Write a newline character
      csvfile.seek(0, os.SEEK_END)  
      csvwriter = csv.writer(csvfile)
      csvwriter.writerows(rows)
  else:
    fields = ['cam0_rgb_fisheye','cam1_rgb_fisheye','cam2_rgb_fisheye','cam0_dist_fisheye','cam1_dist_fisheye','cam2_dist_fisheye','rig_dist_fisheye','rig_rgb_fisheye','rig_rgb_equirect']
    with open(csv_path, mode='w') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(fields)
      csvwriter.writerows(rows)

def update_csv_real():
  # get csv file name
  with open(f'{data_root}/{new_data_type}/{custom_data_parent}/data_partitions.json', 'r') as f:
    data_par = json.load(f)
  
  first_inner_dict = next(iter(data_par.values()))
  first_list = next(iter(first_inner_dict.values()))
  csv_name = first_list[0] if first_list else None
  cvs_path = f'{data_root}/{new_data_type}/{custom_data_parent}/{csv_name}'
  print(csv_name)

  #get all frames
  frames = os.listdir(f'{data_root}/{new_data_type}/{custom_data_parent}/{custom_data_path}/images_0')
  frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
  if debug_purpose:
    frames = np.random.choice(frames, 5)
  rows = []

  for frame in frames:
    row = [f'{custom_data_path}/images_0/{frame}',
          f'{custom_data_path}/images_1/{frame}', 
          f'{custom_data_path}/images_2/{frame}']
    rows.append(row)

  if os.path.exists(cvs_path):
    with open(cvs_path, mode='a') as csvfile:
      # Check if the file is empty or not
      csvfile.seek(0, os.SEEK_END)  # Move the file pointer to the end of the file
      if csvfile.tell():  # If the file is not empty (pointer is not at the start)
          csvfile.write('\n')  # Write a newline character
      csvwriter = csv.writer(csvfile)
      csvwriter.writerows(rows)
  else:
    with open(cvs_path, 'w') as csvfile:
      csvwriter = csv.writer(csvfile)
      fields = ['cam0', 'cam1', 'cam2']
      csvwriter.writerow(fields)
      csvwriter.writerows(rows)

def update_ev_yaml(update_exist = False):
  with open(f'{sample_workspace}/config/ev_data_dirs.yaml', 'r') as file:
      yaml_dict = yaml.safe_load(file)
      print(yaml_dict)
  for dataset in yaml_dict: #update config for projecting 3d point for all dataset 
    yaml_dict[dataset]['project_3d_points'] = project_3d_point
  with open(f'{data_root}/{new_data_type}/{custom_data_parent}/data_partitions.json', 'r') as f:
    data_par = json.load(f)
  for key in data_par:
    dataset_name = key#must be same as data_partitions.json
    if dataset_name not in yaml_dict or update_exist:
      innerDict = dict()
      innerDict['path'] = f"/dataset/{new_data_type}/{custom_data_parent}/"
      innerDict['mask_path'] = "masks.json"
      innerDict['print_imgs'] = True
      innerDict['project_3d_points'] = project_3d_point
      innerDict['print_warped_inputs'] = True
      innerDict['print_imgs_batch_freq'] = 1
      innerDict['compute_metrics'] = False
      innerDict['batch_size'] = 1
      innerDict['num_samples'] = 0
      innerDict['step_size'] = 1
      innerDict['order'] = [2,1,0]
      innerDict['type'] = "real"
      innerDict['keep_raw_imgs'] = False
      yaml_dict[dataset_name] = innerDict
  print(yaml_dict)
  with open(f'{sample_workspace}/config/ev_data_dirs.yaml', 'w') as file:
    yaml.dump(yaml_dict, file)
  
if __name__ == '__main__':
  if new_data_type == 'synthetic':
    data_partition_synthetic()
  elif new_data_type == 'real_world':
    update_csv_real()
    update_ev_yaml(update_exist=True)
  #no need to change sample_workspace/config/ev_data_dirs.yaml if synthetic

