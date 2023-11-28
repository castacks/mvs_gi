import cv2
import os

data_path = '/home/migo/Downloads/depth_estimation/dsta_inference_gascola/media/images'
save_path = '/home/migo/Downloads/depth_estimation/dsta_inference_gascola'
video_name = 'gascola_dsta.mp4'

files = [f for f in os.listdir(data_path) if f.split('_')[2]=='real' and f.split('_')[6].isdigit()]
images = sorted(files, key=lambda x: int(x.split('_')[6]))
print(len(images))
frame = cv2.imread(os.path.join(data_path, images[0]))
h, w, layers = frame.shape
size = (w, h)
out = cv2.VideoWriter(f'{save_path}/{video_name}', cv2.VideoWriter_fourcc(*'mp4v'), 24, size)

for i in images:
    img_path = os.path.join(data_path, i)
    img = cv2.imread(img_path)
    out.write(img)

out.release()
print('Video creation completed!')