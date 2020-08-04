import os, sys
import glob
import numpy as np
SOURCE_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SOURCE_CODE_DIR))
from misc.upload_gdrive import upload_file_to_gdrive, create_new_folder_on_gdrive

root_dir = '/media/diskstation/datasets/UCF101/pose/train/'
np.random.seed(0)
for root, dirs, files in os.walk(root_dir):
    for dir in dirs:
        sub_dir = os.path.join(root_dir, dir)
        files = np.array(glob.glob(os.path.join(sub_dir, '*/*_pose.png')))
        # print(os.path.join(sub_dir, '*/*_pose.jpg'))
        selected_files = np.random.choice(files, size=3)

        for file in selected_files:
            file_name = '_'.join(file.split('/')[-2:])
            upload_file_to_gdrive(file, 'pose_estimation', file_name=file_name)
