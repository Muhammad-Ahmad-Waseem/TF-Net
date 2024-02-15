import glob
import numpy as np
import os
import shutil

base_dir = "/mnt/Ahmad/TeamUrbanTechDHA_2/Preprocessed"
splits = [dir for dir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dir))]
source_folder = "/mnt/Ahmad/TeamUrbanTechDHA/Data/train"

folderA = 'edges'
folderB = 'georeferenced_images'
folderC = 'images'
folderD = 'masks'

for split in splits:
    split_path = os.path.join(base_dir, split)
    image_paths = glob.glob(os.path.join(split_path, 'images_preprocessed/*'))
    destin_folder = "/mnt/Ahmad/TeamUrbanTechDHA_2/Data/{}".format(split)

    for path in image_paths:
        id = os.path.split(path)[-1]
        src_fld = os.path.join(source_folder, folderA)
        dst_fld = os.path.join(destin_folder, folderA)
        if not os.path.exists(dst_fld):
            os.makedirs(dst_fld)
        file_name = "edge_{}".format(id.replace('.npy', '.geojson'))
        try:
            shutil.copy(os.path.join(src_fld, file_name), os.path.join(dst_fld, file_name))
        except:
            print("File {} is not found at {}, skipping it..!".format(file_name, src_fld))
        src_fld = os.path.join(source_folder, folderB)
        dst_fld = os.path.join(destin_folder, folderB)
        if not os.path.exists(dst_fld):
            os.makedirs(dst_fld)
        file_name = "{}".format(id.replace('.npy', '.tif'))
        shutil.copy(os.path.join(src_fld, file_name), os.path.join(dst_fld, file_name))
        src_fld = os.path.join(source_folder, folderC)
        dst_fld = os.path.join(destin_folder, folderC)
        if not os.path.exists(dst_fld):
            os.makedirs(dst_fld)
        file_name = "{}".format(id.replace('.npy', '.png'))
        shutil.copy(os.path.join(src_fld, file_name), os.path.join(dst_fld, file_name))
        src_fld = os.path.join(source_folder, folderD)
        dst_fld = os.path.join(destin_folder, folderD)
        if not os.path.exists(dst_fld):
            os.makedirs(dst_fld)
        file_name = "mask_{}".format(id.replace('.npy', '.geojson'))
        try:
            shutil.copy(os.path.join(src_fld, file_name), os.path.join(dst_fld, file_name))
        except:
            print("File {} is not found at {}, skipping it..!".format(file_name, src_fld))