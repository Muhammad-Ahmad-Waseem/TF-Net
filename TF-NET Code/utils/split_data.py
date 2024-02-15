import glob
import numpy as np
import os
import shutil

base_dir = "/mnt/Ahmad/TeamUrbanTechDHA/Preprocessed"
path_to_imgs = base_dir+"/train/images_preprocessed/*.npy"

out_path_im = "/mnt/Ahmad/TeamUrbanTechDHA_2/Preprocessed/test/images_preprocessed"
out_path_ms = "/mnt/Ahmad/TeamUrbanTechDHA_2/Preprocessed/test/masks"
out_path_ed = "/mnt/Ahmad/TeamUrbanTechDHA_2/Preprocessed/test/boundaries"
out_path_wm = "/mnt/Ahmad/TeamUrbanTechDHA_2/Preprocessed/test/masks_wt"

if not os.path.exists(out_path_im):
    os.makedirs(out_path_im)

if not os.path.exists(out_path_ms):
    os.makedirs(out_path_ms)

if not os.path.exists(out_path_ed):
    os.makedirs(out_path_ed)

if not os.path.exists(out_path_wm):
    os.makedirs(out_path_wm)

image_ids = glob.glob(path_to_imgs)

np.random.seed(1997)
image_ids = np.random.permutation(image_ids)
split = int(len(image_ids) * 0.8)
split_ids = image_ids[split:]
for p_id in split_ids:
    id = os.path.split(p_id)[-1]
    image_path = p_id
    out_path_img = os.path.join(out_path_im, id)

    shutil.copy(image_path, out_path_img)

    mask_path = p_id.replace(".npy", "_mask.npy").replace("images_preprocessed", "masks")
    out_path_mask = os.path.join(out_path_ms, id.replace(".npy", "_mask.npy"))

    shutil.copy(mask_path, out_path_mask)

    edge_path = p_id.replace(".npy", "_boundary.npy").replace("images_preprocessed", "boundaries")
    out_path_edge = os.path.join(out_path_ed, id.replace(".npy", "_boundary.npy"))

    shutil.copy(edge_path, out_path_edge)

    wm_path = p_id.replace(".npy", "_mask_wt.npy").replace("images_preprocessed", "masks_wt")
    out_path_wmsk = os.path.join(out_path_wm, id.replace(".npy", "_mask_wt.npy"))

    shutil.copy(wm_path, out_path_wmsk)
    # print(out_path_wmsk)

