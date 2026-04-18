import rasterio
import os
from tqdm import tqdm
import pandas as pd
import subprocess

num_tiles = {
    'train': [64, 74],
    'val': [14, 74],
    'test': [17, 42],
    'test2': [23, 74]
}

base_path = "/mnt/Ahmad/WHU_Aerial_Data/WHU_Data"

partitions = os.listdir(base_path)
tiff_folder = 'image'

for partition in partitions:
    out_path = "/mnt/Ahmad/WHU_Aerial_Data/WHU_Data/{}/geo_referenced_rasters".format(partition)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    base_inp_path = os.path.join(base_path, partition, tiff_folder)
    images = os.listdir(base_inp_path)
    tile_info = num_tiles[partition]
    x_tiles = tile_info[0]
    y_tiles = tile_info[1]
    extents = pd.read_csv('/mnt/Ahmad/WHU_Aerial_Data/WHU_Data/{}/{}_area.csv'.format(partition, partition))

    x_start = extents['xmin']
    y_start = extents['ymax']
    x_delta = 153.6  # (extents['xmax'] - x_start) / x_tiles
    y_delta = -153.6  # (extents['ymin'] - y_start) / y_tiles

    pbar = tqdm(images)

    for file in pbar:
        dataset = rasterio.open(os.path.join(base_inp_path, file))
        file_name = file.split('.')[0]
        if partition == 'test2' or partition == 'val':
            file_name = file_name.split('_')[-1]

        f_int = int(file_name)
        y_idx = int(f_int / x_tiles)
        x_idx = f_int % x_tiles

        x_min = float(x_start + x_idx*x_delta)     # left edge
        x_max = float(x_min + x_delta)             # right edge
        y_max = float(y_start + y_idx*y_delta)     # bottom edge
        y_min = float(y_max + y_delta)             # top edge

        gcp1 = "{} {} {} {}".format(0, 0, x_min, y_max)
        gcp2 = "{} {} {} {}".format(512, 0, x_max, y_max)
        gcp3 = "{} {} {} {}".format(512, 512, x_max, y_min)
        gcp4 = "{} {} {} {}".format(0, 512, x_min, y_min)

        command1 = "gdal_translate -of GTiff -gcp {} -gcp {} -gcp {} -gcp {} {} {}".format(
            gcp1, gcp2, gcp3, gcp4, os.path.join(base_inp_path, file), os.path.join(base_path, file)
        )

        command2 = "gdalwarp -r near -order 1 -co COMPRESS=NONE -t_srs EPSG:2193 {} {}".format(
            os.path.join(base_path, file), os.path.join(out_path, file)
        )

        subprocess.call(command1, shell=True)
        subprocess.call(command2, shell=True)
        subprocess.check_output("rm {}".format(os.path.join(base_path, file)), shell=True)

        pbar.set_postfix_str(partition)
    # break

print("Complete")
