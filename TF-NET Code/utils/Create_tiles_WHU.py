import rasterio
import os
from tqdm import tqdm
import pandas as pd

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
    base_inp_path = os.path.join(base_path, partition, tiff_folder)
    images = os.listdir(base_inp_path)
    tile_info = num_tiles[partition]
    x_tiles = tile_info[0]
    y_tiles = tile_info[1]
    extents = pd.read_csv('/mnt/Ahmad/WHU_Aerial_Data/WHU_Data/{}/{}_area.csv'.format(partition, partition))

    x_start = extents['xmin']
    y_start = extents['ymax']
    x_delta = 153.6 #(extents['xmax'] - x_start) / x_tiles
    y_delta = -153.6 #(extents['ymin'] - y_start) / y_tiles

    pbar = tqdm(images)
    out_file = os.path.join(base_path, partition, '{}_tiles.csv'.format(partition))
    print(out_file)
    fw = open(out_file, 'w')
    fw.write("AOI,id,filename,Geometry\n")
    fw.close()

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

        polygon_str = "{} {},{} {},{} {},{} {},{} {}".format(x_min, y_min, x_max, y_min,
                                                        x_max, y_max, x_min, y_max, x_min, y_min)

        fa = open(out_file, 'a')
        fa.write("%s,%d,%s,\"POLYGON ((%s))\"\n" % ("WHU_{}".format(partition), f_int, file_name, polygon_str))
        fa.close()

        pbar.set_postfix_str(partition)
    # break

print("Complete")