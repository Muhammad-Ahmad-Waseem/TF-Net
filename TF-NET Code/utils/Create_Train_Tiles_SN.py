import rasterio
import os
from tqdm import tqdm

base_tiff_path = "/mnt/Ahmad/SpaceNet_Data/SpaceNet2/AOI_5_Khartoum_Train/RGB-PanSharpen"
base_inp_path = "/mnt/Ahmad/SpaceNet_Data/SpaceNet2/AOI_5_Khartoum_Train/geojson/buildings"
out_path = "/mnt/Ahmad/SpaceNet_Data/SpaceNet2/AOI_5_Khartoum_Train"
base_tiff_name = "RGB-PanSharpen_AOI_5_Khartoum"
images = os.listdir(base_inp_path)
pbar = tqdm(images)

out_file = os.path.join(out_path, 'polygons.csv')
fw = open(out_file, 'w')
fw.write("filename,id,edge,Geometry\n")
fw.close()

ct = 0

for file in pbar:
    file_name = file.split('_')[-1].split('.')[0]
    # print(file_name)
    file_name = base_tiff_name + '_' + file_name+'.tif'
    out_filename = '_'.join(file.split('_')[1:])
    out_filename = out_filename.split('.')[0]
    # print(out_filename)
    # break

    dataset = rasterio.open(os.path.join(base_tiff_path, file_name))
    x_min = dataset.bounds[0]  # left edge
    x_max = dataset.bounds[2]  # right edge
    y_min = dataset.bounds[1]  # bottom edge
    y_max = dataset.bounds[3]  # top edge
    dataset.close()

    polygon_str = "{} {},{} {},{} {},{} {},{} {}".format(x_min, y_min, x_max, y_min,
                                                    x_max, y_max, x_min, y_max, x_min, y_min)
    fa = open(out_file, 'a')
    fa.write("%s,%d,%s,\"POLYGON ((%s))\"\n" % ("Khartoum", ct, out_filename, polygon_str))
    fa.close()

    ct = ct+1

print("Complete")
