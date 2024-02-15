import numpy as np
from skimage import measure
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import re
import os


# //////////////////////// Custom Functions for postprocess \\\\\\\\\\\\\\\\\\\\\\\\
def save_as_csv(out_path, contours):
    out_path = os.path.join(out_path, 'polygons.csv')
    print('save csv: %s, npoly = %d' % (out_path, len(contours)))
    fw = open(out_path, 'w')
    fw.write("filename,id,Geometry\n")
    for j, contour in enumerate(contours):
        polygon_str = re.sub(r"[\[\]]", '', ",".join(map(str, contour)))
        fw.write("%s,%d,\"POLYGON ((%s))\"\n" % ("2022_09_12", j, polygon_str))
    fw.close()


'''
This function takes the contours made from pixels and map them to geo-coords.
The mapping is computed using cordinates of its corresponding raster.
'''


def assign_geocoords(contours, imgs_dir, img_path, y_flip=True):
    # img_path = get_respond_img(imgs_dir,npy_file)
    dataset = rasterio.open(img_path)

    x_min_coord = dataset.bounds[0]  # left edge
    x_max_coord = dataset.bounds[2]  # right edge
    y_min_coord = dataset.bounds[1]  # bottom edge
    y_max_coord = dataset.bounds[3]  # top edge

    y_img_size, x_img_size = dataset.read(1).shape
    dataset.close()

    for it in range(len(contours)):
        contour = contours[it]

        x = (contour[:, 0])
        y = (contour[:, 1])
        if (y_flip):
            y = y_img_size - (y)

        geo_x = x_min_coord + (x / x_img_size) * (x_max_coord - x_min_coord)
        geo_y = y_min_coord + (y / y_img_size) * (y_max_coord - y_min_coord)

        contours[it] = np.vstack((geo_x, geo_y)).T

    return contours


x_div = 1
y_div = 1

ct = 0
geoms_np = []
geoms_polygons = []
for x_tile in range(x_div):
    for y_tile in range(y_div):
        img = np.load('/mnt/Ahmad/Results/Zoom21/SpaceNet_Paris/our_final_SN_all/SN_Paris_preds.npy')
        sizex, sizey = img.shape
        sizex_new, sizey_new = int(sizex / x_div), int(sizey / y_div)
        print(img.shape)
        img = img[x_tile * sizex_new:(x_tile + 1) * sizex_new, y_tile * sizey_new:(y_tile + 1) * sizey_new]
        print(img.shape)
        labels = measure.label(img, connectivity=2, background=0).astype('uint16')
        polygon_gen = shapes(labels, labels > 0)
        for polygon, value in polygon_gen:
            ct = ct + 1
            p = shape(polygon)
            if p.area >= 0:
                p = p.simplify(tolerance=0.5)
                geoms_polygons.append(p)
                try:
                    p = np.array(p.boundary.xy, dtype='int32').T
                except:
                    p = np.array(p.boundary[0].xy, dtype='int32').T

                pcoords_a = 1 * (y_tile * sizey_new + p[:, 0])
                pcoords_b = 1 * (x_tile * sizex_new + p[:, 1])
                p = np.vstack((pcoords_a, pcoords_b)).T
                geoms_np.append(p)

        # Clear out RAM
        img = None
        M1 = None
        startpts = None
        ws_out = None
        labels = None
        polygon_gen = None

geo_poly = (assign_geocoords(geoms_np, "", r"/mnt/Ahmad/Tiff_files/SN_Paris.tif"))
save_as_csv(r"/mnt/Ahmad/Results/Zoom21/SpaceNet_Paris/our_final_SN_all", geo_poly)
print("Complete")