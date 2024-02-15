import os
import numpy as np
import rasterio
import cv2
from tqdm import tqdm
import json
from skimage.draw import polygon
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt


def coord_to_px(geo_x, geo_y, dataset=None, y_flip=True, use_img=True, size=None, coords=None):
    if use_img:
        assert dataset is not None, "Provide rasterio dataset for conversion"
        y_img_size, x_img_size = dataset.read(1).shape
        x_min_coord = dataset.bounds[0]  # left edge
        x_max_coord = dataset.bounds[2]  # right edge
        y_min_coord = dataset.bounds[1]  # bottom edge
        y_max_coord = dataset.bounds[3]  # top edge
        pix_x = (((geo_x - x_min_coord) / (x_max_coord - x_min_coord)) * (x_img_size - 1)).round().astype(int)
        pix_y = (((geo_y - y_min_coord) / (y_max_coord - y_min_coord)) * (y_img_size - 1)).round().astype(int)
        if y_flip:
            pix_y = y_img_size - pix_y - 1
    else:
        assert size is not None, "Provide additional size to map, if not us img size"
        assert coords is not None, "Provide additional coords to map, if not using img coords"
        y_img_size, x_img_size = size
        x_min_coord, x_max_coord, y_min_coord, y_max_coord = coords
        pix_x = (((geo_x - x_min_coord) / (x_max_coord - x_min_coord)) * x_img_size).round().astype(int)
        pix_y = (((geo_y - y_min_coord) / (y_max_coord - y_min_coord)) * y_img_size).round().astype(int)
        if y_flip:
            pix_y = y_img_size - pix_y

    return pix_x, pix_y


def geoJsonToMask(geojson, dataset):
    polyMasks = np.zeros((650, 650))
    for i, bldg in enumerate(geojson['features']):
        feature_type = bldg['geometry']['type']
        if 'Polygon' not in feature_type:
            continue

        polygons = [bldg['geometry']['coordinates']] if feature_type == "Polygon" else bldg['geometry']['coordinates']

        for mask in polygons:
            rasteredPolygon = np.array(mask[0])
            # print(rasteredPolygon)
            # xs, ys = tiff.coord_to_px(rasteredPolygon[:,0], rasteredPolygon[:,1], latlon=True)
            xs, ys = coord_to_px(rasteredPolygon[:, 0], rasteredPolygon[:, 1], dataset=dataset, y_flip=True)
            cc, rr = polygon(xs, ys)
            polyMasks[rr, cc] = 1

    if len(geojson['features']) > 0:
        # print(np.max(polyMasks))
        # print(np.min(polyMasks))
        assert np.max(polyMasks) == 1, "No built=up pixel found"

        if np.sum(polyMasks) <= 5:
            return None
    else:
        return None

    return polyMasks

AOI_id = 5
AOI_name = 'Khartoum'
root_dir = '/mnt/Ahmad/SpaceNet_Data/SpaceNet2/AOI_{}_{}_Train'.format(AOI_id, AOI_name)                       # base path for SN data
# tiff_img_path = "/mnt/Ahmad/Tiff_files/SN_Shanghai.tif"                                   # path to combined tiff file
output_dir = '/mnt/Ahmad/SpaceNet_Data/Preprocessed/{}'.format(AOI_name)
                                                                                    # path to save pre-processed tiles

image_prefix = "RGB-PanSharpen_AOI_{}_{}_img".format(AOI_id, AOI_name)
mask_prefix = "buildings_AOI_{}_{}_img".format(AOI_id, AOI_name)
edge_prefix = "edge_AOI_{}_{}_img".format(AOI_id, AOI_name)
target_size = (816, 816)
padding_pixels = (83, 83)
padding_value = 0
PIX_VALUE_MAX = 2048  # The max data value we have
PIX_VALUE_MAX_REQ = 255  # The max data value we need
# Define k_x, k_y to define 'useful' portion, since we are taking patches with overlapping area.
k_y = target_size[0] - 2 * padding_pixels[0]
k_x = target_size[1] - 2 * padding_pixels[1]
# img = rasterio.open(tiff_img_path)
# coords = (img.bounds[0], img.bounds[2], img.bounds[1], img.bounds[3])  # left edge, right edge, bottom edge, top edge
# img = np.transpose(img.read(), (1, 2, 0))
# print("Actual Image Size: {}".format(img.shape))
# org_size = (img.shape[0], img.shape[1])
# img = cv2.copyMakeBorder(img, padding_pixels[0], padding_pixels[0], padding_pixels[1],
#                              padding_pixels[1], cv2.BORDER_CONSTANT, value=padding_value)
# print("Image Size after adding ({}, {}) boundary pixels: {}".format(padding_pixels[0], padding_pixels[0], img.shape))


geojson_files = os.listdir(os.path.join(root_dir, 'geojson/buildings'))
ids = [g[g.index('img') + 3: g.index('.')] for g in geojson_files]
pbar = tqdm(ids)

if not os.path.exists(os.path.join(output_dir, "images_preprocessed")):
    os.makedirs(os.path.join(output_dir, "images_preprocessed"))

if not os.path.exists(os.path.join(output_dir, "masks")):
    os.makedirs(os.path.join(output_dir, "masks"))

if not os.path.exists(os.path.join(output_dir, "new_boundaries")):
    os.makedirs(os.path.join(output_dir, "new_boundaries"))

num_blank = 0
for img_id in pbar:
    tiff_file = os.path.join(root_dir, "RGB-PanSharpen", image_prefix + img_id + ".tif")
    geojson_file = os.path.join(root_dir, "geojson/buildings", mask_prefix + img_id + ".geojson")
    dataset = rasterio.open(tiff_file)
    x_coords, y_coords = np.array([dataset.bounds[0]]), np.array([dataset.bounds[3]])
    # out1, out2 = coord_to_px(x_coords, y_coords, y_flip=True, use_img=False, size=org_size, coords=coords)
    # x_idx, y_idx = int(out1[0]/k_x), int(out2[0]/k_y)
    # y1 = out2[0] + padding_pixels[0]
    # x1 = out1[0] + padding_pixels[1]
    # y2 = y1 + k_y
    # x2 = x1 + k_x
    # img_crop = img[y1 - padding_pixels[0]: y2 + padding_pixels[0],
    #            x1 - padding_pixels[1]: x2 + padding_pixels[1]]
    # img_crop = ((img_crop / PIX_VALUE_MAX) * (PIX_VALUE_MAX_REQ)).astype(np.uint8)
    # np.save(os.path.join(output_dir, "images_preprocessed", img_id), img_crop)

    with open(geojson_file) as gf:
        geojson = json.load(gf)

    mask = geoJsonToMask(geojson, dataset)
    if mask is None:
        num_blank += 1
        pbar.set_description("{} samples are blank".format(num_blank))
        # image = tiff.r / 2048.0
        empty_array = np.zeros((650, 650))
        # np.save(os.path.join(output_dir, "masks", img_id + "_mask"), empty_array)
        np.save(os.path.join(output_dir, "new_boundaries", img_id + "_boundary"), empty_array)
    else:
        # save masks
        mask = mask.astype(np.uint8)
        # np.save(os.path.join(output_dir, "masks", img_id + "_mask"), mask)

        boundary = find_boundaries(mask, mode='inner')
        ZZ = np.zeros((650, 650))
        ZZ[boundary] = 1
        kernel = np.ones((7, 7), np.uint8)
        boundary = cv2.dilate(ZZ, kernel, iterations=1)
        boundary = boundary * mask

        boundary = boundary.astype(np.uint8)
        # save edges/boundaries
        np.save(os.path.join(output_dir, "new_boundaries", img_id + "_boundary"), boundary)
    # print(x_idx, y_idx)