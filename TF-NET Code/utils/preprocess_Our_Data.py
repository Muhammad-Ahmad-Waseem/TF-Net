import os
import numpy as np
import rasterio
import cv2
from tqdm import tqdm
import json
from skimage.draw import polygon


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
    polyMasks = np.zeros((512, 512))
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
        # assert np.max(polyMasks) == 1, "No built=up pixel found"

        if np.sum(polyMasks) <= 5:
            return None
    else:
        return None

    return polyMasks


base_path = "/mnt/Ahmad/TeamUrbanTechPh12/Data"
outp_path = '/mnt/Ahmad/TeamUrbanTechPh12/Preprocessed'
partitions = [name for name in os.listdir(base_path) if '.' not in name]
tiff_folder = 'georeferenced_images'
masks_folder = 'masks'
edges_folder = 'edges'

tiff_img_path = "/mnt/Ahmad/Tiff_files/DHA_Ph12.tif"

target_size = (640, 640)
padding_pixels = (64, 64)
padding_value = 0

img = rasterio.open(tiff_img_path)
coords = (img.bounds[0], img.bounds[2], img.bounds[1], img.bounds[3]) # left edge, right edge, bottom edge, top edge
img = np.transpose(img.read(), (1, 2, 0))
print("Actual Image Size: {}".format(img.shape))
org_size = (img.shape[0], img.shape[1])
img = cv2.copyMakeBorder(img, padding_pixels[0], padding_pixels[0], padding_pixels[1],
                         padding_pixels[1], cv2.BORDER_CONSTANT, value=padding_value)
print("Image Size after adding ({}, {}) boundary pixels: {}".format(padding_pixels[0],
                                                                    padding_pixels[0], img.shape))

PIX_VALUE_MAX = 255  # The max data value we have
PIX_VALUE_MAX_REQ = 255  # The max data value we need
for partition in partitions:
    num_blank_mask = 0
    num_blank_cont = 0
    # print(tiff_folder)
    base_inp_path = os.path.join(base_path, partition, tiff_folder)
    images = os.listdir(base_inp_path)
    out_dir = os.path.join(outp_path, partition)

    if not os.path.exists(os.path.join(out_dir, "images_preprocessed")):
        os.makedirs(os.path.join(out_dir, "images_preprocessed"))

    if not os.path.exists(os.path.join(out_dir, "masks")):
        os.makedirs(os.path.join(out_dir, "masks"))

    if not os.path.exists(os.path.join(out_dir, "boundaries")):
        os.makedirs(os.path.join(out_dir, "boundaries"))

    # Define k_x, k_y to define 'useful' portion, since we are taking patches with overlapping area.
    k_y = target_size[0] - 2 * padding_pixels[0]
    k_x = target_size[1] - 2 * padding_pixels[1]

    pbar = tqdm(images)

    for file in pbar:
        dataset = rasterio.open(os.path.join(base_inp_path, file))
        file_name = file.split('.')[0]

        tiff_file = os.path.join(base_inp_path, file)
        dataset = rasterio.open(tiff_file)
        x_coords, y_coords = np.array([dataset.bounds[0]]), np.array([dataset.bounds[3]])
        out1, out2 = coord_to_px(x_coords, y_coords, y_flip=True, use_img=False, size=org_size, coords=coords)

        y1 = out2[0] + padding_pixels[0]
        x1 = out1[0] + padding_pixels[1]
        y2 = y1 + k_y
        x2 = x1 + k_x
        img_crop = img[y1 - padding_pixels[0]: y2 + padding_pixels[0], x1 - padding_pixels[1]: x2 + padding_pixels[1]]
        img_crop = ((img_crop / PIX_VALUE_MAX) * PIX_VALUE_MAX_REQ).astype(np.uint8)

        np.save(os.path.join(out_dir, "images_preprocessed", file_name), img_crop)

        mask_file = os.path.join(base_inp_path.replace(tiff_folder, masks_folder), 'mask_'+file_name+".geojson")
        edge_file = os.path.join(base_inp_path.replace(tiff_folder, edges_folder), 'edge_' + file_name + ".geojson")
        empty_mask = True
        empty_cont = True
        # print(mask_file)
        # print(file_name)
        if os.path.exists(mask_file):
            with open(mask_file) as gf:
                geojson = json.load(gf)
            mask = geoJsonToMask(geojson, dataset)
            empty_mask = False if mask is not None else True

        if os.path.exists(edge_file):
            with open(mask_file) as gf:
                geojson = json.load(gf)
            cont = geoJsonToMask(geojson, dataset)
            empty_cont = False if mask is not None else True

        if not empty_mask:
            # save masks
            mask = mask.astype(np.uint8)
            np.save(os.path.join(out_dir, "masks", file_name + "_mask"), mask)
        else:
            num_blank_mask += 1
            pbar.set_description("({}, {}) samples are blank in {} partition".format(num_blank_mask,
                                                                                     num_blank_cont, partition))
            empty_array = np.zeros((512, 512))
            np.save(os.path.join(out_dir, "masks", file_name + "_mask"), empty_array)

        if not empty_cont:
            cont = cont.astype(np.uint8)
            # save edges/boundaries
            np.save(os.path.join(out_dir, "boundaries", file_name + "_boundary"), cont)
        else:
            num_blank_cont += 1
            pbar.set_description("({}, {}) samples are blank in {} partition".format(num_blank_mask,
                                                                                     num_blank_cont, partition))
            empty_array = np.zeros((512, 512))
            np.save(os.path.join(out_dir, "boundaries", file_name + "_boundary"), empty_array)
        # pbar.set_postfix_str(partition)
    # break

print("Complete")