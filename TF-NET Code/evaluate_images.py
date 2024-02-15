# //////////////////////// Evaluate on SpaceNet Images \\\\\\\\\\\\\\\\\\\\\\\\
import numpy as np
from skimage import measure
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import re
import os
from tqdm import tqdm
import torch
import cv2
import segmentation_models_pytorch as smp
from model import DeepLabV3Plus as MyModel
from train import get_preprocessing

# //////////////////////// Custom Functions for postprocess \\\\\\\\\\\\\\\\\\\\\\\\
def save_as_csv(out_path, contours):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_path = os.path.join(out_path, 'polygons.csv')
    print('save csv: %s, npoly = %d' % (out_path, len(contours)))
    fw = open(out_path, 'w')
    fw.write("filename,id,Geometry\n")
    for j, contour in enumerate(contours):
        polygon_str = re.sub(r"[\[\]]", '', ",".join(map(str, contour)))
        fw.write("%s,%d,\"POLYGON ((%s))\"\n" % ("CITY_train", j, polygon_str))
    fw.close()


'''
This function takes the contours made from pixels and map them to geo-coords.
The mapping is computed using cordinates of its corresponding raster.
'''


def assign_geocoords(contours, img_path, y_flip=True):
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

ct = 0
geo_polygons = []
base_inp_path = "/mnt/Ahmad/TeamUrbanTechDHA_2/Preprocessed/train/images_preprocessed"
base_tiff_path = "/mnt/Ahmad/TeamUrbanTechDHA_2/Data/train/georeferenced_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
# Load pre-processing function
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing = get_preprocessing(preprocessing_fn)
# load best weights
output_dir = "/mnt/Ahmad/Models/Segmentation/trained_model/our_final_wo_nepagg_CITY_small"
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
model = MyModel(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION
    )
model = torch.load(os.path.join(output_dir, 'current_model.h5'), map_location=DEVICE)
model.eval()
images = os.listdir(base_inp_path)
pbar = tqdm(images)
for file in pbar:
    img = np.load(os.path.join(base_inp_path, file))
    img = img[64:576, 64:576]
    # img = img[83:733, 83:733]
    # img = cv2.copyMakeBorder(img, 0, 6, 0, 6, cv2.BORDER_CONSTANT, value=0)
#     img = resize_image(img, new_size=(816, 816))
    sample = preprocessing(image=img)
    img = torch.Tensor(sample['image']).permute(2, 0, 1).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
                pred_mask, _ = model(img)
    pr_mask = pred_mask.round().squeeze()
    pr_mask = pr_mask.detach().squeeze().cpu().numpy()
    # pr_mask = pr_mask[64:576, 64:576]
    # pr_mask = pr_mask[83:733, 83:733]
    np_polygons = []
    labels = measure.label(pr_mask, connectivity=2, background=0).astype('uint16')
    polygon_gen = shapes(labels, labels > 0)
    for polygon, value in polygon_gen:
        ct = ct + 1
        p = shape(polygon)
        if p.area >= 25:
            p = p.simplify(tolerance=0.5)
            try:
                p = np.array(p.boundary.xy, dtype='int32').T
            except:
                p = np.array(p.boundary[0].xy, dtype='int32').T
            np_polygons.append(p)
    _id = file.split('.')[0]
    # name = "RGB-PanSharpen_AOI_2_Vegas_img" + _id + '.tif'
    # name = "val_" + _id + '.tif'
    name = _id + '.tif'
    # name = '2_' + _id + '.tif'
    geo_poly = (assign_geocoords(np_polygons, os.path.join(base_tiff_path, name)))
    geo_polygons.extend(geo_poly)

save_as_csv("/mnt/Ahmad/Results/Zoom21/TeamUrbanTechDHA_2/train/our_final_wo_nepagg_CITY_small", geo_polygons)
# np.save("/home/gcf/Desktop/Ahmad/Building_Footprints_Extraction/Previous/labels.npy", labels)
print("Complete")