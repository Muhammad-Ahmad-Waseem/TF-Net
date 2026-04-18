import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from skimage import measure
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import re
from tqdm import tqdm
import torch
import cv2
import argparse
from utils.pixel_metric import Evaluator
import segmentation_models_pytorch as smp
from model import DeepLabV3Plus as MyModel
from train import get_preprocessing


# //////////////////////// ARGUMENT PARSER \\\\\\\\\\\\\\\\\\\\\\\\
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="weights/Trained_wo_NePAGG/CITY_DHA/current_model.h5",
                    help="Path to trained model")
parser.add_argument('--data_dir', default=r"D:\Ahmad\3090 Data\TeamUrbanTechDHA_2\Preprocessed\test\images_preprocessed",
                    help='Path to images')
parser.add_argument('--tiff_images_path', default=r"D:\Ahmad\3090 Data\TeamUrbanTechDHA_2\Data\test\georeferenced_images",
                    help="Path to trained model")
parser.add_argument('--db', default="CITY_DHA", help='Overwrite original model')
parser.add_argument('--partition', default='test', help='Name of partition to be used')
parser.add_argument('--use_nepagg', action='store_true', default=False, help='Flag to use NePagg or not')


# //////////////////////// Custom Functions for postprocess \\\\\\\\\\\\\\\\\\\\\\\\
def save_as_csv(out_path, contours, db, partition):
    out_path = os.path.join(out_path, 'polygons.csv')
    print('save csv: %s, npoly = %d' % (out_path, len(contours)))
    fw = open(out_path, 'w')
    fw.write("filename,id,Geometry\n")
    for j, contour in enumerate(contours):
        polygon_str = re.sub(r"[\[\]]", '', ",".join(map(str, contour)))
        fw.write("%s,%d,\"POLYGON ((%s))\"\n" % (f"{db}_{partition}", j, polygon_str))
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

supported_dbs = ['WHU', 'CITY_DHA', 'SpaceNet2']
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# criterion = nn.BCELoss().to(device)
assert args.db in supported_dbs, "Unsupported Database"
images = os.listdir(args.data_dir)
evaluator = Evaluator(num_class=2)
evaluator.reset()
pbar = tqdm(images)
geo_polygons = []
ct = 0

preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
preprocessing = get_preprocessing(preprocessing_fn)
net = MyModel(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid'
    )
print(sum(p.numel() for p in net.parameters()))

net = torch.load(args.model_path, map_location=device)
net.eval()

with torch.no_grad():
    for file in pbar:
        img = np.load(os.path.join(args.data_dir, file))
        gt_mask = np.load(os.path.join(args.data_dir, file).replace(".npy", "_mask.npy").replace("images_preprocessed",
                                                                                                 "masks"))
        if not args.use_nepagg:
            if args.db != supported_dbs[-1]:
                img = img[64:576, 64:576]
            else:
                img = img[83:733, 83:733]
                img = cv2.copyMakeBorder(img, 0, 6, 0, 6, cv2.BORDER_CONSTANT, value=0)

        sample = preprocessing(image=img)
        img = torch.Tensor(sample['image']).permute(2, 0, 1).to(device).unsqueeze(0)
        output, _ = net(img)
        mask_array = output.round().squeeze().detach().cpu().numpy()

        # if args.db == supported_dbs[-1]:
        #     mask_array = mask_array[3:-3, 3:-3]

        if args.use_nepagg:
            if args.db != supported_dbs[-1]:
                mask_array = mask_array[64:576, 64:576]
            else:
                mask_array = mask_array[83:733, 83:733]

        pr_mask = mask_array.astype(np.uint8)
        evaluator.add_batch(pre_image=pr_mask, gt_image=gt_mask)
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
        if args.db == supported_dbs[-1]:
            path_split = args.tiff_images_path.split(os.sep)
            name = f"{path_split[-1]}_{path_split[-2]}_img" + _id + '.tif'
        elif args.db == supported_dbs[0]:
            if args.partition == 'test2':
                name = '2_' + _id + '.tif'
            elif args.partition == 'val':
                name = 'val_' + _id + '.tif'
            else:
                name = _id + '.tif'
        else:
            name = _id + '.tif'
        geo_poly = (assign_geocoords(np_polygons, os.path.join(args.tiff_images_path, name)))
        geo_polygons.extend(geo_poly)

if not os.path.exists(r"D:\Ahmad\3090 Data\Predictions\TFNet\{}\{}".format(args.db, args.partition)):
    os.makedirs(r"D:\Ahmad\3090 Data\Predictions\TFNet\{}\{}".format(args.db, args.partition))
save_as_csv(r"D:\Ahmad\3090 Data\Predictions\TFNet\{}\{}".format(args.db, args.partition), geo_polygons,
            db=args.db, partition=args.partition)
iou_per_class = evaluator.Intersection_over_Union()
f1_per_class = evaluator.F1()
OA = evaluator.OA()
precision = evaluator.Precision()
recall = evaluator.Recall()
print('F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(f1_per_class[1], iou_per_class[1], OA, precision[1], recall[1]))
fw = open(r"D:\Ahmad\3090 Data\Predictions\TFNet\{}\{}\pixel_acc.txt".format(args.db, args.partition), 'w')
fw.write('F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(f1_per_class[1], iou_per_class[1], OA, precision[1], recall[1]))
fw.close()
print("Complete")