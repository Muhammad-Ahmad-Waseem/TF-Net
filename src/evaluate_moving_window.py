import torch
import os
import glob
import numpy as np
import rasterio
import math
import cv2
import segmentation_models_pytorch as smp
from model import DeepLabV3Plus as MyModel
from train import get_preprocessing


input_imgs = "/mnt/Ahmad/Tiff_files/SN_Shanghai.tif"
model_path = "/mnt/Ahmad/Models/Segmentation/trained_model/our_final_CITY_small"
output_dir = "/mnt/Ahmad/Results/Zoom21/SpaceNet_Shanghai/our_final_CITY_small"

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'

target_size = (816, 816)
padding_pixels = (83, 83)
padding_value = 0
downsampling_factor = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIX_VALUE_MAX = 2048  # The max data value we have
PIX_VALUE_MAX_REQ = 255  # The max data value we need

model = MyModel(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION
    )
model = torch.load(os.path.join(model_path, 'best_model.h5'), map_location=DEVICE)
model.eval()
imgs = [file for file in glob.glob(input_imgs) if file.endswith('.tif')]
print(imgs)
assert len(imgs) > 0, "The number of images equal to zero"


print("Running on {} images".format(len(imgs)))  # using {} parallel processes".format(len(imgs), num_processes))

for img_path in imgs:
    file_name = os.path.split(img_path)[-1].split('.')[0]
    print("Running for {}".format(file_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, file_name + "_preds.npy")
    img = np.transpose(rasterio.open(img_path).read(), (1, 2, 0))
    #     max_value = np.iinfo(img.dtype).max
    print("Actual Image Size: {}".format(img.shape))

    # Define k_x, k_y to define 'useful' portion, since we are taking patches with overlapping area.
    k_y = target_size[0] - 2 * padding_pixels[0]
    k_x = target_size[1] - 2 * padding_pixels[1]

    # First padding: To make divisible by k
    cols = (math.ceil(img.shape[0] / k_y))
    rows = (math.ceil(img.shape[1] / k_x))

    pad_bottom = cols * k_y - img.shape[0]  # pixels to add in y direction
    pad_right = rows * k_x - img.shape[1]  # pixels to add in x direction
    if pad_bottom > 0 or pad_right > 0:
        print("Running cv2 padding..")
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=padding_value)
    print("Image Size after making divisible by ({}, {}): {}".format(k_x, k_y, img.shape))

    output_image = np.zeros((int(img.shape[0] * downsampling_factor), int(img.shape[1] * downsampling_factor)),
                            dtype=np.uint8) * 255
    print("Size of output image after downsampling factor of {}: {}".format(downsampling_factor, output_image.shape))

    # Second Padding: To add boundary padding pixels
    img = cv2.copyMakeBorder(img, padding_pixels[0], padding_pixels[0], padding_pixels[1],
                             padding_pixels[1], cv2.BORDER_CONSTANT, value=padding_value)
    print(
        "Image Size after adding ({}, {}) boundary pixels: {}".format(padding_pixels[0], padding_pixels[0], img.shape))

    # Load pre-processing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)

    total_patches = rows * cols
    print("Total {} patches for the given image {}".format(rows * cols, file_name))
    for y_idx in range(cols):
        y1 = y_idx * k_y + padding_pixels[0]
        y2 = y1 + k_y
        #         if y_idx <= 0:
        #             continue
        for x_idx in range(rows):
            x1 = x_idx * k_x + padding_pixels[1]
            x2 = x1 + k_x
            patch_number = y_idx * rows + x_idx + 1

            img_crop = img[y1 - padding_pixels[0]: y2 + padding_pixels[0],
                       x1 - padding_pixels[1]: x2 + padding_pixels[1]]
            print("Patch {} of {}: [{}:{}, {}:{}]".format(patch_number, total_patches, y1 - padding_pixels[0],
                                                          y2 + padding_pixels[0], x1 - padding_pixels[1],
                                                          x2 + padding_pixels[1]), end=" ")

            img_crop = ((img_crop / PIX_VALUE_MAX) * (PIX_VALUE_MAX_REQ)).astype(np.uint8)
            sample = preprocessing(image=img_crop)
            image = cv2.resize(sample['image'],
                               (int(downsampling_factor * target_size[0]),
                                int(downsampling_factor * target_size[1])),
                               interpolation=cv2.INTER_AREA)
            x_tensor = torch.Tensor(image).permute(2, 0, 1).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                pred_mask, _ = model(x_tensor)
            pr_mask = pred_mask.squeeze()
            pr_mask = pr_mask.detach().squeeze().cpu().numpy().round()

            patch = pr_mask[int(downsampling_factor * padding_pixels[0]): int(downsampling_factor * (target_size[0]
                    -padding_pixels[0])), int(downsampling_factor * padding_pixels[1]): int(downsampling_factor *
                    (target_size[1] - padding_pixels[1]))]
            output_image[int(downsampling_factor * (y_idx * k_y)): int(downsampling_factor * (y_idx * k_y + k_y)),
            int(downsampling_factor * (x_idx * k_x)): int(downsampling_factor * (x_idx * k_x + k_x))] = patch
            print("..... Done!")
#         break


output_image = output_image[:output_image.shape[0] - int(downsampling_factor * pad_bottom),
               :output_image.shape[1] - int(downsampling_factor * pad_right)]
print("Final shape of downsampled output image: {}".format(output_image.shape))

np.save(save_path, output_image)
print("Completed!")