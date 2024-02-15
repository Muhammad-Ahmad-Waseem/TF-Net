import os
import numpy as np
import matplotlib.pyplot as plt
from model import DeepLabV3Plus as MyModel
import torch
import segmentation_models_pytorch as smp
from train import get_preprocessing
from skimage.segmentation import find_boundaries
import cv2



img_path = "/mnt/Ahmad/TeamUrbanTechDHA_2/Preprocessed/train/boundaries/1903_boundary.npy"
model_path = "/mnt/Ahmad/Models/Segmentation/trained_model/our_final_WHU"

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
preprocessing = get_preprocessing(preprocessing_fn)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVATION = 'sigmoid'
model = MyModel(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION
    )
model = torch.load(os.path.join(model_path, 'best_model.h5'), map_location=DEVICE)
model.eval()
print(model)

img = np.load(img_path)
# sample = preprocessing(image=img)
# img = torch.Tensor(sample['image']).permute(2, 0, 1).to(DEVICE).unsqueeze(0)
#
# with torch.no_grad():
#     pred_mask, pred_edge = model(img)
#
# pr_mask = pred_mask.round().squeeze()
# pr_mask = pr_mask.detach().squeeze().cpu().numpy()
#
# pred_edge = pred_edge.round().squeeze()
# pred_edge = pred_edge.detach().squeeze().cpu().numpy()
boundary = find_boundaries(img, mode='inner')
ZZ = np.zeros((512, 512))
ZZ[boundary] = 1
kernel = np.ones((7, 7), np.uint8)
boundary = cv2.dilate(ZZ, kernel, iterations=1)
boundary = boundary * img

boundary = boundary.astype(np.uint8)

plt.imshow(boundary)
plt.show()

plt.imsave("/home/gcf/Desktop/img.png",boundary)