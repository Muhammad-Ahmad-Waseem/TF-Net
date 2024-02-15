import glob
import os
import numpy as np
import cv2
import albumentations
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# classes for data loading and preprocessing
class Dataset:
    """ Dataset Class. Read images, apply augmentation and preprocessing transformations.
        This dataset class is assuming you have image and its corresponding mask + boundary.
    Args:
        base_dir (str): path to base folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
        size (int): size needed for image
        db (str): name of database/dataset to use
        train (bool): Load training or testing (valid) portion
        split (float): A value between 0 and 1, representing ratio of training portion
    """

    # CLASSES = ['built-up', 'background']
    supported_dbs = ['SpaceNet', 'WHU', 'CITY_DHA']
    def __init__(
            self,
            base_dir,
            augmentation=None,
            preprocessing=None,
            # size=656,
            db='SpaceNet',
            train=True,
            split=0.8,
            seed_value=7120
    ):

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        # self.size = size
        self.db = db
        self.train = train

        if self.db == self.supported_dbs[0]:
            path_to_imgs = base_dir+"/*/images_preprocessed/*.npy"
            image_ids = glob.glob(path_to_imgs)

            np.random.seed(seed_value)
            image_ids = np.random.permutation(image_ids)

            self.image_paths = [os.path.join(base_dir, image_id) for image_id in image_ids]
            self.mask_paths = [os.path.join(base_dir,
                    image_id.replace(".npy", "_mask.npy").replace("images_preprocessed", "masks")) for image_id in image_ids]
            self.edge_paths = [os.path.join(base_dir,
                                            image_id.replace(".npy", "_boundary.npy").replace("images_preprocessed", "new_boundaries"))
                               for image_id in image_ids]

            length = len(self.image_paths)
            # print(length)
            if self.train:
                self.image_paths = self.image_paths[:int(split*length)]
                self.mask_paths = self.mask_paths[:int(split * length)]
                self.edge_paths = self.edge_paths[:int(split * length)]

            else:
                self.image_paths = self.image_paths[int(split * length):]
                self.mask_paths = self.mask_paths[int(split * length):]
                self.edge_paths = self.edge_paths[int(split * length):]
        elif self.db == self.supported_dbs[1]:
            if self.train:
                path_to_imgs = base_dir + "/train/images_preprocessed/*.npy"
                image_ids = glob.glob(path_to_imgs)
                # print(image_ids[:10])
                np.random.seed(seed_value)
                image_ids = np.random.permutation(image_ids)
                self.image_paths = [os.path.join(base_dir, image_id) for image_id in image_ids]
                self.mask_paths = [os.path.join(base_dir,
                                    image_id.replace(".npy", "_mask.npy").replace("images_preprocessed", "masks"))
                                    for image_id in image_ids]
                self.edge_paths = [os.path.join(base_dir,
                                                image_id.replace(".npy", "_boundary.npy").replace("images_preprocessed",
                                                                                                  "new_boundaries"))
                                   for image_id in image_ids]
            else:
                path_to_imgs = base_dir + "/val/images_preprocessed/*.npy"
                image_ids = glob.glob(path_to_imgs)
                # print(image_ids[:10])
                np.random.seed(seed_value)
                image_ids = np.random.permutation(image_ids)
                self.image_paths = [os.path.join(base_dir, image_id) for image_id in image_ids]
                self.mask_paths = [os.path.join(base_dir,
                                                image_id.replace(".npy", "_mask.npy").replace("images_preprocessed",
                                                                                              "masks"))
                                   for image_id in image_ids]
                self.edge_paths = [os.path.join(base_dir,
                                                image_id.replace(".npy", "_boundary.npy").replace("images_preprocessed",
                                                                                                  "boundaries"))
                                   for image_id in image_ids]
        elif self.db == self.supported_dbs[2]:
            if self.train:
                path_to_imgs = base_dir + "/train/images_preprocessed/*.npy"
                image_ids = glob.glob(path_to_imgs)
                # print(image_ids[:10])
                np.random.seed(seed_value)
                image_ids = np.random.permutation(image_ids)
                self.image_paths = [os.path.join(base_dir, image_id) for image_id in image_ids]
                self.mask_paths = [os.path.join(base_dir,
                                    image_id.replace(".npy", "_mask.npy").replace("images_preprocessed", "masks"))
                                    for image_id in image_ids]
                self.edge_paths = [os.path.join(base_dir,
                                                image_id.replace(".npy", "_boundary.npy").replace("images_preprocessed",
                                                                                                  "boundaries"))
                                   for image_id in image_ids]
            else:
                assert True, "Validation split not defined for this dataset"
                # path_to_imgs = base_dir + "/val/images_preprocessed/*.npy"
                # image_ids = glob.glob(path_to_imgs)
                # # print(image_ids[:10])
                # np.random.seed(seed_value)
                # image_ids = np.random.permutation(image_ids)
                # self.image_paths = [os.path.join(base_dir, image_id) for image_id in image_ids]
                # self.mask_paths = [os.path.join(base_dir,
                #                                 image_id.replace(".npy", "_mask.npy").replace("images_preprocessed",
                #                                                                               "masks"))
                #                    for image_id in image_ids]
                # self.edge_paths = [os.path.join(base_dir,
                #                                 image_id.replace(".npy", "_boundary.npy").replace("images_preprocessed",
                #                                                                                   "boundaries"))
                #                    for image_id in image_ids]

    def __getitem__(self, i):
        assert self.db in self.supported_dbs, "The provided database: {} is not supported".format(self.db)

        # read data
        image = np.load(os.path.join(self.image_paths[i]))
        image = image[64:576, 64:576]
        # image = image[83:733, 83:733]
        mask = np.load(os.path.join(self.mask_paths[i]))
        mask = (mask > 0).astype(np.int32)
        edge = np.load(os.path.join(self.edge_paths[i]))
        edge = (edge > 0).astype(np.int32)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask, edge=edge)
            image, mask, edge = sample['image'], sample['mask'], sample['edge']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask, edge=edge)
            image, mask, edge = sample['image'], sample['mask'], sample['edge']

        image = np.transpose(image, (2, 0, 1)).astype('float32')

        return image, mask, edge, self.image_paths[i]

    def __len__(self):
        return len(self.image_paths)