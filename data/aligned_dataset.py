import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import torch

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = cv.imread(A_path).transpose(2, 0, 1) / 255.0  # optical

        if self.inchannel == 4:
            B_img = cv.imread(B_path, cv.IMREAD_LOAD_GDAL).transpose(2, 0, 1)/100  # SAR, using for spacenet with 4-channel SAR
            A_img, B_img = self.center_crop(A_img, B_img, 512)  # for spacenet dataset
        else:
            B_img = cv.imread(B_path).transpose(2, 0, 1)/255.0 # SAR, using for dataset with 1/3-channel SAR
            
        A = torch.from_numpy(A_img.astype(np.float32))
        B = torch.from_numpy(B_img.astype(np.float32))
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""

        return len(self.A_paths)

    def random_crop(self, sar, rgb, crop_size):
        h = np.random.randint(0, sar.shape[1] - crop_size)
        w = np.random.randint(0, sar.shape[2] - crop_size)
        sar = sar[:, h:h + crop_size, w:w + crop_size]
        rgb = rgb[:, h:h + crop_size, w:w + crop_size]
        return sar, rgb

    def center_crop(self, sar, rgb, crop_size):
        sar = sar[:, 194:194 + crop_size, 194:194 + crop_size]
        rgb = rgb[:, 194:194 + crop_size, 194:194 + crop_size]
        return sar, rgb









