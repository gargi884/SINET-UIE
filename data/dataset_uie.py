import os.path
import os
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    
class Dataset(data.Dataset): 
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, opt):
        super(Dataset, self).__init__()
        print('Dataset for UIE task.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 256

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.dataroot_A = opt['dataroot_A']
        self.dataroot_B = opt['dataroot_B']
        names = []
        for dirpath, _, fnames in sorted(os.walk(self.dataroot_A)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    names.append(fname)
                    
        self.names_A = names            

    def __getitem__(self, index):

        # ------------------------------------
        # get image pair
        # ------------------------------------
        A_path = self.dataroot_A+'/'+self.names_A[index] #self.paths_A[index]
        B_path = self.dataroot_B+'/'+self.names_A[index]
        img_A = util.imread_uint1(A_path, self.n_channels, self.patch_size)
        img_B = util.imread_uint1(B_path, self.n_channels, self.patch_size)

        if self.opt['phase'] == 'train': 
            """
            # --------------------------------
            # get patch pairs
            # --------------------------------
            """
            H, W, _ = img_A.shape
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_A = img_A[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]
            patch_B = img_B[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,:]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0,7)
            
            patch_A, patch_B = util.augment_img(patch_A, mode=mode), util.augment_img(patch_B, mode=mode)
            img_A = util.uint2tensor3(patch_A)
            img_B = util.uint2tensor3(patch_B)
            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

        else: 
            """
            # --------------------------------
            # get image pairs
            # --------------------------------
            """
            img_A = util.uint2single(img_A)
            img_B = util.uint2single(img_B)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_A = util.single2tensor3(img_A)
            img_B = util.single2tensor3(img_B)

            return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.names_A)
