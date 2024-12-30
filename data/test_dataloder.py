import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class Dataset(data.Dataset): 
    def __init__(self, root_A, in_channelsA):
        super(Dataset, self).__init__()
        self.paths_A = util.get_image_paths(root_A)
        self.inchannelsA = in_channelsA

    def __getitem__(self, index):

        A_path = self.paths_A[index]
        img_A = util.imread_uint(A_path, self.inchannelsA)
        
        """
        # --------------------------------
        # get testing image 
        # --------------------------------
        """
        img_A = util.uint2single(img_A)
        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_A = util.single2tensor3(img_A)

        return {'A': img_A, 'A_path': A_path}

    def __len__(self):
        return len(self.paths_A)
