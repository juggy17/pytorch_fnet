from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import argparse
import warnings
warnings.filterwarnings("ignore")
import scipy
from fnet.data.bufferedpatchdataset import BufferedPatchDataset
from PIL import Image

def normalize(img):
    """Subtract mean, set STD to 1.0"""
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

class Resizer(object):
    def __init__(self, factors):
        """
        factors - tuple of resizing factors for each dimension of the input array"""
        self.factors = factors

    def __call__(self, x):
        return scipy.ndimage.zoom(x, (self.factors), mode='nearest')

    def __repr__(self):
        return 'Resizer({:s})'.format(str(self.factors))


# class for dataset
class TIFdataset(Dataset):
    def __init__(self, csvFile, transform=None):
        self.ds = pd.read_csv(csvFile)        
        self.transform = transform 
        assert all(i in self.ds.columns for i in ['path_tif_signal', 'path_tif_target']) 
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        element = self.ds.iloc[index, :]
        
        signalFile = element['path_tif_signal'] + '.tif'
        targetFile = element['path_tif_target'] + '.tif'

        signal = io.imread(signalFile)[:508, :397]
        target = io.imread(targetFile)[:508, :397]
                        
        im_out = list()
        im_out.append(signal)
        im_out.append(target)

        print(signal.shape)
        
        if self.transform is not None:
            for t in self.transform: 
                im_out[0] = eval(t)(im_out[0])

        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        return im_out        
