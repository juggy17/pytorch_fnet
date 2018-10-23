from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
import argparse
import warnings
warnings.filterwarnings("ignore")
import scipy
from .bufferedpatchdataset import BufferedPatchDataset
from PIL import Image
from .. import transforms

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
    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, 
                    transform_source = [transforms.normalize],
                    transform_target = None):
                
        if dataframe is not None:
            self.ds = dataframe
        else:
            self.ds = pd.read_csv(path_csv)
            
        self.transform_source = transform_source
        self.transform_target = transform_target
        
        assert all(i in self.ds.columns for i in ['path_tif_signal', 'path_tif_target']) 
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        element = self.ds.iloc[index, :]
        
        signalFile = element['path_tif_signal']
        targetFile = element['path_tif_target']

        signal = io.imread(signalFile)
        target = io.imread(targetFile)
                        
        im_out = list()        
        im_out.append(signal)
        im_out.append(target)

        if self.transform_source is not None:
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])

        if self.transform_target is not None:
            for t in self.transform_target: 
                im_out[1] = t(im_out[1])

        print(signal.shape)

        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]
        
        #unsqueeze to make the first dimension be the channel dimension
        im_out = [torch.unsqueeze(im, 0) for im in im_out]
        return im_out        
