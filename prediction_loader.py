from abc import ABC, abstractmethod
import os
import h5py
import numpy as np


class PredictionLoader(ABC):
    @abstractmethod
    def get_pred_r(self, id, space):
        '''Please implement in subclass'''
        raise NotImplemented


class Li_2018_CGI_Loader(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir

    def get_pred_r(self, id, space):
        assert space in ["srgb"]
        pred_path = os.path.join(self.dir, f"{id}.png.h5")
        hdf5_file_read = h5py.File(pred_path, 'r')
        pred_R = hdf5_file_read.get('/prediction/R')
        pred_R = np.array(pred_R)
        # pred_S = hdf5_file_read.get('/prediction/S')
        # pred_S = np.array(pred_S)
        hdf5_file_read.close()
        return pred_R
