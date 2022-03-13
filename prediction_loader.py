from abc import ABC, abstractmethod
import os
import h5py
import numpy as np

import util


class PredictionLoader(ABC):
    raw_dir = None
    image_dir = None
    img_postfix = None

    @abstractmethod
    def get_pred_r(self, id, space):
        '''Please implement in subclass'''
        raise NotImplemented

    @abstractmethod
    def get_pred_rs_img_path(self, id):
        '''Please implement in subclass'''
        raise NotImplemented

    def set_img_dir(self, img_dir, img_postfix):
        assert img_postfix in ["png", "jpg", "jpeg"]
        self.image_dir = img_dir
        self.img_postfix = img_postfix


class Li_2018_CGI_Loader(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir
        self.raw_dir = os.path.join(self.dir, "release_iiw")
        self.image_dir = os.path.join(self.dir, "release_iiw_images")
        self.img_postfix = "png"

    def get_pred_r(self, id, space):
        assert space in ["srgb"]
        pred_path = os.path.join(self.raw_dir, f"{id}.png.h5")
        hdf5_file_read = h5py.File(pred_path, 'r')
        pred_R = hdf5_file_read.get('/prediction/R')
        pred_R = np.array(pred_R)
        # pred_S = hdf5_file_read.get('/prediction/S')
        # pred_S = np.array(pred_S)
        hdf5_file_read.close()
        return pred_R

    def get_pred_rs_img_path(self, id):
        r_img_path = os.path.join(self.image_dir, f"{id}-r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}-s.{self.img_postfix}")
        return r_img_path, s_img_path


class Luo_2020_NIID_Net_Loader(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir
        self.raw_dir = os.path.join(self.dir, "final_raw")
        self.image_dir = os.path.join(self.dir, "SAW_pred_imgs")
        self.img_postfix = "png"

    def get_pred_r(self, id, space):
        assert space in ["srgb"]
        pred_R_path = os.path.join(self.raw_dir, f"{id}-r.npy")
        pred_R = np.load(pred_R_path).astype(np.float32)
        pred_R = util.rgb_to_srgb(pred_R)
        return pred_R

    def get_pred_rs_img_path(self, id):
        r_img_path = os.path.join(self.image_dir, f"{id}_R.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_S.{self.img_postfix}")
        return r_img_path, s_img_path


class CRefNet(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir
        self.image_dir = os.path.join(self.dir, "split")
        self.img_postfix = "jpg"

    def get_pred_r(self, id, space):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_pred_rs_img_path(self, id):
        r_img_path = os.path.join(self.image_dir, f"{id}_r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_s.{self.img_postfix}")
        return r_img_path, s_img_path


class Wang_2019_Discriminative_Loader(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir
        self.image_dir = os.path.join(self.dir, "test-imgs_ep12_results")
        self.img_postfix = "png"

    def get_pred_r(self, id, space):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_pred_rs_img_path(self, id):
        r_img_path = os.path.join(self.image_dir, f"{id}_r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}_sr.{self.img_postfix}")
        return r_img_path, s_img_path


class Bi_2015_L1smoothing_Loader(PredictionLoader):
    def __init__(self, dir):
        self.dir = dir
        self.image_dir = os.path.join(self.dir, "our_result")
        self.img_postfix = "png"

    def get_pred_r(self, id, space):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_pred_rs_img_path(self, id):
        r_img_path = os.path.join(self.image_dir, f"{id}-r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}-s.{self.img_postfix}")
        return r_img_path, s_img_path


class General_Loader(PredictionLoader):
    def __init__(self, dir, img_postfix="png"):
        assert img_postfix in ["png", "jpeg", "jpg"]
        self.dir = dir
        self.image_dir = self.dir
        self.img_postfix = img_postfix

    def get_pred_r(self, id, space):
        '''Please implement in subclass'''
        raise NotImplemented

    def get_pred_rs_img_path(self, id):
        r_img_path = os.path.join(self.image_dir, f"{id}-r.{self.img_postfix}")
        s_img_path = os.path.join(self.image_dir, f"{id}-s.{self.img_postfix}")
        return r_img_path, s_img_path


class InputLoader(object):
    def __init__(self, dir):
        self.dir = dir
        self.data_dir = os.path.join(self.dir, "data")
        self.img_postfix = "png"

    def get_input_img_path(self, id):
        return os.path.join(self.data_dir, f"{id}.{self.img_postfix}")

    def set_img_dir(self, data_dir, img_postfix):
        assert img_postfix in ["png", "jpg", "jpeg"]
        self.data_dir = data_dir
        self.img_postfix = img_postfix
