import numpy as np
from matplotlib import pyplot as plt
import pyexr
from scipy import ndimage
import random
from random import randint
import os

def preprocess_data(exr_path, gt_path):
    data = {}

    # high spp
    types = ['.hdr.exr'] #['specular.exr', 'diffuse.exr', 'color.exr']
    names = ['gt'] #['spec_gt', 'diff_gt', 'gt']
    for type, name in zip(types, names):
        path = gt_path + type
        exr = pyexr.open(path)
        d = exr.get_all()
        data[name] = d['default']
    # low spp
    types = [".hdr.exr"] #[".alb.exr", ".nrm.exr", ".hdr.exr"] #['specular.exr', 'diffuse.exr', 'normal.exr', 'depth.exr', 'albedo.exr', 'color.exr']
    names = ['noisy'] #['albedo', 'normal', 'noisy'] #['spec', 'diff', 'normal', 'depth', 'albedo', 'noisy']
    for type, name in zip(types, names):
        path = exr_path + type
        noisy_exr = pyexr.open(path)
        d = noisy_exr.get_all()
        data[name] = d['default']

    # nan to 0.0, inf to finite number
    for channel_name, channel_value in data.items():
        data[channel_name] = np.nan_to_num(channel_value)

    data['gt'] = np.clip(data['gt'], 0, np.max(data['gt']))
    data['noisy'] = np.clip(data['noisy'], 0, np.max(data['noisy']))


    return data