from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
from .models import networks
from .options.inference_options import InferenceOptions
import sys
from .data.data_loader import *
from .models.models import create_model
import random
from tensorboardX import SummaryWriter
import os, json, cv2, random

# EVAL_BATCH_SIZE = 8
# opt = InferenceOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

# eval_num_threads = 3
# test_data_loader = CreateInferenceDataLoader(opt, opt.img_path,
#                                                 False, EVAL_BATCH_SIZE,
#                                                 eval_num_threads)
# test_dataset = test_data_loader.load_data()
# test_data_size = len(test_data_loader)

# model = create_model(opt, _isTrain=False)
# model.switch_to_train()

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# global_step = 0


def infer_model_on_image(model, dataset, global_step):
    rot_e_list = []
    roll_e_list = []
    pitch_e_list = []

    count = 0.0

    model.switch_to_eval()

    count = 0

    r_up_vector = None
    r_roll = None
    r_pitch = None

    for i, data in enumerate(dataset):
        stacked_img = data[0]
        targets = data[1]

        est_up_n, pred_roll, pred_pitch = model.infer_model(stacked_img, targets)

        print('**********************************')
        print('**********************************')
        print('**********************************')
        print('CAM UP VEC3', est_up_n)
        print('PRED ROLL', pred_roll)
        print('PRED PITCH', pred_pitch)
        print('**********************************')
        print('**********************************')
        print('**********************************')
        r_up_vector = est_up_n
        r_roll = pred_roll
        r_pitch = pred_pitch
    
    return r_up_vector, r_roll, r_pitch

# infer_model_on_image(model, test_dataset, global_step)

# call the required steps to run the inferrence
def run_inference(image_path, dataset):
    # base options
    opt = SetupOpts().setup(image_path, dataset)

    EVAL_BATCH_SIZE = 8
    eval_num_threads = 3
    test_data_loader = CreateInferenceDataLoader(opt, opt.img_path,
                                                    False, EVAL_BATCH_SIZE,
                                                    eval_num_threads)
    test_dataset = test_data_loader.load_data()
    test_data_size = len(test_data_loader)

    model = create_model(opt, _isTrain=False)
    model.switch_to_train()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    global_step = 0

    # call infer_model_on_image
    up, roll, pitch = infer_model_on_image(model, test_dataset, global_step)

    # return results
    return up.tolist(), roll, pitch

class SetupOpts:
    def __init__(self):
        self.gpu_ids=[0]
        self.name='test_local'
        self.nThreads=2
        self.checkpoints_dir='./UprightNet/checkpoints/'
        self.log_comment='exp_upright_9_sphere_ls'
        self.mode='ResNet'

    def setup(self, image_path, dataset):
        self.dataset=dataset
        self.img_path=image_path
        self.isTrain=True
        return self