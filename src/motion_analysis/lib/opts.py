from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

class opts:
  def __init__(self):
    self.fps = 3
    self.resize = True
    self.resolution = 540
    self.task = 'multi_pose'
    file_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(file_path, '../models/multi_pose_dla_3x.pth')
    self.load_model = model_path

    
    self.K = 10
    self.arch = 'dla_34'
    self.dataset = 'coco_hp'
    self.dense_hp = False
    self.down_ratio = 4
    self.fix_res = True
    self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    self.flip_test = False
    self.gpus = [0]
    self.gpus_str = '0'
    self.head_conv = 256
    self.heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
    self.hm_hp = True
    self.hm_hp_weight = 1
    self.hp_weight = 1
    self.input_h = 512
    self.input_res = 512
    self.input_w = 512
    self.keep_res = False
    self.mean = [0.408, 0.447, 0.47]
    self.mse_loss = False
    self.not_hm_hp = False
    self.not_reg_bbox = False
    self.not_reg_hp_offset = False
    self.not_reg_offset = False
    self.num_classes = 1
    self.num_stacks = 1
    self.output_h = 128
    self.output_res = 128
    self.output_w = 128
    self.pad = 31
    self.reg_bbox = True
    self.reg_hp_offset = True
    self.reg_offset = True
    self.std = [0.289, 0.274, 0.278]
    self.test_scales = [1.0]