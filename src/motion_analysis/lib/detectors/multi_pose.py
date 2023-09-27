from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

from lib.models.decode import multi_pose_decode
from lib.models.utils import flip_tensor, flip_lr_off, flip_lr
from lib.utils.image import get_affine_transform
from lib.utils.post_process import multi_pose_post_process

from .base_detector import BaseDetector

class MultiPoseDetector(BaseDetector):
  def __init__(self, opt):
    super(MultiPoseDetector, self).__init__(opt)
    self.flip_idx = opt.flip_idx

  def process(self, images, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(images)[-1]
      output['hm'] = output['hm'].sigmoid_()
      if self.opt.hm_hp and not self.opt.mse_loss:
        output['hm_hp'] = output['hm_hp'].sigmoid_()

      reg = output['reg'] if self.opt.reg_offset else None
      hm_hp = output['hm_hp'] if self.opt.hm_hp else None
      hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      dets = multi_pose_decode(
        output['hm'], output['wh'], output['hps'],
        reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    results[1] = results[1].tolist()
    return results
