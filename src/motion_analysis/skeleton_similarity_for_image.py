from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path))

import os
import cv2
import numpy as np

import time
from tqdm import tqdm

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory

time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


class sk_sim:
    def __init__(self, opt):
        self.Detector = detector_factory['multi_pose']
        self.detector = self.Detector(opt)
        self.resize = opt.resize
        self.resolution = opt.resolution
        print('resize: {}'.format(self.resize))
        print('resolution: {}'.format(self.resolution))


    
    def _similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    

    # 计算std
    def _cal_std(self, sk_res):
        degree_matrix = []
        for line in sk_res:
            tmp_line = line[15:27]

            src_skeleton = np.reshape(tmp_line, (-1, 2))

            skeleton = []
            skeleton.append(src_skeleton[5])
            skeleton.append(src_skeleton[3])
            skeleton.append(src_skeleton[1])
            skeleton.append(src_skeleton[0])
            skeleton.append(src_skeleton[2])
            skeleton.append(src_skeleton[4])

            skeleton = np.array(skeleton)

            vector = []
            for i in range(4):
                v1 = skeleton[i] - skeleton[i+1]
                v2 = skeleton[i+2] - skeleton[i+1]
                _sim = self._similarity(v1, v2)
                degree = np.degrees(np.arccos(_sim))
                vector.append(degree)

            horizontal_vector = skeleton[3]- skeleton[2]
            horizontal_sim = self._similarity(horizontal_vector, [1, 0])
            horizontal_degree = np.degrees(np.arccos(horizontal_sim))
            vector.append(horizontal_degree)

            degree_matrix.append(vector)

        frame_sim_v = []
        for i in range(len(degree_matrix) - 1):
            sim = self._similarity(degree_matrix[i], degree_matrix[i+1])
            frame_sim_v.append(sim)

        std = np.nanstd(frame_sim_v)

        return std


    def run_images(self, image_paths):
        start_time = time.time()


        total_frame = len(image_paths)
        print('total_frame: {}'.format(total_frame))
        
        frame_0 = cv2.imread(image_paths[0])
        need_resize = False
        if self.resize:
            w = int(frame_0.shape[1])
            h = int(frame_0.shape[0])
            if min(w, h) > self.resolution:
                need_resize = True
                if w > h:
                    resized_h = self.resolution
                    resized_w = int(w * self.resolution / h)
                else:
                    resized_w = self.resolution
                    resized_h = int(h * self.resolution / w)

        pbar = tqdm(total = total_frame)
        sk_res = []
        for image_path in image_paths:
            pbar.update(1)
            frame = cv2.imread(image_path)

            if frame is not None:
               
                if need_resize:
                   frame = cv2.resize(frame, (resized_w, resized_h))
   
                ret = self.detector.run(frame)

                sk_res.append(ret['results'][1][0])
            else:
                continue


        pbar.close()
        sk_std = self._cal_std(sk_res)

        end_time = time.time() - start_time

        if sk_std > 0.13:
            sk_std = -1

        return sk_std, end_time


opt = opts()
sk = sk_sim(opt)

if __name__ == '__main__':
    # 输入请求图片集的路径，获取图片给路径列表
    image_dir = r'/home/guoweiye/workspace/video_tag/app/algorithm/motion_analysis/images/'
    image_list = []
    image_names = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, image_name) for image_name in image_names]

    # skstd：表情丰富度，ti：耗时
    skstd, ti = sk.run_images(image_paths)
    print(skstd, ti)

