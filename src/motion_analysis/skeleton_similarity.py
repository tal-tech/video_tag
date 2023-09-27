#!/usr/bin/env python
#-*-coding:GBK -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import skvideo.io
import numpy as np

import time
import json
from tqdm import tqdm

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory

time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


class sk_sim:
    def __init__(self, opt):
        self.Detector = detector_factory['multi_pose']
        self.detector = self.Detector(opt)
        self.fps = opt.fps
        self.resize = opt.resize
        self.resolution = opt.resolution
        print('FPS: {}'.format(self.fps))
        print('resize: {}'.format(self.resize))
        print('resolution: {}'.format(self.resolution))


    def update_address(self, addr):
        self.address = addr


    def _get_rotation_info(self, addr):
        metadata = skvideo.io.ffprobe(addr)

        rotate = 0
        for line in metadata['video']['tag']:
            if line['@key'] == 'rotate':
                rotate = int(line['@value'])
                break

        return rotate


    def _similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


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

            # debug
            # white_img = np.ones([1920//2, 1920//2, 3]) * 255
            # for idx, line in enumerate(skeleton):
            #     cv2.putText(white_img, str(idx), (int(line[0]//2), int(line[1]//2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            #     cv2.circle(white_img, (int(line[0]//2), int(line[1]//2)), 3, (255, 0, 0), -1)

            # cv2.imwrite('./debug/' + str(index) + '.jpg', white_img)
            # debug

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


    def run_video(self):
        start_time = time.time()

        video = cv2.VideoCapture(self.address)
        video_fps = video.get(5)
        print('video_fps: {}'.format(video_fps))

        total_frame = video.get(7)
        print('total_frame: {}'.format(total_frame))

        frame_interval = np.rint(video_fps / self.fps)
        print('frame_interval: {}'.format(frame_interval))

        rotate = self._get_rotation_info(self.address)
        print('rotation info: {}'.format(rotate))

        need_resize = False
        if self.resize:
            w = int(video.get(3))
            h = int(video.get(4))
            if min(w, h) > self.resolution:
                need_resize = True
                if w > h:
                    resized_h = self.resolution
                    resized_w = int(w * self.resolution / h)
                else:
                    resized_w = self.resolution
                    resized_h = int(h * self.resolution / w)

        pbar = tqdm(total = total_frame)
        frame_count = 0
        sk_res = []
        while True:
            pbar.update(1)
            ret, frame = video.read()

            if ret:
                if frame_count % frame_interval == 0:
                    if need_resize:
                        frame = cv2.resize(frame, (resized_w, resized_h))
                    if rotate == 0:
                        frame_ro = frame
                    elif rotate == 90:
                        frame_ro = cv2.rotate(frame, 0)
                    elif rotate == 180:
                        frame_ro = cv2.rotate(frame, 1)
                    elif rotate == 270:
                        frame_ro = cv2.rotate(frame, 2)
                    ret = self.detector.run(frame_ro)
                    # time_str = ''
                    # for stat in time_stats:
                    #     time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
                    # print(time_str)

                    sk_res.append(ret['results'][1][0])
            else:
                break

            frame_count += 1
        pbar.close()
        sk_std = self._cal_std(sk_res)

        end_time = time.time() - start_time

        if sk_std > 0.1:
            sk_std = -1

        return sk_std, end_time


if __name__ == '__main__':
    opt = opts()
    print(opt)
    sk = sk_sim(opt)
    
    video_dir = r'/workspace/wenqiang/codes/motion_analysis-master/video/'
    video_list = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            video_path = os.path.join(root, file)
            if '.mp4' in video_path:
                video_list.append(video_path)

    skstd_dic = {}
    for v_path in video_list:
        print(v_path)
        sk.update_address(v_path)
        skstd, ti = sk.run_video()
        print(skstd, ti)
        skstd_dic[v_path] = [skstd, ti]

    with open('./res.json', 'w') as f:
        json.dump(skstd_dic, f)

