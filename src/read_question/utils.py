#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from scipy.stats import mode
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks

def jaccard_similarity(sent_1, sent_2):
    char_list_1 = [i for i in sent_1]
    char_list_2 = [i for i in sent_2]
    intersection = len(set(char_list_1).intersection(char_list_2))
    union = len(set(char_list_1)) + len(set(char_list_2)) - intersection
    jaccard = intersection / union
    return jaccard

def calculate_similarity(content_list, teacher_analysis):
    '''
    content_list:
        type: list
        sample: ['假设小兔子有两条腿','小乌龟有4条腿','那他们总共有几条腿']
    teacher_analysis:
        type: list
        sample: ['那我们来看看这道变态的问题'，‘他说小兔子’，‘啊，小兔子有两条腿']
    '''
    sim_distribution = []
    for analysis in teacher_analysis:
        sim_list = []
        for content in content_list:
            sim_list.append(jaccard_similarity(content, analysis))
        sim_distribution.append(max(sim_list))
    return sim_distribution

def split_doc(paragraph):
    paragraph = paragraph.replace('|||','')
    return re.findall(u'[^!?。，,\.\!\?]+[!?。，,\.\!\?]?', paragraph, flags=re.U)

def split_doc_2(paragraph):
    paragraph = paragraph.replace('|||','').replace('\\n','').replace('\\times','乘').replace('\\underline','').replace('\\quad','_').replace('\\left','').replace('\\right','').replace('\\frac','').replace('{','').replace('}','').replace('\\sim','_')
    return re.findall(u'[^!?。，;,\.\!\?]+[!?。，;,\.\!\?]?', paragraph, flags=re.U)

def moving_average(similarity_array, content_list, teacher_analyais):
    window_size = len(content_list)
    move = []
    for idx in range(0, len(similarity_array)-window_size, window_size):
        move.append(np.mean(similarity_array[idx:idx+window_size]))
    argmax = np.argmax(move)
    #argmax_ = np.argmax(similarity_array)
    ratio_position = argmax / (len(similarity_array)-window_size)
    #, teacher_analyais[argmax:argmax+window_size], move[argmax], similarity_array[argmax_], teacher_analyais[argmax_:argmax_+2]
    return ratio_position, argmax, len(similarity_array)-window_size, teacher_analyais[argmax:argmax+window_size]

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

# extract feature form numpy array
# 'mean','median','mode','max','where','min','top','variance','peak_num','peak_distance','z_scores_signal'
def extract_feature(asr_result, question):
    question_split = split_doc_2(question)
    asr_split = split_doc_2(''.join([seg['text'] for seg in asr_result['data']['result']]))
    #asr_split = [seg['text'] for seg in asr_result['data']['result']]
    ndarray = calculate_similarity(question_split,asr_split)
    if len(ndarray) <= len(question_split):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #basic statistic
    mean_x, median_x, mode_x, max_x, min_x, var_x = np.mean(ndarray), np.median(ndarray), mode(ndarray)[0][0], max(ndarray), min(ndarray), np.var(ndarray)
    where_x = moving_average(ndarray, question_split, asr_split)[0]
    where_top_x = np.argmax(ndarray)/len(ndarray)

    # calculate peak num
    low_bound = np.mean(np.sort(ndarray)[::-1][:10])
    peaks,_ = find_peaks(ndarray, height=low_bound, distance=20)
    peak_num_x = len(peaks)
    peak_distance_x = sum(np.diff(peaks))

    # z_score method features
    z_result = thresholding_algo(ndarray, lag=5, threshold=25, influence=1)
    z_score_x = sum(z_result['signals'])
    one_feature = np.array([mean_x, median_x, mode_x, max_x, where_x, min_x, where_top_x, var_x, peak_num_x, peak_distance_x, z_score_x]).reshape(1,-1)
    return one_feature

