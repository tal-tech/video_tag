#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle as pkl
import configparser
import logging
import numpy as np
import os
import sys
import json
from utils import *
from logging import handlers

base_path = os.path.dirname(os.path.realpath(__file__))
config_read = configparser.ConfigParser()
config_read.read(os.path.join(base_path, 'config.ini'))
log_path = config_read['log']['log_path']

logging.basicConfig(
    filename=os.path.join(base_path,log_path),
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
)

class readQuestionType:
    def __init__(self, model_path = os.path.join(os.path.dirname(__file__),'model','read_question.pkl')):
        with open(model_path, 'rb') as f:
            self.model = pkl.load(f)

    def forward(self, asr_result, question_text):
        pred_result = None
        msg = None
        try:
            feature_x = extract_feature(asr_result, question_text)
            if (feature_x==0).all():
                pred_result = '无念题'
            else:
                pred_result = self.model.predict(feature_x)[0]
        except Exception as e:
            logging.error('input: {}, {}, with bug {}'.format(asr_result,question_text, e))
            msg = 'get error in feature extraction' + str(e)
        one_res = {'read_question_type':pred_result, 'msg':msg}
        return one_res

_readType_worker = readQuestionType()

def get_read_type(asr_result: str, question_text:str):
    return _readType_worker.forward(json.loads(asr_result), question_text)

if __name__ == "__main__":
    #readQuestionType_predicter = readQuestionType()
    # 初始化对象，复用对象
    #asr = '你也填不出来，对不对|||往下探索吧，嗯开始难度呢，稍微深了一点，点击，但是他也没有前面那个idvi的周期皮呢啊，所以跟着我来理解好吧。'
    asr = {
    "data": {
        "task_id": "7cb4d7ac-5460-11e9-8741-484d7e98d566",
        "result": [
            {
                "begin_time": 7570,
                "end_time": 8845,
                "text": "好了好了。",
                "sentence_id": 1,
                "keywords": [{
                        "word": "KW204-00004",
                        "begin": 20, 
                        "end": 37,
                        "score": 8.797852
                }]
            },
            {
                "begin_time": 8830,
                "end_time": 9845,
                "text": "hello",
                "sentence_id": 2
            },
            {
                "begin_time": 15000,
                "end_time": 16145,
                "text": "谢谢",
                "sentence_id": 3
            },
            {
                "begin_time": 52280,
                "end_time": 53405,
                "text": "嗯",
                "sentence_id": 4
            },
            {
                "begin_time": 59180,
                "end_time": 60275,
                "text": "嗯",
                "sentence_id": 5
            },
            {
                "begin_time": 76290,
                "end_time": 77755,
                "text": "大哥",
                "sentence_id": 6
            },
            {
                "begin_time": 80710,
                "end_time": 82805,
                "text": "有时候",
                "sentence_id": 7
            }
        ]
    },
    "code": 20000,
    "msg": "success",
    "models":"kws",
    "requestId": "as"
    }
    question = '哈哈哈哈，我是小神仙。你是什么龟。我的妈呀呀，你是什么鬼'
    #nianti = readQuestionType_predicter.forward(asr, question)
    nianti = get_read_type(asr, question)
    print(nianti)
    