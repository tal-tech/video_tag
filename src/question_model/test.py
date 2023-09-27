#coding=utf-8
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path, "question_model"))
import json
from rule_based import quest_detector


path = "/share/haifeng/small_project/数据分析/data/音频文件1230/asr_result"
json_files = os.listdir(path)
json_files = json_files[:100]
for json_file in json_files:
    json_path = os.path.join(path, json_file)
    with open(json_path, 'r') as f:
        json_str = f.readline()

    res = quest_detector.predict(json_str)
    res_dict = json.loads(res)

    print(res_dict)

