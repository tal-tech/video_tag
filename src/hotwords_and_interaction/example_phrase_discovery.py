#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-12-05
'''

import re
import os
import json
from collections import defaultdict


base_dir = os.path.dirname(os.path.abspath(__file__))
data_file_dir = os.path.join(base_dir, "data")
example_phrase_file_path = os.path.join(data_file_dir, "example_phrases.txt")

example_phrase_table = list()
with open(example_phrase_file_path, mode="r", encoding="utf8") as fr:
    for word in fr:
        example_phrase_table.append(word.strip())

punctuates_pattern = "[\，\。\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}'\:\;\,\[\]\.\<\>\/\?\~\！\%]"
chinese_pattern = "[\u4e00-\u9fff]"

def discover_example_phrase(fields):
    example_phrase_threshold = 1
    example_phrase_dict = defaultdict(int)
    detail = defaultdict(list)
    for field in fields:
        for phrase in example_phrase_table:
            phrase_cnt = field['text'].count(phrase)
            if phrase_cnt > 0:
                if phrase == "举例":
                    # 过滤掉不合理的pattern
                    filter_patterns = re.compile(
                        "(叫做举例|(通过|自己|可以|应该){chinese}{{0,10}}举例|举例{chinese}{{0,10}}多了)".format(
                            punctuates=punctuates_pattern,
                            chinese = chinese_pattern
                        )
                    )
                    filter_num = len(filter_patterns.findall(field['text']))
                    phrase_cnt -= filter_num


            if phrase_cnt > 0:
                new_field = field.copy()
                '''一句至多视为一个例子'''
                new_field["count"] = 1
                example_phrase_dict[phrase] += 1
                detail[phrase].append(new_field)
    example_phrase_record = dict()
    for phrase in example_phrase_dict:
        phrase_cnt = example_phrase_dict[phrase]
        if phrase_cnt >= example_phrase_threshold:
            example_phrase_record[phrase] = {"count": phrase_cnt,
                                         "detail": detail[phrase]}
    return example_phrase_record


def discover_example_phrase_api(asr_result):
    asr_result = json.loads(asr_result)["data"]["result"]
    return_example_phrases_dict = dict()
    query_example_phrases_dict = discover_example_phrase(asr_result)
    for phrase in query_example_phrases_dict:
        if query_example_phrases_dict[phrase]["count"] > 0:
            details = query_example_phrases_dict[phrase]["detail"]
            return_example_phrases_dict[phrase] = list()
            for detail in details:
                tmp_detail = {"begin_time": detail["begin_time"],
                              "end_time": detail["end_time"],
                              "count": detail["count"]}
                return_example_phrases_dict[phrase].append(tmp_detail)

    return json.dumps(return_example_phrases_dict)

def example():
    asr_result = json.dumps(
        {"data": {"result": [{'begin_time': 450670, 'end_time': 452460, 'sentence_id': 96, 'status_code': 0,
                              'text': '在这有一个b'},
                             {'begin_time': 452800, 'end_time': 454720, 'sentence_id': 97, 'status_code': 0,
                              'text': '假如说，咱们的。'},
                             {'begin_time': 455040, 'end_time': 457700, 'sentence_id': 98, 'status_code': 0,
                              'text': '在这儿的话，这一段得意。'},
                             {'begin_time': 458170, 'end_time': 467100, 'sentence_id': 99, 'status_code': 0,
                              'text': '萨德姆那里的ac加bc也等于的是个6，但是并不能说明你的c是个中点，所以，哎'},
                             ]}})

    res = discover_example_phrase_api(asr_result)
    print(res)


if __name__ == "__main__":

    # statistics_api_response_time()
    example()
