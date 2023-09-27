#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-12-05
'''

import os
import re
import json
from collections import defaultdict


base_dir = os.path.dirname(os.path.abspath(__file__))
data_file_dir = os.path.join(base_dir, "data")
review_phrase_file_path = os.path.join(data_file_dir, "review_phrases.txt")

review_phrase_table = list()
with open(review_phrase_file_path, mode="r", encoding="utf8") as fr:
    for word in fr:
        review_phrase_table.append(word.strip())

punctuates_pattern = "[\，\。\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}'\:\;\,\[\]\.\<\>\/\?\~\！\%]"
chinese_pattern = "[\u4e00-\u9fff]"

def discover_review_phrase(fields):
    review_phrase_threshold = 1
    review_phrase_dict = defaultdict(int)
    detail = defaultdict(list)
    for field in fields:
        for phrase in review_phrase_table:
            phrase_cnt = field['text'].count(phrase)
            if phrase_cnt > 0:
                if phrase == "上":
                    # 匹配合理的pattern
                    match_patterns = re.compile(
                        "([上]{chinese}{{0,3}}节)".format(
                            punctuates=punctuates_pattern,
                            chinese=chinese_pattern
                        )
                    )
                    match_num = len(match_patterns.findall(field['text']))
                    phrase_cnt = match_num

                elif phrase == "前":
                    # 匹配合理的pattern
                    match_patterns = re.compile(
                        "([前]{chinese}{{0,3}}节)".format(
                            punctuates=punctuates_pattern,
                            chinese=chinese_pattern
                        )
                    )
                    match_num = len(match_patterns.findall(field['text']))
                    phrase_cnt = match_num
                    
            if phrase_cnt > 0:
                new_field = field.copy()
                # 一句话至多视为一次复习
                new_field["count"] = 1
                review_phrase_dict[phrase] += 1

                detail[phrase].append(new_field)
    review_phrase_record = dict()
    for phrase in review_phrase_dict:
        phrase_cnt = review_phrase_dict[phrase]
        if phrase_cnt >= review_phrase_threshold:
            review_phrase_record[phrase] = {"count": phrase_cnt,
                                         "detail": detail[phrase]}
    return review_phrase_record

def discover_review_phrase_api(asr_result):
    asr_result = json.loads(asr_result)["data"]["result"]
    return_review_phrases_dict = dict()
    query_review_phrases_dict = discover_review_phrase(asr_result)
    for phrase in query_review_phrases_dict:
        if query_review_phrases_dict[phrase]["count"] > 0:
            details = query_review_phrases_dict[phrase]["detail"]
            return_review_phrases_dict[phrase] = list()
            for detail in details:
                tmp_detail = {"begin_time": detail["begin_time"],
                              "end_time": detail["end_time"],
                              "count": detail["count"]}
                return_review_phrases_dict[phrase].append(tmp_detail)
    return json.dumps(return_review_phrases_dict)


def example():
    asr_result = json.dumps(
        {"data": {"result": [{'begin_time': 750, 'end_time': 18670, 'sentence_id': 1, 'status_code': 0, 'text': '好同学们那下面我们来看一个全新的立体力三，这道题，我们在前面复习的，你的数轴复习的绝对值还复习的，咱们的有理数的计算，那么咱们把它整个融合在一起，这一道题，老师待会要疯狂提问，你一定要积极的动脑哎迅速的反应出来，按你看好第一个他。'}, {'begin_time': 19110, 'end_time': 29340, 'sentence_id': 2, 'status_code': 0, 'text': 'ab互为相反，数那么老师第一个问题就出来了，如果ab互为相反，数的话，哪个小朋友能知道a加b应该得什么'}, {'begin_time': 30000, 'end_time': 40890, 'sentence_id': 3, 'status_code': 0, 'text': '非常好，ab互为相反，数就说明a加b等于零，那么第二个问题呢，是问了c和d互为倒数，说明什么？'}, {'begin_time': 41930, 'end_time': 48450, 'sentence_id': 4, 'status_code': 0, 'text': '调亮cd互为倒数，就说明c乘d得一那么第三个。'}, {'begin_time': 48630, 'end_time': 55030, 'sentence_id': 5, 'status_code': 0, 'text': 'x的绝对值等于2的话，那那一个小朋友能告诉老师x都可以得谁呢'}, {'begin_time': 57550, 'end_time': 65420, 'sentence_id': 6, 'status_code': 0, 'text': '政府二随便谁都行好，那你读到这儿之后，人家要你求这道题德基那不就直接带回来。'}, {'begin_time': 65840, 'end_time': 68100, 'sentence_id': 7, 'status_code': 0, 'text': '你看姐'}, {'begin_time': 68180, 'end_time': 69860, 'sentence_id': 8, 'status_code': 0, 'text': '远视'}, {'begin_time': 70790, 'end_time': 72720, 'sentence_id': 9, 'status_code': 0, 'text': '好等于'}, {'begin_time': 73770, 'end_time': 82670, 'sentence_id': 10, 'status_code': 0, 'text': 'x就两种不同取值，咱们先不写x我起一个x的平方减膘，看好了啊同学们听好a加b'}, {'begin_time': 83340, 'end_time': 88050, 'sentence_id': 11, 'status_code': 0, 'text': '非诚d得一所以就剪掉一ex。'}, {'begin_time': 88130, 'end_time': 91120, 'sentence_id': 12, 'status_code': 0, 'text': '好，下面你一块来回答老师的问题，给你个机会。'}, {'begin_time': 91240, 'end_time': 94920, 'sentence_id': 13, 'status_code': 0, 'text': 'a加be德里那0到20114方都什么样？'}, {'begin_time': 95700, 'end_time': 100880, 'sentence_id': 14, 'status_code': 0, 'text': '非常好，一定要大声回答老师的问题啊这时候，他就得了是一个零儿了。'}, {'begin_time': 101360, 'end_time': 104880, 'sentence_id': 15, 'status_code': 0, 'text': '非诚必得一呢，负的c乘d呢？'}, {'begin_time': 105540, 'end_time': 110000, 'sentence_id': 16, 'status_code': 0, 'text': '负一那负一的2012次方呢？'}, {'begin_time': 111650, 'end_time': 121730, 'sentence_id': 17, 'status_code': 0, 'text': '非常好，负一的2012次方得的是，一那么做到这儿，我就发现了，原是可以写成x方减x，在加一。'}, {'begin_time': 122530, 'end_time': 127690, 'sentence_id': 18, 'status_code': 0, 'text': '那么同时，咱们发现了虽然你的xx得的是二，你看2'}, {'begin_time': 127770, 'end_time': 142170, 'sentence_id': 19, 'status_code': 0, 'text': '的平方得4-2的平方还得4，所以x的平方啊，他永远都得的是四也就可以把原来的式子写成一个五减矮'}, {'begin_time': 143050, 'end_time': 144930, 'sentence_id': 20, 'status_code': 0, 'text': '那么你写到这儿的时候，x可以。'}, {'begin_time': 145030, 'end_time': 147430, 'sentence_id': 21, 'status_code': 0, 'text': '正二还可以得负二一定要写一部当'}, {'begin_time': 148100, 'end_time': 150880, 'sentence_id': 22, 'status_code': 0, 'text': 'x等于2的时候'}, {'begin_time': 151360, 'end_time': 154180, 'sentence_id': 23, 'status_code': 0, 'text': '那么原来的式子白。'}, {'begin_time': 154260, 'end_time': 160670, 'sentence_id': 24, 'status_code': 0, 'text': '同学给老师孔算一个靴子，如果xxcu2的时候，那就变成5减2的应该得几呢'}, {'begin_time': 162570, 'end_time': 167350, 'sentence_id': 25, 'status_code': 0, 'text': '知道，小编一定要大声回答老师的问题啊吴洁二等于的是一个三那。'}, {'begin_time': 167440, 'end_time': 172100, 'sentence_id': 26, 'status_code': 0, 'text': '另外一种情况是，如果x等于-2的时候。'}, {'begin_time': 172220, 'end_time': 174530, 'sentence_id': 27, 'status_code': 0, 'text': '这时候，你的原饰'}, {'begin_time': 174710, 'end_time': 178600, 'sentence_id': 28, 'status_code': 0, 'text': '买回来就变成了一个5减掉一个父母2'}, {'begin_time': 178720, 'end_time': 181040, 'sentence_id': 29, 'status_code': 0, 'text': '那夫妇应该得的是'}, {'begin_time': 181120, 'end_time': 184480, 'sentence_id': 30, 'status_code': 0, 'text': '正爱那吴姐负二结果应该得什么呢？'}, {'begin_time': 185100, 'end_time': 199900, 'sentence_id': 31, 'status_code': 0, 'text': '嗯，非常好，他应该得的是一个7，所以这道题的正确答案应该优两个，要么得三要么的亲就ok了，看明白的话，小朋友们好那么你的例三，整个学完了之后啊咱们整个的第一。'}, {'begin_time': 200070, 'end_time': 215960, 'sentence_id': 32, 'status_code': 0, 'text': '大模块乖有理数的计算，就全部按照大家顾客的这个难题了，好那么，老师给大家一点时间呢，我们来看一波幸苦的笔记，咱们把前面这部分知识也说是的，内容做一个总结来赶紧记一下大姐的笔记吗？'}]}})

    res = discover_review_phrase_api(asr_result)
    print(res)

if __name__ == "__main__":

    example()