#coding=utf-8
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path))
import json
import re
import logging

__all__ = ["quest_detector"]
logging.basicConfig(filename=os.path.join(base_path, 'logger.log'), level=logging.INFO)
logger = logging.getLogger()

class RuleBasedDetector:
    """
    This class defines a Scikit-Learn classifier to detect interrogative sentences.
    """

    def __init__(self, key_word_list, stop_list, regex_list, general_list, special_list):
        self.key_word_list = key_word_list
        self.stop_list = stop_list
        self.regex_list = regex_list
        self.general_list = general_list
        self.special_list = special_list

        if set(self.key_word_list) != set(self.special_list + self.general_list):
            diff = set(self.key_word_list).symmetric_difference(set(self.special_list + self.general_list))
            raise ValueError("一般疑问句与特殊疑问句并非全集，差集为: ", diff)

    def is_interrogative(self, x):
        '''
        :param x: 输入一句话
        :return: 0：非问句。1：一般疑问句。2：特殊疑问句。-1：出错
        '''

        logger.info("输入的句子为：{}".format(x))
        if len(x) < 1:
            logger.info("句子为空.")
            return 0, None

        if set(self.key_word_list) != set(self.special_list + self.general_list):
            diff = set(self.key_word_list).symmetric_difference(set(self.special_list + self.general_list))
            logger.error("一般疑问句与特殊疑问句并非全集, 差集为：{}".format(str(diff)))
            return -1, None

        for i in self.key_word_list:
            if i in x:
                logger.info("命中关键词：{}".format(i))
                for j in self.stop_list:
                    if i not in j:
                        continue
                    if j in x:
                        logger.info("命中停用关键词：{}".format(j))
                        return 0, None
                if i in self.general_list:
                    return 1, i
                elif i in self.special_list:
                    return 2, i
                else:
                    print("一般疑问句与特殊疑问句并非全集。")
                    return -1, None

        for i in self.regex_list:
            result = re.search(i, x, flags=0)
            if result is not None:
                logger.info("命中正则模版：{}".format(i))
                for j in self.stop_list:
                    if j in x:
                        logger.info("命中停用关键词：{}".format(j))
                        return 0, None
                return 2, i
        return 0, None

    def predict(self, asr_result_str):
        logger.debug("开始预测，收到asr结果：{}".format(asr_result_str))
        asr_results = json.loads(asr_result_str)["data"]["result"]
        results = {}
        for sent in asr_results:
            begin_time = sent["begin_time"]
            end_time = sent["end_time"]
            text = sent["text"].strip()
            if_question, pattern = self.is_interrogative(text)
            if if_question > 0:
                one = {
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "count": 1
                }
                if pattern not in results:
                    results[pattern] = []
                    results[pattern].append(one)
                else:
                    results[pattern].append(one)
        logger.debug("预测结束，返回结果为：{}".format(json.dumps(results)))
        return json.dumps(results)


def preprocess(x):
    return x.strip()

def get_rules():
    with open(os.path.join(base_path, 'rules/rules.txt'), encoding='utf-8') as f:
        rules_list = f.readlines()
        rules_list = list(map(preprocess, rules_list))
    return rules_list


def get_stop():
    with open(os.path.join(base_path, 'rules/stop_rules.txt'), encoding='utf-8') as f:
        rules_list = f.readlines()
        rules_list = list(map(preprocess, rules_list))
    return rules_list


def get_regex():
    with open(os.path.join(base_path, 'rules/regex_rules.txt'), encoding='utf-8') as f:
        rules_list = f.readlines()
        rules_list = list(map(preprocess, rules_list))
    return rules_list


def get_general():
    with open(os.path.join(base_path, 'rules/general_question.txt'), encoding='utf-8') as f:
        rules_list = f.readlines()
        rules_list = list(map(preprocess, rules_list))
    return rules_list


def get_special():
    with open(os.path.join(base_path, 'rules/special_question.txt'), encoding='utf-8') as f:
        rules_list = f.readlines()
        rules_list = list(map(preprocess, rules_list))
    return rules_list


def quest_detector_init():
    """
    实例化一个问句检测模型，并加载目前的规则。
    :return:
    """
    rule_list = get_rules()
    stop_list = get_stop()
    regex_list = get_regex()
    general_list = get_general()
    special_list = get_special()

    model = RuleBasedDetector(key_word_list=rule_list, stop_list=stop_list, regex_list=regex_list,
                              general_list=general_list, special_list=special_list)
    return model


quest_detector = quest_detector_init()


if __name__ == "__main__":

    def get_test_dict_data():
        return {
        "data": {
            'result': [{'begin_time': 1600, 'sentence_id': 1, 'status_code': 0, 'text': '细思今天，我给大家讲到时再加减到一些套娃，将全部的一半送给了vi',
                     'end_time': 11440},
                    {'begin_time': 12060, 'sentence_id': 2, 'status_code': 0, 'text': '下周可能分给了艾迪，这时候加入海上4个专家，最开始呢多少个套娃',
                     'end_time': 20510},
                    {'begin_time': 21540, 'sentence_id': 3, 'status_code': 0, 'text': '我们先把他看成一个涨用品。', 'end_time': 25140},
                    {'begin_time': 26020, 'sentence_id': 4, 'status_code': 0, 'text': '他说，他把全部的答复给了vr，所以这卖旧书威尔',
                     'end_time': 32020},
                    {'begin_time': 32830, 'sentence_id': 5, 'status_code': 0, 'text': '还有把剩下的一半重点爱情就是爱迪，这时候，他还是四个，他们两家让郭',
                     'end_time': 43040
                     }]
                }
        }

    json_str = json.dumps(get_test_dict_data())
    res = quest_detector.predict(json_str)
    print(res)


