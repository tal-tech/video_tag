#!/usr/bin/env python
# encoding: utf-8

'''
From: TAL-AI Data Mining Group
Contact: qiyubi@100tal.com
Date: 2019-12-02
'''

import os
import re

base_dir = os.path.dirname(os.path.abspath(__file__))

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return set(stopwords)

def filter_empty(text):
    empty_pattern = re.compile("[\t\n]")
    text = re.sub(empty_pattern, '', text)
    return text

def filter_special(text):
    special_pattern = re.compile("[0-9\.\`\，\。\！\~\!\$\(\)\=\|\{\}\:\;\,\[\]\<\>\/\?\~\@\#\&\*\%]")
    text = re.sub(special_pattern, '', text)
    return text
