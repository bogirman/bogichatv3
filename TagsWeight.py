# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:59:35 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse





file_name = "textsentenct.txt"

topK=10
withWeight=True

content = open(file_name, 'r',encoding='utf-8').read()

tags = jieba.analyse.extract_tags(content, topK=topK, withWeight=withWeight)

if withWeight is True:
    for tag in tags:
        print("tag: %s\t\t weight: %f" % (tag[0],tag[1]))
else:
    print(",".join(tags))
