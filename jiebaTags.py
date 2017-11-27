# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse

#topK=10
#file_name = "textsentenct.txt"

#content = open(file_name, 'r',encoding='utf-8').read()
tags = jieba.cut('使用牌照稅的納稅義務人為誰？')
#tags = jieba.analyse.extract_tags(content, topK=topK)

print(",".join(tags))
#print(",".join(jieba.analyse.extract_tags("顧立雄")))