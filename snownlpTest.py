# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:24:17 2017

@author: asus
"""
import sys
sys.path.append('../')
from snownlp import SnowNLP



file_name = "textsentenct.txt"
content = open(file_name, 'r',encoding='utf-8').read()

s= SnowNLP(content)
print(s.tags)

