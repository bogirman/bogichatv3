# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:38:52 2017

@author: asus
"""
import sys
import jieba
import jieba.analyse
from gensim.models import word2vec
from opencc import OpenCC
openCC = OpenCC('s2tw')
sys.path.append('../')

file_name = 'wiki_texts.txt'
#seg_list = jieba.cut("請問，報稅需要準備哪些資料呢?")
#seg_list2 = jieba.cut_for_search("請問，報稅需要準備哪些資料呢?")
sentenct=open(file_name,'r',encoding='utf-8').read()
sentenct = openCC.convert(sentenct)

#print(sentenct)
seg_list=jieba.cut(sentenct)

#transent = word2vec.Text8Corpus(seg_list)
#model = word2vec.Word2Vec(seg_list,size=250)
#model.save('med250.model.bin')
output = open('wiki_seg.txt','w')
    
texts_num = 0
    
with open('wiki_zh_tw.txt','r') as content :
    for line in content:
        words = jieba.cut(line, cut_all=False)
        for word in words:
                if word not in stopwordset:
                    output.write(word +' ')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % texts_num)
output.close()
hash = {}
for item in seg_list:
    if item in hash:
        hash[item] +=1
    else:
        hash[item] =1
fd = open('wiki_seg.tw','w')
fd.write('word,count\n')
for k in hash:    
    fd.write("%s,%d\n"%(k.encode('utf-8'),hash[k]))
        


