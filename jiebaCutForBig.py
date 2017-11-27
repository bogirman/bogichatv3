# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:25:32 2017

@author: asus
"""

import jieba
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
        #jieba custom setting.
    jieba.set_dictionary('jieba/dict.txt.big')
    
        #load stopwords set
    stopwordset = set()
    with open('jieba/stop_words.txt','r') as sw:
        for line in sw:
            stopwordset.add(line.strip('\n'))
    
    output = open('wiki_seg.txt','w',encoding='utf-8')
        
    texts_num = 0
        
    with open('wiki_texts.txt','r',encoding='utf-8') as content :
        for line in content:
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopwordset:
                    if word is str:
                            #word = word.decode('utf-8')  
                            #logging.info("斷字(str)： %s " % word)
                            output.write(word.encode('utf-8') + ' ')
                    else:
                            #logging.info("斷字(byte)： %s " % word)
                            output.write(word)
            texts_num += 1
            if texts_num % 1000 == 0:
                logging.info("已完成前 %d 行的斷詞" % texts_num)
    output.close()
    
if __name__ == '__main__':
	main()