# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:26:27 2017

@author: asus
"""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import sys

log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)
sys.path.append('../')
from gensim.corpora import WikiCorpus
print('go')
file_name = 'zhwiki-20171103-pages-articles.xml.bz2'
wiki_corpus = WikiCorpus(file_name, dictionary={})
texts_num = 0
print(' for')
with open("wiki_texts.txt",'w',encoding='utf-8') as output:
        for text in wiki_corpus.get_texts():
            default_logger.debug('text in')
            output.write(' '.join(text) + '\n')
            texts_num += 1
            default_logger.debug('已處理 %d 篇文章' % texts_num)
            #if texts_num % 10 == 0:
                
