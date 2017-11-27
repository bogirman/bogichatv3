# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Tue Nov 14 15:17:17 2017)---
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')

## ---(Tue Nov 14 17:39:19 2017)---
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
runfile('D:/Anaconda3/lib/site-packages/matplotlib/pyplot.py', wdir='D:/Anaconda3/lib/site-packages/matplotlib')
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')

## ---(Tue Nov 14 17:54:41 2017)---
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
plt.pause(0.1)
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)

ax.plot(0.1,0.5,'r--',lw=5)
plt.ion()
plt.show()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)

ax.plot(0.1,0.5,'r--',lw=50)
plt.ion()
plt.show()
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data, y_data)

ax.plot(0.1,0.5,'red','-',lw=50)
plt.ion()
plt.show()
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data, y_data)

ax.plot(0.1,0.5,'r-',lw=5)
plt.ion()
plt.show()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data, y_data)

ax.plot(x_data, y_data,'r-',lw=5)
plt.ion()
plt.show()
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)

#ax.plot(x_data, y_data,'r-',lw=5)
plt.ion()
plt.show()
 print(x_data)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)

#ax.plot(x_data, y_data,'r-',lw=5)
plt.ion()
plt.show()

# 之後就可以用for迴圈訓練了
for i in range(30000):
     
     # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
     # x_data:[300,1]   y_data:[300,1]
    #sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    print(x_data)

runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定義layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   
    
    #自由選擇激活函數
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
y_data = np.square(x_data) - 0.5 + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 為了可以可視化我們訓練的結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/CNNtest.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/tenboard.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/temp.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTextCutTest.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTextCutTest.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k 20"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()


if len(args) < 1:
    print(USAGE)
    sys.exit(1)


file_name = args[0]
print(file_name)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k 20"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()


if len(args) < 1:
    print(USAGE)
    sys.exit(1)

print(args[0])
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k 20"

parser = OptionParser(USAGE)
print(parser)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k 20"

parser = OptionParser(USAGE)
print(parser)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
print(args)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k 20"

parser = OptionParser(USAGE)
print(parser)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
print(args[0])
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k 20"

parser = OptionParser(USAGE)
print(parser)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k [20]"

parser = OptionParser(USAGE)
print(parser)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k 20"

parser = OptionParser(USAGE)
print(parser)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 20"

parser = OptionParser(USAGE)
print(parser)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k [top k]"

parser = OptionParser(USAGE)
print(parser)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k [top k]"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags.py 123123 -k [topk]"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()
print(opt)
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
%tb
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
extract_tags.py textsentenct.txt -k 20
textsentenct.txt -k 20
textsentenct.txt -k
textsentenct.txt
extract_tags.py textsentenct.txt
"""
Created on Thu Nov 16 10:06:00 2017

@author: asus
"""

import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser
import argparse

USAGE = "usage:    python extract_tags.py [file name] -k [topk]"

#parser = OptionParser(usage='python extract_tags.py [textsentenct.txt] -k [topk]=20')
#parser = OptionParser(USAGE)
parser = argparse.ArgumentParser()
parser.parse_args()
-h
python3 jiebaTags.py --help
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')

## ---(Thu Nov 16 11:57:41 2017)---
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('D:/Anaconda3/lib/site-packages/jieba/analyse/tfidf.py', wdir='D:/Anaconda3/lib/site-packages/jieba/analyse')
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('D:/Anaconda3/lib/site-packages/jieba/analyse/tfidf.py', wdir='D:/Anaconda3/lib/site-packages/jieba/analyse')
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/TagsWeight.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
python3
--help
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
echo
usage: prog.py [-h] echo
prog.py [-h] echo
[-h] echo
argsTest.py [-h] echo
echo
echo -h
echo --help
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
echo
-h
echo -h
argsTst.py -h
echo.py -h
echo
python3
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
square
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
echo
square
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
square
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
echo
-h
-help
_Helper
help
help()
echo -help
echo help
echo --h
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
test
'echo -h'
echo
echo -h --help -help
echo [-h]
echo
echo [-h]
echo [-help]
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
echo
-h
echo -h
square
runfile('C:/Users/asus/.spyder-py3/argsTest.py', wdir='C:/Users/asus/.spyder-py3')
tb
runfile('C:/Users/asus/.spyder-py3/TagsWeight.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/TagsWeight.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/snownlpTest.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/snownlpTest.py', wdir='C:/Users/asus/.spyder-py3')

## ---(Fri Nov 17 08:42:15 2017)---
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTextCutTest.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('D:/Anaconda3/lib/site-packages/gensim/corpora/wikicorpus.py', wdir='D:/Anaconda3/lib/site-packages/gensim/corpora')
runfile('D:/Anaconda3/lib/site-packages/gensim/corpora/wikicorpus.py', wdir='D:/Anaconda3/lib/site-packages/gensim/corpora')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/wiki_to_txt.py', wdir='C:/Users/asus/.spyder-py3')
python wiki_to_txt.py zhwiki-20171103-pages-articles.xml.bz2

## ---(Mon Nov 20 08:45:42 2017)---
runfile('C:/Users/asus/.spyder-py3/jiebaTextCutTest.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/jiebaCutForBig.py', wdir='C:/Users/asus/.spyder-py3')

## ---(Fri Nov 24 14:20:17 2017)---
runfile('C:/Users/asus/.spyder-py3/jiebaTags.py', wdir='C:/Users/asus/.spyder-py3')

## ---(Mon Nov 27 08:50:50 2017)---
runfile('C:/Users/asus/.spyder-py3/demo.py', wdir='C:/Users/asus/.spyder-py3')
runfile('C:/Users/asus/.spyder-py3/app.py', wdir='C:/Users/asus/.spyder-py3')