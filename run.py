#!/usr/bin/env python
#coding=utf-8
#使用魔幻透镜处理一些图像
'''
gabor_threads.py
=========

Sample demonstrates:
- use of multiple Gabor filter convolutions to get Fractalius-like image effect (http://www.redfieldplugins.com/filterFractalius.htm)
- use of python threading to accelerate the computation

Usage
-----
gabor_threads.py [image filename]

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from multiprocessing.pool import ThreadPool

# 创建滤波器
def build_filters():
    filters = []
    ksize = 31
    # 此处创建16个滤波器，只有getGaborKernel的第三个参数theta不同。
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

# 单线程处理
def process(img, filters):
    # zeros_like：返回和输入大小相同，类型相同，用0填满的数组
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        # maximum：逐位比较取其大
        np.maximum(accum, fimg, accum)
    return accum

# 多线程处理，threadn = 8
#def process_threaded(img, filters, threadn = 8):
#    accum = np.zeros_like(img)
#    def f(kern):
#        return cv.filter2D(img, cv.CV_8UC3, kern)
#    pool = ThreadPool(processes=threadn)
#    for fimg in pool.imap_unordered(f, filters):
#        np.maximum(accum, fimg, accum)
#    return accum

if __name__ == '__main__':
    import sys
    #from common import Timer

    # 输出文件开头由''' '''包含的注释内容
    print(__doc__)

    try:
        img_fn = sys.argv[1]
    except:
        img_fn = 'example.jpg'

    img = cv.imread(img_fn)
    # 判断图片是否读取成功
    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)

    filters = build_filters()

    #with Timer('running single-threaded'):
    res1 = process(img, filters)
    #with Timer('running multi-threaded'):
    #    res2 = process_threaded(img, filters)

    #print('res1 == res2: ', (res1 == res2).all())
    cv.imshow('img', res1)
    #cv.imshow('result', res2)
    cv.waitKey(0)
    cv.destroyAllWindows()
