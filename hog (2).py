# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 01:56:49 2018

@author: Shruti
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
import imageio.core.util
import glob

def findHog(data, img, class_label):
    img = cv2.resize(img, (24, 24))
    #plt.imshow(img)
    winSize = (16,16)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (4,4)
    nbins = 9
    
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    
    winStride = (4,4)
    hist = hog.compute(img, winStride)
    i=0

#There are no headers in the file. First 2916 columns are feature values while 
#the last column contains the class label

    
    while i < hist.size:
        for a in hist[i]:
            data.write(str(a))
            data.write(',')
            i=i+1
    if class_label == 0:
        data.write('0')  #label for positive sample
    elif class_label == 1:
        data.write('1')
            

with open('train_data.csv','w') as train_data:
    id=0
    for file in glob.glob('train_pos\*.jpg'):
        img = cv2.imread(file)
        findHog(train_data, img, 0)
        train_data.write('\n')
        id = id+1
    print("Total pos images " + str(id))
    id=0
    for file in glob.glob('train_neg\*.jpg'):
        img = cv2.imread(file)
        findHog(train_data, img, 1)
        train_data.write('\n')
        id = id+1
    print("Total neg images " + str(id))
    
with open('test_data.csv','w') as test_data:
    id=0
    for file in glob.glob('test_pos\*.jpg'):
        img = cv2.imread(file)
        findHog(test_data, img, 0)
        test_data.write('\n')
        id = id+1
    print("Total pos images " + str(id))
    id=0
    for file in glob.glob('test_neg\*.jpg'):
        img = cv2.imread(file)
        findHog(test_data, img, 1)
        test_data.write('\n')
        id = id+1
    print("Total neg images " + str(id))



    

    
    



    
