# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:39:09 2018

@author: malee
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from xgboost import XGBClassifier 
import xgboost as xgb
from sklearn.externals import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display


model = joblib.load('XGBoost.pkl')
winSize = (16,16)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)
nbins = 9
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
winStride = (4,4)
image_num=0 
for file in glob.glob("demo\*.jpg"):
    image_num= image_num+1
    img = cv2.imread(file)
    disp_save = cv2.imread(file)
    img = cv2.resize(img, (250, 250)) 
    disp_save = cv2.resize(img, (250, 250))
    img = cv2.bilateralFilter(img,10, 80, 80)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, dst = cv2.threshold(gray,100,220,0)
    im2, contours, hierarchy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    disp=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    list_a = []
    height = []
    width = []
    x_coord = []
    y_coord = []
    for p in contours:
        x, y, w, h = cv2.boundingRect(p)
        if w*h < 125:
            continue
        roi_to_predict = disp_save[y:y+h, x:x+w]
        height.append(h)
        width.append(w)
        x_coord.append(x)
        y_coord.append(y)
        #print(roi_to_predict)
        roi_to_predict = cv2.resize(roi_to_predict, (24, 24))
        hist = hog.compute(roi_to_predict, winStride)
        features = []
        i=0
        vals = ""
        while i < hist.size:
            for a in hist[i]:
                vals = vals+str(a)
                if (i <hist.size-1):
                    vals = vals + ','
            i=i+1
        list_a.append(vals) 
        #roi = cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)    
                       
    s = 0
    length = len(list_a)
    with open('predict.csv','w') as file:
        for a in list_a:
            if s < length-1:
                a = a+ "\n"
            file.write(str(a))
        s = s + 1			
    df_test = pd.read_csv('predict.csv', header=None)
    #print(df_test)
    #print(df_test.size)
    X_test = df_test.iloc[:, :2915].values
    #print(X_test.size)
    prediction = model.predict(X_test)
    #print(prediction)
    
    i=0
    display("The coordinates of detected rois of image " + str(image_num) + " are :")

    for x in x_coord:
        #print(r)
        if(prediction[i] == 0):
            cv2.rectangle(disp, (x, y_coord[i]), (x+width[i], y_coord[i]+height[i]), (0, 255, 0), 2)
            display("("+str(x)+","+str(y_coord[i])+")")
        else :
            cv2.rectangle(disp, (x, y_coord[i]), (x+width[i], y_coord[i]+height[i]), (255, 0, 0), 2)  
        i=i+1
    plt.figure(image_num)
    plt.clf()
    plt.axis('off')    
    plt.imshow(disp)
