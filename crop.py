import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
import os
import imageio.core.util
import glob

def root_warning_handler(*args, **kwargs):
    pass
out_dir = 'cropped'
id=1;
dirs = ['train' , 'test']
for dir in dirs:
    if not os.path.exists(dir+ '_cropped'):
        os.mkdir(dir+ '_cropped')
    for file in glob.glob(dir+"\*.jpg"):
        img = cv2.imread(file)
        disp_save = cv2.imread(file)
    
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        img = cv2.resize(img, (250, 250)) 
        disp_save = cv2.resize(img, (250, 250))
    #plt.imshow(img)

        img = cv2.bilateralFilter(img,10, 80, 80)
        display=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(display)

        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh, dst = cv2.threshold(gray,100,220,0)
        im2, contours, hierarchy = cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        disp=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #disp_save = disp.copy()

        for p in contours:
            x, y, w, h = cv2.boundingRect(p)
            if w*h < 125:
                continue
            roi_save = disp_save[y:y+h, x:x+w]
            cv2.imwrite(dir+ '_cropped\img'+str(id) + '.jpg', roi_save)   
            print(str(id))
            roi = cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
       #plt.figure(id)
       #plt.clf()
       #plt.axis('off')
            plt.imshow(disp)
            id=id+1


