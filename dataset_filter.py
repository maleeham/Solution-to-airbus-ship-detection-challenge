import pandas as pd
df=pd.read_csv('train_ship_segmentations_v2.csv')


xy=df[df.EncodedPixels.notnull()]

from PIL import Image
import os

path1 = 'train_dataset'    
path2 = 'positive_data_samples'
if not os.path.exists(path2):
    os.mkdir(path2)
listing = os.listdir(path1)
already = os.listdir(path2)    
for file in listing:
    im = Image.open(path1 + '/' + file)
    if(any(xy.ImageId == file)):
        if not os.path.exists(path2 + '/' + file):
            print(file)
            im.save(path2 + '/' + file)
