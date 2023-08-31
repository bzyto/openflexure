import cv2
import os
dir='SCAN_N_unlabeled'
for file in os.listdir(dir):
    print(dir+file)
    if file.endswith('.jpeg'):
        img = cv2.imread(dir+'\\'+file)
        cv2.imwrite(dir+'\\'+file[:-5] + '.jpg', img)