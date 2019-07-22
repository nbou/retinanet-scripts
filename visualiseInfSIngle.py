import os
import numpy as np
import cv2
from keras_retinanet.utils.visualization import draw_box
import matplotlib.pyplot as plt
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from sys import argv

def checkArgs(args):

    if len(args)!=6:
        print('usage: python3 visualiseInfSIngle.py b1 b2 b3 b4')
        exit()
    return 0
    
checkArgs(argv)

filename = str(argv[1])#'/media/nader/ML_fish_data/lobsters/r20100604_061515_huon_13_deep_in2/i20100604_061515_cv/PR_20100604_064254_985_LC16.png'
print(filename)
box = np.array(argv[2:6],dtype=np.float) #np.array([426.39728,	316.00714,	539.0801,	470.19647])
#print(argv[2:6])
im = cv2.imread(filename)
draw=im.copy()
draw_box(draw, box, [31, 0, 255])
plt.imshow(draw)
plt.show()

imsv = input('save_image (y/n)?')
if imsv == 'y':
    cv2.imwrite('FALSE_POS_' + os.path.basename(filename), draw)

