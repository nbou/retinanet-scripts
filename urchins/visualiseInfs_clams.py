import os
import numpy as np
import cv2
from keras_retinanet.utils.visualization import draw_box
import matplotlib.pyplot as plt
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


# read box positions from output of retinanet_inf_batch.py
def readInf(inf_path):
    f = open(inf_path, "r")
    lines = f.readlines()
    idx = 0

    out = []
    for line in lines[1:]:
        line = line.strip().split(',')
        imnme = line[0]
        x1,y1,x2,y2 = np.array(line[1:5], dtype=np.float32)
        # midpoint = np.array([(x2+x1)/2, (y2+y1)/2])
        score = np.float(line[5])
        out.append([imnme,np.array([x1,y1,x2,y2], dtype=np.float32), score, idx])
        idx+=1
    return out


inf_path = '/home/nader/scratch/St_helens_is_2009_night_deep/bottom_box_mos_1kpix/inf_boxes.txt'#'/home/nader/scripts/retinanet_scripts/urchins/inf_boxes.txt'
inflines = readInf(inf_path)

scoring = 'n'
saveims = 'n'
if scoring == 'y':
    outpth = '/home/nader/scratch/huon_13_accuracy.txt'
    outfle = open(outpth, "w+")



for i in inflines:#[25:-1:50]:
    im = cv2.imread(i[0])
    # im = np.reshape(im,(1,1024,1360,3))
    # copy to draw on
    draw = im.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)


    im = preprocess_image(im)
    im, scale = resize_image(im)
    b = i[1].astype(int)
    draw_box(draw, b, [31, 0, 255])
    print(i[-2])
    # caption = "{} {:.3f}".format(labels_to_names[label], score)
    # draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


    if scoring == 'y':
        tf = input('Accurate detetection (t/f)?')
        out = str(i[-1]) + ',' + os.path.basename(i[0]) + ',' + tf + '\n'
        outfle.write(out)

    if saveims == 'y':
        imsv = input('save_image (y/n)?')
        if imsv == 'y':
            cv2.imwrite('detection_' + os.path.basename(i[0]), draw)

