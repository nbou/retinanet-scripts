from keras_retinanet import models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os


model = models.load_model('/home/nader/scripts/retinanet_scripts/urchins/aaresnet50_csv_29_inf.h5', backbone_name='resnet50')
# anns = '/home/nader/scratch/anns_test.csv'
# cls = '/home/nader/scratch/classes.csv'

# imnames = open('/home/nader/scripts/retinanet_scripts/urchins/val_images.txt', 'r')
imnames = open('/home/nader/scratch/St_helens_is_2009_night_deep/bottom_box_mos_1kpix/imnames.txt', 'r')
imnames = imnames.readlines()
len(imnames)
# validation_generator = CSVGenerator(anns, cls)

# ims = np.array(())
# for i in range(validation_generator.size()):
#     im = validation_generator.load_image(i)
#     ims.append(im)
# print(np.shape(ims))
a=0
# b=0

outfilepth = '/home/nader/scratch/St_helens_is_2009_night_deep/bottom_box_mos_1kpix/inf_boxes.txt' #'/home/nader/scripts/retinanet_scripts/urchins/inf_boxes.txt'
outfile = open(outfilepth, "w+")
outfile.write('image_path,x1,y1,x2,y2,score\n')

for i in range(len(imnames)):
    print("reading image {0}/{1}".format(i+1,len(imnames)) )
    imname = imnames[i].strip()
    im = cv2.imread(imname)
    # im = np.reshape(im,(1,1024,1360,3))
    # copy to draw on
    # draw = im.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # print(b)
    # b+=1
    # preprocess image for network
    im = preprocess_image(im)
    im, scale = resize_image(im)

    # print(i)
    labels_to_names = {0: 'urchin'}

    boxes,scores,labels = model.predict_on_batch(np.expand_dims(im, axis=0))


    if np.sum(labels)>-300:

        # correct for image scale
        boxes /= scale
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score > 0.5:
            # if score > 0:
                line = imname + ',' + str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' + str(box[3]) + ',' + str(score) + '\n'
                outfile.write(line)
                a+=1
                print("found {0} urchins".format(a))
                print(imname)
                # print(a, i)
            # color = label_color(label)
            #
            # b = box.astype(int)
            # draw_box(draw, b, color=color)
            #
            # caption = "{} {:.3f}".format(labels_to_names[label], score)
            # draw_caption(draw, b, caption)
    else:
        continue

    # plt.figure(figsize=(15, 15))
    # plt.axis('off')
    # plt.imshow(draw)
    # plt.show()
    # imname = validation_generator.image_names[i][:-4]+'out.png'
    # cv2.imwrite(imname, draw)

# cv2.imwrite('out.png', draw)
outfile.close()
