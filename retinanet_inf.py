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


model = models.load_model('/home/nader/scratch/resnet50_csv_50_inf.h5', backbone_name='resnet50')
# anns = '/home/nader/scratch/anns_test.csv'
anns = '/home/nader/scratch/inf_boxes_huon_13_ans.csv'
cls = '/home/nader/scratch/classes.csv'

validation_generator = CSVGenerator(anns, cls)

# ims = np.array(())
# for i in range(validation_generator.size()):
#     im = validation_generator.load_image(i)
#     ims.append(im)
# print(np.shape(ims))

for i in range(validation_generator.size()):
    im = validation_generator.load_image(i)
    # im = np.reshape(im,(1,1024,1360,3))
    # copy to draw on
    draw = im.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    im = preprocess_image(im)
    im, scale = resize_image(im)


    labels_to_names = {0: 'lobster'}

    boxes,scores,labels = model.predict_on_batch(np.expand_dims(im, axis=0))



    # correct for image scale
    boxes /= scale
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        print(score)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    # imname = validation_generator.image_names[i][:-4]+'out.png'
    # cv2.imwrite(imname, draw)

# cv2.imwrite('out.png', draw)