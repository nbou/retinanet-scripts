# script to convert detections (e.g. of urchins) from mosaic slices, to lat/longs. Use this version if the inferences
# are points (rather than boxes) e.g. if using manual labels, not auto

from osgeo import gdal,osr
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def readInf(inf_path):
    f = open(inf_path, "r")
    lines = f.readlines()
    idx = 0

    out = []
    for line in lines[1:]:

        line = line.strip().split(',')
        # if float(line[-1]) > 0.5:
        imnme = line[0]
        x1,y1,x2,y2 = np.array(line[1:5], dtype=np.float32)
        midpoint = np.array([(x2+x1)/2, (y2+y1)/2])
        # score = np.float(line[5])
        out.append([imnme,np.array(midpoint, dtype=np.float32), idx])
        idx+=1
    return out


# inf_data = ('/home/nader/scripts/retinanet_scripts/urchins/inf_boxes.txt')
inf_data = '/home/nader/scratch/St_helens_is_2009_night_deep/bottom_box_mos_1kpix/manual_labels_box.txt'
outfile = inf_data[:-4] + '_geo.txt'


if os.path.exists(outfile):
    appwri = "a"
else:
    appwri = "w"

f = open(outfile,appwri)


inf_lines = readInf(inf_data)


driver = gdal.GetDriverByName('GTiff')
imname_last = "blah"
for line in inf_lines:
    # print(line)

    imname = line[0]
    if imname != imname_last:
        dataset = gdal.Open(imname)
        GT = dataset.GetGeoTransform()
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(dataset.GetProjection())
        tgt_srs = src_srs.CloneGeogCS()
        transform = osr.CoordinateTransformation(src_srs, tgt_srs)
        # print(GT)



    coord = np.round(line[1]).astype(np.int)
    # im = cv2.imread(imname)
    # plt.imshow(im)
    # plt.scatter(coord[0],coord[1])
    # plt.show()

    x=GT[0]+(coord[0]*GT[1] + coord[1]*GT[2])
    y=GT[3]+(coord[0]*GT[4] + coord[1]*GT[5])

    x,y,z = transform.TransformPoint(x,y)
    # print(x ,y)
    f.write(str(x)+" "+str(y)+"\n")
f.close()



    # print(x,y)


