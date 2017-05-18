import glob
import cv2
import os
import numpy as np

workspace = os.getcwd() + '/'
data_folder = 'data/cars/'
image_path = 'set3'
label_path = 'set3_gt'
out_path = 'target'
yolo_results = 'results/tiny_yolo_car.txt'
acf_results = 'results/acf_car.txt'

if not os.path.exists(workspace + data_folder + out_path):
    os.mkdir(workspace + data_folder + out_path)

yolo_recs = {}
lines = open(workspace + yolo_results).readlines()
for line in lines:
    splitline = line.strip().split(' ')
    image_id = splitline[0]
    if image_id not in yolo_recs:
        yolo_recs[image_id] = []
    score = float(splitline[1])
    bbox = [float(x) for x in splitline[2:]]
    rect = {'bbox': bbox, 'score': score}
    yolo_recs[image_id].append(rect)

acf_recs = {}
lines = open(workspace + acf_results).readlines()
for line in lines:
    splitline = line.strip().split(' ')
    image_id = splitline[0]
    if image_id not in acf_recs:
        acf_recs[image_id] = []
    score = float(splitline[1])
    bbox = [float(x) for x in splitline[2:]]
    rect = {'bbox': bbox, 'score': score}
    acf_recs[image_id].append(rect)

def draw_label(image, label, color, font_color, xmin, ymin):
    box, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    poly = np.array([(xmin, ymin-box[1]), (xmin+box[0], ymin-box[1]), (xmin+box[0], ymin), (xmin, ymin)])
    image = cv2.fillConvexPoly(image, poly, color)
    cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, font_color, thickness=1, lineType=cv2.LINE_4)

white = (255, 255, 255)
image_sets = glob.glob(workspace + data_folder + image_path + '/*.jpg')
for idx, image_name in enumerate(image_sets):
    image_id = image_name.strip().split('/')[-1].replace('.jpg', '')
    image = cv2.imread(image_name.strip())

    label_name = workspace + data_folder + label_path + '/' + image_id + '.txt'
    if not os.path.exists(label_name):
        print('%s: ground truth not exists.' % image_id)
    else:
        green = (0, 255, 0)
        lines = open(label_name).readlines()
        for line in lines:
            splitline = line.strip().split(',')
            if (splitline[0] in ['p', 'f', 'w']):
                xmin = int(splitline[2])
                ymin = int(splitline[3])
                xmax = int(splitline[2]) + int(splitline[4])
                ymax = int(splitline[3]) + int(splitline[5])
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), green, thickness=1)

    if image_id not in yolo_recs:
        print('%s: no detection from yolo.' % image_id)
    else:
        red = (0, 0, 255)
        for rect in yolo_recs[image_id]:
            if (rect['score'] > 0.24):
                xmin = int(rect['bbox'][0])
                ymin = int(rect['bbox'][1])
                xmax = int(rect['bbox'][2])
                ymax = int(rect['bbox'][3])
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), red, thickness=1)
                # draw_label(image, '%.0f%%'%(rect['score']*100), red, white, xmin, ymin)

    if image_id not in acf_recs:
        print('%s: no detection from acf.' % image_id)
    else:
        blue = (255, 0, 0)
        for rect in acf_recs[image_id]:
            xmin = int(rect['bbox'][0])
            ymin = int(rect['bbox'][1])
            xmax = int(rect['bbox'][2])
            ymax = int(rect['bbox'][3])
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), blue, thickness=1)

    cv2.imwrite(workspace + data_folder + out_path + '/' + image_id + '.jpg', image)
    print("%d / %d completed." % (idx + 1, len(image_sets)))
