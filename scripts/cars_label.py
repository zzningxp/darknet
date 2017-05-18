import xml.etree.ElementTree as ET
import pickle
import os
from os.path import join
from PIL import Image
import glob

workspace = os.getcwd() + '/'
data_folder = workspace + 'data/cars/'
trainset = open(data_folder + 'trainset.txt').readlines()
validset = open(data_folder + 'validset.txt').readlines()
sets = []
for idx, subset in enumerate(trainset):
    subsetid = subset.strip()
    sets.append([data_folder + subsetid + '/', data_folder + subsetid + '_gt/', 'train_' + str(idx)])
for idx, subset in enumerate(validset):
    subsetid = subset.strip()
    sets.append([data_folder + subsetid + '/', data_folder + subsetid + '_gt/', 'valid_' + str(idx)])
nameset = open(data_folder + 'cars.names').readlines()
classes = [x.strip() for x in nameset]

print(sets)
print(classes)

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_path, label_path, image_id):
    w, h = Image.open(image_path + image_id + '.jpg').size
    in_file = open(label_path + image_id + '.txt')
    #out_file = open(image_path + 'labels/' + image_id + '.txt', 'w')
    out_file = open(image_path + image_id + '.txt', 'w')
    lines = in_file.readlines()
    for line in lines:
        label = line.split(',')
        if (label[0] in ['p', 'f', 'w']):
            b = (float(label[2]), float(label[2]) + float(label[4]), float(label[3]), float(label[3]) + float(label[5]))
            bb = convert((w, h), b)
            out_file.write('0 ' + ' '.join([str(a) for a in bb]) + '\n')

def get_images(path):
    lists = glob.glob(path + '*.jpg')
    images = []
    for image in lists:
        image = image.split('/')[-1].replace('.jpg', '')
        if (image[0] != '.'):
            images.append(image)
    return images

for image_path, label_path, image_set in sets:
    image_ids = get_images(image_path)
    list_file = open(data_folder + image_set + '.txt', 'w')
    total = 0
    for image_id in image_ids:
        if (os.path.exists(image_path + image_id + '.jpg') and os.path.exists(label_path + image_id + '.txt')):
            total += 1
            list_file.write(image_path + image_id + '.jpg\n')
            convert_annotation(image_path, label_path, image_id)
    list_file.close()
    print(total)

command = 'cat '
for idx, subset in enumerate(trainset):
    command = command + data_folder + 'train_' + str(idx) + '.txt '
command = command + '> ' + data_folder + 'train.txt'
os.system(command)

command = 'cat '
for idx, subset in enumerate(validset):
    command = command + data_folder + 'valid_' + str(idx) + '.txt '
command = command + '> ' + data_folder + 'valid.txt'
os.system(command)
