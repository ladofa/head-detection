import cv2
import json
import os
import numpy as np

root_dir = '/home/ladofa9/test/'
coco_path_train = root_dir + 'mpii_coco_train.json'
coco_path_val = root_dir + 'mpii_coco_val.json'

data = json.load(open(coco_path_train, 'r'))
image = data['images'][1]
annos = [anno for anno in data['annotations'] if anno['image_id'] == image['id']]
full_path = root_dir + image['file_name']
src = cv2.imread(full_path)

for anno in annos:
    bbox = anno['bbox']
    bbox = tuple(np.array(bbox, dtype=np.int32))
    cv2.rectangle(src, bbox, (255, 255, 255), 2)

cv2.imwrite('test.jpg', src)