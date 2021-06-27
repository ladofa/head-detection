import scipy.io as sio
import json
import numpy as np
import random

from image_info import get_image_size

#create coco base
coco_train = dict()
coco_train['images'] = []
coco_train['annotations'] = []
coco_train['categories'] = [
    {
        'id':1,
        'name':'head'
    }
]

coco_val = dict()
coco_val['images'] = []
coco_val['annotations'] = []
coco_val['categories'] = [
    {
        'id':1,
        'name':'head'
    }
]

coco_image_id = 1
coco_annotation_id = 1
image_prefix = 'images/'
root_dir = '/home/ladofa9/test/'
image_fullpath_prefix = root_dir + 'images/'
mat_path = root_dir + 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
coco_path_train = root_dir + 'mpii_coco_train.json'
coco_path_val = root_dir + 'mpii_coco_val.json'
mat = sio.loadmat(mat_path)

image_count = mat['RELEASE'][0]['annolist'][0][0].shape[0]
annolist = mat['RELEASE'][0]['annolist'][0][0]
train = mat['RELEASE'][0]['img_train'][0][0]

print('image count', image_count)

for i in range(image_count):

    if i % 100 == 0:
        print('i', i)
    anno = annolist[i]

    if anno['annorect'].shape == (0, 0):
        continue

    annorect = anno['annorect'][0]
    if annorect[0] is None:
        continue
    if 'x1' not in annorect.dtype.names:
        continue

    filename = anno['image'][0]['name'][0][0]
    fullpath = image_fullpath_prefix + filename
    relpath = image_prefix + filename

    image_size = get_image_size(fullpath)
    if image_size is None:
        continue

    if random.random() > 0.1:
        target_coco = coco_train
    else:
        target_coco = coco_val

    target_coco['images'].append(
        {
            'file_name':image_prefix + filename,
            'width':image_size[0],
            'height':image_size[1],
            'id':coco_image_id,
        }
    )

    person_count = annorect.shape[0]
    
    for p in range(person_count):
        rect = annorect[p]
        x1 = float(rect['x1'][0][0])
        y1 = float(rect['y1'][0][0])
        x2 = float(rect['x2'][0][0])
        y2 = float(rect['y2'][0][0])
        bw = x2 - x1
        bh = y2 - y1

        target_coco['annotations'].append(
            {
                'image_id':coco_image_id,
                'bbox':[x1, y1, bw, bh],
                'area':bw*bh,
                'id':coco_annotation_id,
                'category_id':1,
                'iscrowd':0
            }
        )
        coco_annotation_id += 1

    coco_image_id += 1

print('writing json train', len(coco_train['images']))
json.dump(coco_train, open(coco_path_train, 'w'))

print('writing json val', len(coco_val['images']))
json.dump(coco_val, open(coco_path_val, 'w'))

print('done')