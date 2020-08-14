import cv2
import numpy as np
import os
import sys 
import glob 
import random
import imutils
import json

def resize(x, w,h):
    return cv2.resize(x,(w,h))

def custom_read(x):
    return cv2.imread(x,cv2.IMREAD_UNCHANGED)

sword_w, sword_h = 500,600
bg_w,bg_h = 1280,720
DATASET_SIZE = 200

swords = []
labels = []
backgrounds = []

path = "swords/"
for f in os.listdir(path):
    img_cv = cv2.imread(path+f,cv2.IMREAD_UNCHANGED)
    swords.append(resize(img_cv,sword_w,sword_h))

path = "bg/"
for f in os.listdir(path):
    img_cv = cv2.imread(path+f,cv2.IMREAD_UNCHANGED)
    backgrounds.append(img_cv)

S_COUNT = len(swords)
BG_COUNT = len(backgrounds)

def overlay_with_alpha(x_offset, y_offset, bg,sword):
    y1, y2 = y_offset, y_offset + sword.shape[0]
    x1, x2 = x_offset, x_offset + sword.shape[1]
    mask = sword[:,:,3] /255.0
    
    for i in range(0, 3):
        bg[y1:y2, x1:x2, i] = (mask * sword[:, :, i] + (1.0 - mask) * bg[y1:y2, x1:x2, i])

    return bg

for i in range(DATASET_SIZE):
    bg = backgrounds[random.randrange(0,BG_COUNT,1)].copy()
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    bg[:,:,3] = 255
    
    r_s = random.randrange(0,S_COUNT,1)
    sword = swords[r_s].copy()

    rand_angle  = random.randrange(0,359,1)
    sword = imutils.rotate_bound(sword, rand_angle)
    mask = sword[:,:,3]
    bb = cv2.boundingRect(mask)
    bx,by,bw,bh = bb
    sword = sword[by:by+bh,bx:bx+bw]

    x_offset = random.randrange(0,bg.shape[1] - sword.shape[1])
    y_offset = random.randrange(0,bg.shape[0] - sword.shape[0])
    final = overlay_with_alpha(x_offset,y_offset,bg,sword)
    
    bbox = ((x_offset,y_offset),(x_offset + sword.shape[1], y_offset + sword.shape[0]))
    point = (x_offset+sword.shape[1]//2,y_offset+sword.shape[0]//2)

    if (random.random() <= 0.8):
        # Jesli na zbiorze treningowym mozna rysowac bounding boxa.
        # cv2.rectangle(final, bbox[0],bbox[1], (0,0,255), 20)
        labels.append({
            'filename/index' : i,
            'center point' : point,
            'bbox' : bbox
        })

        filename = 'datasets/training/{}.jpg'.format(i)
        cv2.imwrite(filename,final) 
    else:
        # cv2.rectangle(final, bbox[0],bbox[1], (0,0,255), 20)
        filename = 'datasets/testing/{}.jpg'.format(i)
        cv2.imwrite(filename,final) 

# zapisywanie do jsona
with open('labels.json', 'w') as out:
    json.dump(labels,out)

