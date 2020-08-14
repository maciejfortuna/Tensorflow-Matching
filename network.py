import cv2
import numpy as np
import os
import sys 
import glob 
import random
import imutils
import json
import csv
import tensorflow as tf
from tensorflow import keras
try:
    from sklearn.model_selection import train_test_split
except:
    pass
import matplotlib.pyplot as plt

orig_w = 1280
orig_h = 720
w = 128
h = 128

training_dataset_path =  "datasets/training/"
testing_dataset_path =  "datasets/testing/"

def resize(x, w,h):
    return cv2.resize(x,(w,h))

# Wczytywanie etykiet i zbioru treningowego
dataset = []
bbox_label = []
point_label = []
json_array = json.load(open('labels.json'))

for item in json_array:
    # przeskalowanie koordynatow punktu po przeskalowaniu zdjecia
    point_label.append(np.divide(item['center point'],[orig_w//w , orig_h// h]))
    img_cv = cv2.imread(training_dataset_path + str(item['filename/index']) + '.jpg',cv2.IMREAD_UNCHANGED)
    img_cv = resize(img_cv,w,h)
    img_cv = img_cv / 255.0
    dataset.append(img_cv)

dataset = np.array(dataset)
point_label = np.array(point_label,dtype=np.int32)

# Wczytanie zbioru testowego
test_data = []
orig_test_data = []
for f in os.listdir(testing_dataset_path):
    img_cv = cv2.imread(testing_dataset_path+f)
    img_cv = resize(img_cv,w,h)
    orig_test_data.append(img_cv)
    img_cv = img_cv / 255.0
    test_data.append(img_cv)

'''Jesli w create_dataset.py stworzymy zbior bez podzialu na trening/test to tutaj tez mozna podzielic'''
# train_data, test_data, train_label, test_label = train_test_split(dataset, point_label, test_size=0.2, random_state=10)
test_data = np.array(test_data)
# MACHINE LEARNING
LOAD_MODEL = True
SAVE_MODEL = False

def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3),  input_shape= (w,h, 3)))
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Conv2D(32, (3,3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(4, (2,2)))
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2, activation='linear'))
    model.compile(loss = 'mean_squared_error', optimizer=keras.optimizers.Adam( lr=0.0001 ), metrics =[])
    return model

model = None

if(LOAD_MODEL):
    model = tf.keras.models.load_model('models/my_model.h5')
else:
    model = create_model()
    model.fit(dataset,point_label, batch_size=8, epochs=25)

if(SAVE_MODEL):
    model.save('models/my_model.h5')

# results = model.evaluate(test_data,  test_label, verbose = 2)
# print('\nTest accuracy:', results)
model.summary()
points = model.predict( test_data )

# Pokazanie rezultatow
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    bb = points[i]
    temp = orig_test_data[i].copy()
    cv2.circle(temp, (int(bb[0]),int(bb[1])),10, (0,255,0), 3)

    plt.subplot(num_rows, num_cols, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()