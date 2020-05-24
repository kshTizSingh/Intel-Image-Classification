# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:33:50 2020

@author: INSPIRON 3543
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale = 1./255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               rotation_range = 45,
                               horizontal_flip = True)

test_gen = ImageDataGenerator(rescale = 1./255)

training_set=  train_gen.flow_from_directory(r'F:\intel_dataset\seg_train',
                                             target_size = (64,64),
                                             batch_size =32,
                                             class_mode = 'categorical')
testing_set = test_gen.flow_from_directory(r'F:\intel_dataset\seg_test',
                                           target_size= (64,64),
                                           batch_size = 32,
                                           class_mode=  'categorical')


# model 

from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from keras.models import Sequential

cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3,3),
               activation = 'relu',
               input_shape = (64,64,3)))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Conv2D(32, kernel_size=(3,3),
               activation = 'relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Flatten())

cnn.add(Dense(256, activation= 'relu'))
cnn.add(Dropout(0.1))
cnn.add(Dense(6, activation = 'softmax'))

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set,validation_data = testing_set, steps_per_epoch = 439, validation_steps = 94, epochs = 20)

cnn.save("intel_cnn_model.h5")

# predicting

from keras.preprocessing import image

pred_images = image.load_img(r'F:\intel_dataset\seg_pred\225.jpg', target_size = (64,64))
pred_images = image.img_to_array(pred_images)
print(pred_images)
# to convert 3d image to 4d for tensorflow

pred_images = np.expand_dims(pred_images, axis =0)

result = cnn.predict(pred_images)
if result[0][0]==1:
    print("Buildings")
elif result[0][1]==1:
    print("Forest")
elif result[0][2]==1:
    print("Glacier")
elif result[0][3]==1:
    print("Mountain")
elif result[0][4]==1:
    print("Sea")
else:
    print("Street")

class_names=  (['Buildings','Forest', 'Glacier','Mountain','Sea','Street'])

plt.figure(figsize=(10,10))
#for _ in range(3):
rand_x,rand_y = next(testing_set) 
pred_ = cnn.predict(rand_x)
for i in range(15):
    pred,y = pred_[i].argmax(), rand_y[i].argmax()
    plt.subplot(4,4,i+1)
    plt.imshow(rand_x[i])
    title= 'prediction: ' + str(class_names[pred]) 
    plt.title(title, size = 8)
    plt.xticks(fontsize =5)
    plt.yticks(fontsize =5)
plt.show()
    
    
    