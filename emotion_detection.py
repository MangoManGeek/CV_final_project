import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np

from skimage.io import imread
from skimage.transform import resize

from skimage.color import rgb2gray

import matplotlib.pyplot as plt


def create_model():
	model = models.Sequential()

	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7, activation='softmax'))
    
	return model


model=create_model()
model.load_weights('./weights/emotion_detect_without_augumentation')
EMOTION = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

def predict_emotion(face_img):
	face_img = resize(face_img, [48,48])
	face_img = rgb2gray(face_img)
	face_img = np.reshape(face_img, [1,48,48,1])
	pred = model.predict(face_img)
	idx = np.argmax(pred)
	return EMOTION[idx]





