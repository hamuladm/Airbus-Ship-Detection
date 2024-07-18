import cv2 as cv
from keras import models
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from config import *

model = models.load_model(MODEL_PATH, compile=False)

def gen_pred(img, model):
    rgb_path = os.path.join(TEST_PATH, img)
    img = cv.imread(rgb_path)
    img = img[::IMG_SCALING[0], ::IMG_SCALING[1]]
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img / 255
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv.imread(rgb_path), pred


def run_pred():
    rows = 1
    columns = 2

    test_imgs = [random.choice(os.listdir(TEST_PATH)) for _ in range(5)]
    print(test_imgs)
    for i in range(len(test_imgs)):
        img, pred = gen_pred(test_imgs[i], model)
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{test_imgs[i]}')
        fig.add_subplot(rows, columns, 2)
        plt.imshow(pred, interpolation=None)
        plt.axis('off')
        plt.title("Prediction")
    plt.show()

if __name__ == '__main__':
    run_pred()
