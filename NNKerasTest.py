# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Aim:   实验：探究神经网络究竟提取的图片特征是什么，效果怎样，是否适用我们的课题
# Date: 07.05.2018

# 引入python相关包
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 读入文件并显示
picture = cv2.imread('C:\\Users\\wangya\\Desktop\\50.jpg')
cv2.imshow('picture', picture)


# Build NN(中间注释部分可选加)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    filters=3, kernel_size=[3, 3],
    kernel_initializer=tf.keras.initializers.Constant(value=0.12),
    input_shape=picture.shape, name='conv_1'))
"""
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Conv2D(
    filters=3, kernel_size=[3, 3],
    input_shape=picture.shape, name='conv_2'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Activation('relu'))
"""
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(8, activation='relu', name='dens_1'))

model.save_weights('picture.w')



# Build NN layers2
model2 = tf.keras.Sequential()

model2.add(tf.keras.layers.Conv2D(
    3, 3, 3, input_shape=picture.shape))

model2.load_weights('picture.w', by_name=True)


# keras只能按批处理，需要将数据变成4维
picture_batch = np.expand_dims(picture, axis=0)
conv_img = model2.predict(picture_batch)
img = np.squeeze(conv_img, axis=0)
# 显示处理过的图片
cv2.imshow('a', img)

# plt.imshow(img)要处理RGB值的问题，可选用



# 这部分功能可选用
"""
# 像素归一化*255
def visualize(img):
    img = np.squeeze(img, axis=0)

    max_img = np.max(img)
    min_img = np.min(img)
    img = img-(min_img)
    img = (img/(max_img-min_img))*255
    cv2.imshow('ProcessedImg', img)

visualize(conv_img)
"""

