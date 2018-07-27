# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 07.09.2018
# Aim：根据不同能量对应的1D PET重建图像给出一个判据，希望输入另一组1D图像来给出相应能量的区间
from __future__ import absolute_import, division, print_function
import numpy as np
import xlrd
import tensorflow as tf
import matplotlib.pyplot as plt




# Load the xls file
ClassFiles = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\DifferentEnergy.xlsx')


# Print all sheets in excel files
print(ClassFiles.sheet_names())

# Get the contents of sheet
sheetofclass = ClassFiles.sheet_by_name('Sheet1')
print(sheetofclass.name, sheetofclass.nrows, sheetofclass.ncols)


# Get the whole value of sheet(excel中一定不能有占位符！！！不然会被认为都是字符格式的)
E136 = np.delete(np.array(sheetofclass.col_values(1)), 0)
E138 = np.delete(np.array(sheetofclass.col_values(2)), 0)
E140 = np.delete(np.array(sheetofclass.col_values(3)), 0)
E142 = np.delete(np.array(sheetofclass.col_values(4)), 0)
E144 = np.delete(np.array(sheetofclass.col_values(5)), 0)

# 合并几组数得到训练集
input_list = [sheetofclass.col_values(1), sheetofclass.col_values(2), sheetofclass.col_values(3),
              sheetofclass.col_values(4), sheetofclass.col_values(5)]
label_list = [sheetofclass.cell_value(0, 1), sheetofclass.cell_value(0, 2), sheetofclass.cell_value(0, 3),
              sheetofclass.cell_value(0, 4), sheetofclass.cell_value(0, 5)]

# np.delete函数参数：数组；需要删除的位置，这里是列数；1表示列
X_train = np.delete(np.array(input_list), 0, axis=1)
# 0到4分别代表能量值为136、138、140、142、144
Y_train = np.array([0, 1, 2, 3, 4])
# Converts a class vector (integers) to binary class matrix.
# 并不清楚这个函数的num_classes这个参数是否需要？？？
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=5)




# Build NN
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(units=5, activation=tf.nn.softmax))

# Define optimizer
model.compile(loss='mean_squared_error', optimizer='sgd', matrics=['accuracy'])

# Training
print('Training')
model.fit(X_train, Y_train, epochs=1, batch_size=4)


plt.plot(np.arange(0, 200), E136, color='#0000FF', marker='.')
plt.show()


