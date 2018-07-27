# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 07.07.2018
# Aim：将原始数据绘成二维图并保存
# Highlight: Python中列表元素转为数字的方法

import numpy as np
import matplotlib.pyplot as plt
import xlrd
import skimage




# Load the xls file
ExcelFile = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\data.try.xlsx')
ExcelFile0 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\data.try.xlsx')

# Print all sheets in excel files
print(ExcelFile.sheet_names())
print(ExcelFile0.sheet_names())

# Get the contents of sheet
sheet = ExcelFile.sheet_by_name('Sheet1')
sheet0 = ExcelFile0.sheet_by_name('Sheet1')
print(sheet.name, sheet.nrows, sheet.ncols)
# Get the whole value of sheet
# Highlight: Python中列表元素转为数字的方法
cols1 = np.array(sheet.col_values(0))
cols2 = np.array(sheet.col_values(1))
cols3 = np.array(sheet.col_values(2))
cols4 = np.array(sheet.col_values(3))
cols5 = np.array(sheet.col_values(4))
cols6 = np.array(sheet.col_values(5))
cols7 = np.array(sheet.col_values(6))
cols8 = np.array(sheet.col_values(7))



#
plt.plot(cols1, cols2, color='#0000FF', marker='.')
plt.xlabel('Depths')
plt.plot(cols5, cols6, color='#0000FF', marker='.')
plt.show()



cols0 = np.array(sheet0.col_values(0))

cols1 = np.delete(np.array(sheet.col_values(0)), 0)
cols2 = np.delete(np.array(sheet.col_values(1)), 0)
cols3 = np.delete(np.array(sheet.col_values(2)), 0)

cols4 = np.delete(np.array(sheet.col_values(5)), 0)
cols5 = np.delete(np.array(sheet.col_values(6)), 0)
cols6 = np.delete(np.array(sheet.col_values(7)), 0)

cols7 = np.delete(np.array(sheet.col_values(10)), 0)
cols8 = np.delete(np.array(sheet.col_values(11)), 0)
cols9 = np.delete(np.array(sheet.col_values(12)), 0)

cols10 = np.delete(np.array(sheet.col_values(15)), 0)
cols11 = np.delete(np.array(sheet.col_values(16)), 0)
cols12 = np.delete(np.array(sheet.col_values(17)), 0)

cols13 = np.delete(np.array(sheet.col_values(20)), 0)
cols14 = np.delete(np.array(sheet.col_values(21)), 0)
cols15 = np.delete(np.array(sheet.col_values(22)), 0)

cols16 = np.delete(np.array(sheet.col_values(25)), 0)
cols17 = np.delete(np.array(sheet.col_values(26)), 0)
cols18 = np.delete(np.array(sheet.col_values(27)), 0)

cols19 = np.delete(np.array(sheet.col_values(30)), 0)
cols20 = np.delete(np.array(sheet.col_values(31)), 0)
cols21 = np.delete(np.array(sheet.col_values(32)), 0)

cols22 = np.delete(np.array(sheet.col_values(35)), 0)
cols23 = np.delete(np.array(sheet.col_values(36)), 0)
cols24 = np.delete(np.array(sheet.col_values(37)), 0)

cols25 = np.delete(np.array(sheet.col_values(40)), 0)
cols26 = np.delete(np.array(sheet.col_values(41)), 0)
cols27 = np.delete(np.array(sheet.col_values(42)), 0)



plt.plot(cols0, cols1, color='#0000FF', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols4, color='#000000', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols7, color='#A52A2A', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols10, color='#8B0000', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols13, color='#FF1493', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols16, color='#FFD700', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols19, color='#008000', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols22, color='#87CEFA', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols25, color='#800080', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols2, color='#0000FF', marker='.')
plt.xlabel('Depths')
plt.plot(cols0, cols3, color='#0000FF', marker='.')
plt.xlabel('Depths')
plt.show()



# 引入随机噪声:
c = skimage.util.random_noise(cols2, mode='gaussian', seed=None, clip=True)
a = np.arange(0, 200)
plt.plot(a, c, color='#0000FF', marker='.')
plt.plot(a, cols2, color='#8B0000', marker='.')
plt.show()


# Load the xls file
xlsx136 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\136.xlsx')
xlsx138 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\138.xlsx')
xlsx140 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\140.xlsx')
xlsx142 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\142.xlsx')
xlsx144 = xlrd.open_workbook('C:\\Users\\wangya\\Desktop\\MedPhysics\\class_energy\\144.xlsx')


# Get the contents of sheet
sheet136 = xlsx136.sheet_by_name('Sheet1')
sheet138 = xlsx138.sheet_by_name('Sheet1')
sheet140 = xlsx140.sheet_by_name('Sheet1')
sheet142 = xlsx142.sheet_by_name('Sheet1')
sheet144 = xlsx144.sheet_by_name('Sheet1')
print(sheet136.name, sheet136.nrows, sheet136.ncols)


# Get the whole value of sheet(excel中一定不能有占位符！！！不然会被认为都是字符格式的)
x_train = []
y_train = []
for i in range(50, 99):
    col_136 = sheet136.col_values(i)
    x_train.append(col_136)
    y_train.append(0)

    col_138 = sheet138.col_values(i)
    x_train.append(col_138)
    y_train.append(1)

    col_140 = sheet140.col_values(i)
    x_train.append(col_140)
    y_train.append(2)

    col_142 = sheet142.col_values(i)
    x_train.append(col_142)
    y_train.append(3)

    col_144 = sheet144.col_values(i)
    x_train.append(col_144)
    y_train.append(4)


# Translate list into array
x_train = np.array(x_train)
y_train = np.array(y_train)


a = x_train[0:9:2, :]
c = skimage.util.random_noise(a, mode='gaussian', seed=None, clip=True)
a = np.arange(0, 200)
plt.plot(a, x_train[0, :], color='#0000FF', marker='.')
plt.plot(a, x_train[1, :], color='#8B0000', marker='.')
plt.show()
a = np.arange(0, 200)
plt.plot(a, x_train[0, :], color='#0000FF', marker='.')
plt.plot(a, c[0, :], color='#8B0000', marker='.')
plt.show()
