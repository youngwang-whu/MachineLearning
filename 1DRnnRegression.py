# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 07.10.2018
# Aim：用RNN得到一维曲线的变换

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np




# Build RNN
model = tf.keras.Sequential()
# build a LSTM RNN
model.add(tf.keras.layers.LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    return_sequences=True,      # True: output at all steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)
