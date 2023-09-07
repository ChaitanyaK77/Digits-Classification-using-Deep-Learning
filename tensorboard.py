# -*- coding: utf-8 -*-
"""TensorBoard.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b8Rqs-hvRH8ywMCZRioXPPMX4ddwWusz
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

(X_train,y_train) , (X_test,y_test) = keras.datasets.mnist.load_data()

len(X_train)

len(X_test)

X_train.shape

X_train[0]

plt.matshow(X_train[2])

"""LABELS:"""

y_train[2]

y_train[:5]

X_train = X_train / 255
X_test = X_test / 255  #scaling

"""Flattening the arrays(reshape)"""

X_train_flattened=X_train.reshape(len(X_train),28*28)
X_train_flattened.shape

X_test_flattened = X_test.reshape(len(X_test),28*28)
X_test_flattened.shape

X_train_flattened[0]

rm -rf ./logs/

"""Simple Neural Network (INPUT_Layer=784 elements,OUTPUT_Layer = 10 elements)"""

import datetime
model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(
optimizer='adam',
loss ='sparse_categorical_crossentropy',
metrics =['accuracy']
    )

model.fit(X_train_flattened,y_train,epochs=10,callbacks = [tensorboard_callback])

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs/fit

model.evaluate(X_test_flattened,y_test)

plt.matshow(X_test[0])

y_pred = model.predict(X_test_flattened)

np.argmax(y_pred[0])

y_pred_labels = [np.argmax(i) for i in y_pred]
y_pred_labels[:5]

cm = tensorflow.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)
cm

import seaborn as sns
plt.figure(figsize = (10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Truth')

model = keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')

])

model.compile(
optimizer='adam',
loss ='sparse_categorical_crossentropy',
metrics =['accuracy']
    )

model.fit(X_train_flattened,y_train,epochs=10)

model.evaluate(X_test_flattened,y_test)

y_pred = model.predict(X_test_flattened)

y_pred_labels = [np.argmax(i) for i in y_pred]

cm = tensorflow.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)

import seaborn as sns
plt.figure(figsize = (10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Truth')

"""Much better"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')

])

model.compile(
optimizer='adam',
loss ='sparse_categorical_crossentropy',
metrics =['accuracy']
    )

model.fit(X_train,y_train,epochs=10)
