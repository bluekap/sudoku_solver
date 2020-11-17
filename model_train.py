#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
import numpy as np


# In[2]:


mnist = tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()


# In[3]:


X_train = x_train.astype("float32") / 255
X_test = x_test.astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")


# In[24]:


np.random.seed(1337)  # for reproducibility

# input image dimensions
img_rows, img_cols = X_train.shape[1], X_train.shape[2]

# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3) 
input_shape = (img_rows, img_cols, 1)


# In[45]:


nb_filters = 32
batch_size=128
nb_epoch=5
nb_classes=10


# In[23]:


model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[32]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

####Training####
model.fit(X_train_new, y_train, batch_size=128,epochs=30,
          verbose=1)
score = model.evaluate(X_test_new, y_test, verbose=0)


# In[25]:


X_train_new = X_train.reshape(-1,28,28,1)
X_test_new = X_test.reshape(-1,28,28,1)


# In[19]:


X_train_new.shape


# In[29]:


# def train_detector(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=128, nb_epoch=5, nb_classes=2):
#     """ vgg-like deep convolutional network """
    


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

####Training####
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save("my_digit_recognizer.model")


# In[21]:


train_detector(x_train,x_test,y_train,y_test,nb_classes=10)


# In[41]:


X_train.shape


# In[44]:


X_train[0]


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


plt.imshow(X_train[0])


# In[ ]:




