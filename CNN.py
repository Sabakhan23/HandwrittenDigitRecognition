#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras import layers
from tensorflow.keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(10,activation='softmax'))

model.summary()


# In[2]:


train_images=train_images.reshape((60000,28,28,1))
train_images.astype('float32')/255 #scaling

test_images=test_images.reshape((10000,28,28,1))
test_images.astype('float32')/255 #scaling

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)


# In[3]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[4]:


model.fit(train_images,train_labels,epochs=10,batch_size=64)

test_loss,test_acc=model.evaluate(test_images,test_labels)

print(test_acc)

model.save('mnist.h5')


# In[ ]:




