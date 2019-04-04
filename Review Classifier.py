#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
print(tf.__version__)#should be 1.12.0


# In[11]:


#gets the data from datases stored by keras library
#the returned data is preprocessed data (words are converted to integers)
#integers represent the word in a predefined dictionary which is stored and maintained by keras
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)#num_words=10000 which specifies the number of distinct words to be considered


# In[4]:


#splitting th data in test and training sets 50-50 basis
print("Training entries: {}, labels: {}".format(len(train_data),len(train_labels)))#visualizing the train and test data


# In[5]:


print(train_data[0])#test


# In[6]:


print(len(train_data[0]))#number of words in a particular review


# In[61]:


#A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

#The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
#func to convert integers to words 
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# In[13]:


decode_review(train_data[0])


# In[14]:


#making the length of each review equal
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                      value=word_index["<PAD>"],
                                                      padding='post',
                                                      maxlen=256)


# In[15]:


print(train_data[0])


# In[16]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
#build your model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# In[17]:


#compile the model and the train
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['acc'])


# In[18]:


#splitting the training data into training set and validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# In[19]:


history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=40,
                   batch_size=512,
                   validation_data=(x_val, y_val),
                   verbose=1)


# In[20]:


results = model.evaluate(test_data, test_labels)


# In[21]:


#print the Result of the evaluation [loss, accuracy]
print(results)


# In[22]:


#gives the history of the training
history_dict = history.history
history_dict.keys()


# In[23]:


import matplotlib.pyplot as plt

#plotting the training curve TIP:(more the vaidation accuracy more the accuracy of the model)

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[24]:


plt.clf() #clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[27]:


print(train_labels)


# In[31]:


plt.plot(train_labels[:10],'r+')


# In[63]:


#plt.plot(test_labels[:10],'r+')

#here you specify the number of instances you want to predict from the test data
x=10
predictions = model.predict(test_data[:x])
print(predictions)


# In[37]:


predicted_values = []
for i in predictions:
    predicted_values.append(i[0])


# In[38]:


print(predicted_values)


# In[40]:


plt.plot(test_labels[:10],'r+',label="Test-values")
plt.plot(predicted_values,'go',label="Predicted-values")
plt.title('Test-values vs. Predicted-values')
plt.xlabel('Test-cases')
plt.ylabel('Positive-Negative')
plt.legend()

plt.show()


# In[51]:


decode_review(test_data[1])        


# In[60]:


for index,data in enumerate(test_data[:5]):
    print(decode_review(data))
    print("Actual:")
    if test_labels[index] == 0:
        print("Negative Review.")
    else:
        print("Positive Review.")
    print()
    print("Prediction by model:")
    if predicted_values[index] > 0.5 :
        print("Positive Review.")
    else:
        print("Negative Review.")
    print("-"*30)


# In[59]:





# In[ ]:




