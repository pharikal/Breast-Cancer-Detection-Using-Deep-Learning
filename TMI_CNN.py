# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:32:17 2020

@author: Alokparna
"""

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize
import itertools
import scipy.io
import time
import warnings
warnings.filterwarnings("ignore")

    
"""
-----------------------------------------------------------------------------------------------
Create a CNN class inheriting the Keras Model
-----------------------------------------------------------------------------------------------
"""
class CNN(Model):

    def __init__(self):
        super(CNN, self).__init__()
        
        """
        ---------------------------------------------------------------------------------------
        Add the Layeys
        ---------------------------------------------------------------------------------------
        """ 
        self.conv1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation=tf.nn.relu)
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(1, 1))
        self.conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1), activation=tf.nn.relu)
        self.maxpool4 = MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(1, 1))
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, activation=tf.nn.relu)
        self.dense2 = Dense(units=128, activation=tf.nn.relu)
        self.classification = Dense(units=2, activation=tf.nn.sigmoid)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.classification(x)
  

"""
-----------------------------------------------------------------------------------------------
Function for loading TMI dataset and Data-Preprocessing
-----------------------------------------------------------------------------------------------
""" 
def load_TMI_data():
    
    """
    -------------------------------------------------------------------------------------------
     Load the dataset
    -------------------------------------------------------------------------------------------
    """ 
    dataset = scipy.io.loadmat('Dataset/TMI2015/training/training.mat')
    
    """
    -------------------------------------------------------------------------------------------
     Split into train and test. Values are in range [0..1] as float64
    -------------------------------------------------------------------------------------------
    """
    X_train = np.transpose(dataset['train_x'], (3, 0, 1, 2))
    y_train = list(dataset['train_y'][0])
    
    X_test = np.transpose(dataset['test_x'], (3, 0, 1, 2))
    y_test = list(dataset['test_y'][0])
    
    """
    -------------------------------------------------------------------------------------------
     Change shape and range.
    -------------------------------------------------------------------------------------------
    """
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)
    
    """
    -------------------------------------------------------------------------------------------
     1-> 0 : Non-nucleus. 2 -> 1: Nucleus
    -------------------------------------------------------------------------------------------
    """
    y_test -= 1
    y_train -= 1
    
    """
    -------------------------------------------------------------------------------------------
     Resize to 32x32
    -------------------------------------------------------------------------------------------
    """
    X_train_resized = np.empty([X_train.shape[0], 32, 32, X_train.shape[3]])
    for i in range(X_train.shape[0]):
        X_train_resized[i] = resize(X_train[i], (32, 32, 3), mode='reflect')

    X_test_resized = np.empty([X_test.shape[0], 32, 32, X_test.shape[3]])
    for i in range(X_test.shape[0]):
        X_test_resized[i] = resize(X_test[i], (32, 32, 3), mode='reflect')
    
    """
    -------------------------------------------------------------------------------------------
     Normalize images from [0..1] to [-1..1]
    -------------------------------------------------------------------------------------------
    """
    X_train_resized = 2 * X_train_resized - 1
    X_test_resized = 2 * X_test_resized - 1
    
    """
    -------------------------------------------------------------------------------------------
     One-hot encoding the train and test labels
    -------------------------------------------------------------------------------------------
    """    
    num_classes = 2
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return X_train_resized, y_train, X_test_resized, y_test


"""
-----------------------------------------------------------------------------------------------
Function to plot the Confusion Matrix
-----------------------------------------------------------------------------------------------
"""
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    

"""
-----------------------------------------------------------------------------------------------
Load the TMI dataset and Data-Preprocess the data
-----------------------------------------------------------------------------------------------
""" 
x_train, y_train, x_test, y_test = load_TMI_data()


"""
-----------------------------------------------------------------------------------------------
Instantiate a CNN Model
-----------------------------------------------------------------------------------------------
"""
cnn_model = CNN()


"""
-----------------------------------------------------------------------------------------------
Compile the CNN Model
Loss = categorical_crossentropy
Optimizer = Adam
-----------------------------------------------------------------------------------------------
"""
cnn_model.compile(loss=['categorical_crossentropy'], 
                  optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.1), 
                  metrics = ['accuracy'])


"""
-----------------------------------------------------------------------------------------------
Train the CNN Model
-----------------------------------------------------------------------------------------------
"""
start = time.time()

history = cnn_model.fit(x_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(x_test, y_test))

end = time.time()
print ("\nTraining time: %0.1f minutes \n" % ((end-start) / 60))


"""
-----------------------------------------------------------------------------------------------
Evaluate the CNN Model
-----------------------------------------------------------------------------------------------
"""
score = cnn_model.evaluate(x_test, y_test, verbose=0)
loss = score[0]
acc = score[1]

print('[INFO] Test loss: {:5.2f}%'.format(100*loss))
print('[INFO] Test accuracy: {:5.2f}%'.format(100*acc))


"""
-----------------------------------------------------------------------------------------------
Save the Trained CNN Model
-----------------------------------------------------------------------------------------------
"""
cnn_model.save('TMI_CNN.model', save_format='tf')


"""
-----------------------------------------------------------------------------------------------
Plot Training History - Accuracies and Losses
-----------------------------------------------------------------------------------------------
"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()


"""
-----------------------------------------------------------------------------------------------
Create a Classification Report
-----------------------------------------------------------------------------------------------
"""
y_pred = np.round(cnn_model.predict(x_test),0)
class_names = ['Non-Nuclei', 'Nuclei']
print("Classification report:\n %s\n" 
      % (classification_report(y_test, y_pred, target_names=class_names)))


"""
-----------------------------------------------------------------------------------------------
Plot the Confusion Matrix
-----------------------------------------------------------------------------------------------
"""
categorical_test_labels = pd.DataFrame(y_test).idxmax(axis=1)
categorical_preds = pd.DataFrame(y_pred).idxmax(axis=1)
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

plt.figure()
plot_confusion_matrix(confusion_matrix, class_names, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix, class_names, normalize=True, title='Normalized confusion matrix')

