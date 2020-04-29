# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:35:00 2020

@author: Debadeep
"""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, average_precision_score
from skimage.transform import resize
import scipy.io
import matplotlib.pyplot as plt
import itertools
import time
import shutil
import os

class COGAN():
    
    def __init__(self):
        
        """
        -----------------------------------------------------------------------------------------------
        Define image shape for TMI data(32x32x3)
        -----------------------------------------------------------------------------------------------
        """ 
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        """
        -----------------------------------------------------------------------------------------------
        Define number of class
        -----------------------------------------------------------------------------------------------
        """
        self.num_classes =2
        self.latent_dim = 100
        
        """
        -----------------------------------------------------------------------------------------------
         Define training history
        -----------------------------------------------------------------------------------------------
        """
        self.training_history = {
                'D1_loss': [],
                'D1_acc': [],
                'D2_loss': [],
                'D2_acc': [],
                'G_loss': [],
                'G_acc': [],
                }       

        """
        -----------------------------------------------------------------------------------------------
         Build discriminators
        -----------------------------------------------------------------------------------------------
        """
        self.d1, self.d2 = self.build_discriminators()
        
        """
        -----------------------------------------------------------------------------------------------
         Compile discriminator using Adam optimizer
         binary crossentropy is used to distinguish among real or fake samples
         categorical entropy is to distinguish among which real category is (nuclei or non-nuclei)
        -----------------------------------------------------------------------------------------------
        """ 
        optimizer =  K.optimizers.Adam(learning_rate=0.0002, beta_1= 0.5, decay = 0.01)
        self.d1.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d2.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy'])

        """
        -----------------------------------------------------------------------------------------------
         Build generators
        -----------------------------------------------------------------------------------------------
        """ 
        self.g1, self.g2 = self.build_generators()
        
        """
        -----------------------------------------------------------------------------------------------
         Generators take noise as input and generated imgs
        -----------------------------------------------------------------------------------------------
        """        
        z = K.layers.Input(shape=(self.latent_dim,))
        img1 = self.g1(z)
        img2 = self.g2(z)
        
        """
        -----------------------------------------------------------------------------------------------
         For the combined model only train the generators
        -----------------------------------------------------------------------------------------------
        """
        self.d1.trainable = False
        self.d2.trainable = False
        
        """
        -----------------------------------------------------------------------------------------------
         The valid takes generated images as input and determines validity
        -----------------------------------------------------------------------------------------------
        """
        valid1, _ = self.d1(img1)
        valid2, _ = self.d2(img2)
        
        """
        -----------------------------------------------------------------------------------------------
         The combined model (stacked generators and discriminators)
         Trains generators to fool discriminators
        -----------------------------------------------------------------------------------------------
        """
        self.combined = Model(z, [valid1, valid2])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                    optimizer=optimizer)
        
        

    def build_generators(self):
        
        """
        -----------------------------------------------------------------------------------------------
         Model creation for generators
        -----------------------------------------------------------------------------------------------
        """ 
        model = K.Sequential()
        
        """
        -----------------------------------------------------------------------------------------------
         Shared weights between generators
        -----------------------------------------------------------------------------------------------
        """ 
        model.add(K.layers.Dense(128 * 8 * 8, activation="relu", input_dim=100)) 
        model.add(K.layers.Dense(128, input_dim=self.latent_dim))
        model.add(K.layers.LeakyReLU(alpha=0.2))
        model.add(K.layers.BatchNormalization(momentum=0.8))
        model.add(K.layers.Dense(64))
        model.add(K.layers.LeakyReLU(alpha=0.2))
        model.add(K.layers.BatchNormalization(momentum=0.8))

        noise = K.layers.Input(shape=(self.latent_dim,))
        feature_repr = model(noise)
        
        """
        -----------------------------------------------------------------------------------------------
         Generator 1
        -----------------------------------------------------------------------------------------------
        """
        g1 = K.layers.Dense(32)(feature_repr)
        g1 = K.layers.LeakyReLU(alpha=0.2)(g1)
        g1 = K.layers.BatchNormalization(momentum=0.8)(g1)
        g1 = K.layers.Dense(np.prod(self.img_shape), activation='tanh')(g1)
        img1 = K.layers.Reshape(self.img_shape)(g1)

        """
        -----------------------------------------------------------------------------------------------
         Generator 2
        -----------------------------------------------------------------------------------------------
        """
        g2 = K.layers.Dense(32)(feature_repr)
        g2 = K.layers.LeakyReLU(alpha=0.2)(g2)
        g2 = K.layers.BatchNormalization(momentum=0.8)(g2)
        g2 = K.layers.Dense(np.prod(self.img_shape), activation='tanh')(g2)
        img2 = K.layers. Reshape(self.img_shape)(g2)
        
        """
        -----------------------------------------------------------------------------------------------
         Return model
        -----------------------------------------------------------------------------------------------
        """ 
        return Model(noise, img1), Model(noise, img2)
    
    

    def build_discriminators(self): 
        
        """
        -----------------------------------------------------------------------------------------------
         Model creation for discriminator
        -----------------------------------------------------------------------------------------------
        """ 
        model = K.Sequential()
        
        """
        -----------------------------------------------------------------------------------------------
         Shared discriminator layers
        -----------------------------------------------------------------------------------------------
        """
        model.add(K.layers.Flatten(input_shape=self.img_shape))
        model.add(K.layers.Dense(128))
        model.add(K.layers.LeakyReLU(alpha=0.2))
        model.add(K.layers.Dense(64))
        model.add(K.layers.LeakyReLU(alpha=0.2))
        
        """
        -----------------------------------------------------------------------------------------------
         Generate image
        -----------------------------------------------------------------------------------------------
        """ 
        img1 = K.layers.Input(shape=self.img_shape)
        img2 = K.layers.Input(shape=self.img_shape)
        img1_embedding = model(img1)
        img2_embedding = model(img2)
        
        """
        -----------------------------------------------------------------------------------------------
         valid image for Discriminator 1
        -----------------------------------------------------------------------------------------------
        """ 
        validity1 = K.layers.Dense(1, activation='sigmoid')(img1_embedding)
        """
        -----------------------------------------------------------------------------------------------
         valid image for Discriminator 2
        -----------------------------------------------------------------------------------------------
        """ 
        validity2 = K.layers.Dense(1, activation='sigmoid')(img2_embedding)
        
        """
        -----------------------------------------------------------------------------------------------
         iff the image is real label indicates which type of image it is
        -----------------------------------------------------------------------------------------------
        """ 
        label1 = K.layers.Dense(self.num_classes+1, activation="softmax")(img1_embedding)
        label2 = K.layers.Dense(self.num_classes+1, activation="softmax")(img2_embedding)
        
        """
        -----------------------------------------------------------------------------------------------
         Return model
        -----------------------------------------------------------------------------------------------
        """ 
        return Model(img1, [validity1, label1]), Model(img2, [validity2, label2])
    
    

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, save_interval):        
        
        """
        -----------------------------------------------------------------------------------------------
         Delete directory if exist and create it
        -----------------------------------------------------------------------------------------------
        """ 
        shutil.rmtree('COGAN_generators_output', ignore_errors=True)
        os.makedirs("COGAN_generators_output")
        
        """
        -----------------------------------------------------------------------------------------------
         Images in domain A and B (rotated)
        -----------------------------------------------------------------------------------------------
        """
        X1 = X_train[:int(X_train.shape[0]/2)]
        X2 = X_train[int(X_train.shape[0]/2):]       
        X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))
        Y1 = y_train[:int(y_train.shape[0]/2)]
        Y2 = y_train[int(y_train.shape[0]/2):]
        
        """
        -----------------------------------------------------------------------------------------------
         Adversarial ground truths
        -----------------------------------------------------------------------------------------------
        """
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            """
            -----------------------------------------------------------------------------------------------
             Train Discriminators
             Select a random batch of images
            -----------------------------------------------------------------------------------------------
            """
            idx = np.random.randint(0, X1.shape[0], batch_size)
            imgs1 = X1[idx]
            imgs2 = X2[idx]
            
            """
            -----------------------------------------------------------------------------------------------
             Sample noise as generator input
            -----------------------------------------------------------------------------------------------
            """
            noise = np.random.normal(0, 1, (batch_size, 100))
            
            """
            -----------------------------------------------------------------------------------------------
             Generate a batch of new images
            -----------------------------------------------------------------------------------------------
            """
            gen_imgs1 = self.g1.predict(noise)
            gen_imgs2 = self.g2.predict(noise)
            
            """
            -----------------------------------------------------------------------------------------------
             Convert labels to categorical one-hot encoding
            -----------------------------------------------------------------------------------------------
            """ 
            labels1 = tf.keras.utils.to_categorical(Y1[idx], num_classes=self.num_classes+1)
            labels2 = tf.keras.utils.to_categorical(Y2[idx], num_classes=self.num_classes+1)
            fake_labels = tf.keras.utils.to_categorical(np.full((batch_size, 1), 
                                                        self.num_classes),
                                                        num_classes=self.num_classes+1)
            
            """
            -----------------------------------------------------------------------------------------------
             Train the discriminators
            -----------------------------------------------------------------------------------------------
            """
            d1_loss_real = self.d1.train_on_batch(imgs1, [valid, labels1])
            d2_loss_real = self.d2.train_on_batch(imgs2, [valid, labels2])
            d1_loss_fake = self.d1.train_on_batch(gen_imgs1, [fake, fake_labels])
            d2_loss_fake = self.d2.train_on_batch(gen_imgs2, [fake, fake_labels])
            d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
            d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

            """
            -----------------------------------------------------------------------------------------------
             Train Generators
            -----------------------------------------------------------------------------------------------
            """
            g_loss = self.combined.train_on_batch(noise, [valid, valid])
            
            """
            -----------------------------------------------------------------------------------------------
             Add history data
            -----------------------------------------------------------------------------------------------
            """ 
            self.training_history["D1_loss"].append(d1_loss[0]);
            self.training_history["D1_acc"].append(100*d1_loss[1]);
            self.training_history["D2_loss"].append(d2_loss[0]);
            self.training_history["D2_acc"].append(100*d2_loss[1]);
            self.training_history["G_loss"].append(g_loss[0]);
            self.training_history["G_acc"].append(100*g_loss[1]);

            """
            -----------------------------------------------------------------------------------------------
             Print the result
            -----------------------------------------------------------------------------------------------
            """ 
            print ("%d [D1 loss: %f, acc.: %.2f%%] [D2 loss: %f, acc.: %.2f%%] [G loss: %f]" \
                % (epoch, d1_loss[0], 100*d1_loss[3], d2_loss[0], 100*d2_loss[3], g_loss[0]))
                
            """
            -----------------------------------------------------------------------------------------------
             Evaluate test data for each epoch
            -----------------------------------------------------------------------------------------------
            """   
            self.evaluate_discriminator(X_test, y_test)
            
            """
            -----------------------------------------------------------------------------------------------
             If at save interval => save generated image samples
            -----------------------------------------------------------------------------------------------
            """
            if epoch % save_interval == 0:
                self.save_images(epoch)
                
                
                
    def evaluate_discriminator(self, X_test, y_test):
        
        """
        -----------------------------------------------------------------------------------------------
         Images in domain A and B (rotated)
        -----------------------------------------------------------------------------------------------
        """
        X1_test = X_test[:int(X_test.shape[0]/2)]
        X2_test = X_test[int(X_test.shape[0]/2):]       
        X2_test = scipy.ndimage.interpolation.rotate(X2_test, 90, axes=(1, 2))
        Y1_test = y_test[:int(y_test.shape[0]/2)]
        Y2_test = y_test[int(y_test.shape[0]/2):]
        
        valid1 = np.ones((Y1_test.shape[0], 1))
        valid2 = np.ones((Y2_test.shape[0], 1))
        """
        -----------------------------------------------------------------------------------------------
         Convert labels to categorical one-hot encoding
        -----------------------------------------------------------------------------------------------
        """ 
        labels1 = tf.keras.utils.to_categorical(Y1_test, num_classes=self.num_classes+1)
        labels2 = tf.keras.utils.to_categorical(Y2_test, num_classes=self.num_classes+1)
        
        """
        -----------------------------------------------------------------------------------------------
         Evaluating the trained Discriminator
        -----------------------------------------------------------------------------------------------
        """ 
        scores1 = self.d1.evaluate(X1_test, [valid1, labels1], verbose=0)
        scores2 = self.d2.evaluate(X2_test, [valid2, labels2], verbose=0)

        print("Evaluating D1 [loss:  %.4f, bi-loss: %.4f, cat-loss: %.4f, bi-acc: %.2f%%, cat-acc: %.2f%%]\n" %
              (scores1[0], scores1[1], scores1[2], scores1[3]*100, scores1[4]*100))
        print("Evaluating D2 [loss:  %.4f, bi-loss: %.4f, cat-loss: %.4f, bi-acc: %.2f%%, cat-acc: %.2f%%]\n" %
              (scores2[0], scores2[1], scores2[2], scores2[3]*100, scores2[4]*100))

        return (scores1[0], scores1[3]*100), (scores2[0], scores2[3]*100)
    
    

    def save_images(self, epoch):
        r, c = 5, 4
        noise = np.random.normal(0, 1, (r * int(c/2), 100))
        gen_imgs1 = self.g1.predict(noise)
        gen_imgs2 = self.g2.predict(noise)

        gen_imgs = np.concatenate([gen_imgs1, gen_imgs2])

        """
        -----------------------------------------------------------------------------------------------
         Rescale images from [-1..1] to [0..1] just to display purposes.
        -----------------------------------------------------------------------------------------------
        """
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
                
        """
        -----------------------------------------------------------------------------------------------
          Save generator image for each epoch
        -----------------------------------------------------------------------------------------------
        """ 
        fig.savefig("./COGAN_generators_output/cogan_%d.png" % epoch)
        plt.close()
        
        
        
    def save_model(self):

        def save(model, model_name):
            
            """
            -----------------------------------------------------------------------------------------------
            Save weights
            -----------------------------------------------------------------------------------------------
            """
            model_path = "./COGAN_saved_models/%s.json" % model_name
            weights_path = "./COGAN_saved_models/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        shutil.rmtree('COGAN_saved_models', ignore_errors=True)
        os.makedirs("COGAN_saved_models")
        
        """
        -----------------------------------------------------------------------------------------------
          Save models
        -----------------------------------------------------------------------------------------------
        """ 
        save(self.g1, "COGAN_gan_generator1")
        save(self.g2, "COGAN_gan_generator2")
        save(self.d1, "COGAN_gan_discriminator1")
        save(self.d1, "COGAN_gan_discriminator2")
        save(self.combined, "COGAN_gan_adversarial")
        
        
    
    def plot_training_history(self):
        
        """
        -----------------------------------------------------------------------------------------------
          Plot training history
        -----------------------------------------------------------------------------------------------
        """ 
        fig, axs = plt.subplots(1,4,figsize=(15,5))
        plt.title('Training History')
        
        """
        -----------------------------------------------------------------------------------------------
          Summarize history for G and D1 accuracy
        -----------------------------------------------------------------------------------------------
        """
        axs[0].plot(range(1,len(self.training_history['D1_acc'])+1),self.training_history['D1_acc'])
        #axs[0].plot(range(1,len(self.training_history['G_acc'])+1),self.training_history['G_acc'])
        axs[0].set_title('D1 Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        #axs[0].set_xticks(np.arange(1,len(self.training_history['D1_acc'])+1),len(self.training_history['D1_acc'])/10)
        axs[0].set_yticks([n for n in range(0, 101,10)])
        axs[0].legend(['Discriminator', 'Generator'], loc='best')
        
        
        """
        -----------------------------------------------------------------------------------------------
          Summarize history for G and D2 accuracy
        -----------------------------------------------------------------------------------------------
        """
        axs[1].plot(range(1,len(self.training_history['D2_acc'])+1),self.training_history['D2_acc'])
        #axs[1].plot(range(1,len(self.training_history['G_acc'])+1),self.training_history['G_acc'])
        axs[1].set_title('D2 Accuracy')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_xlabel('Epoch')
        #axs[1].set_xticks(np.arange(1,len(self.training_history['D2_acc'])+1),len(self.training_history['D2_acc'])/10)
        axs[1].set_yticks([n for n in range(0, 101,10)])
        axs[1].legend(['Discriminator', 'Generator'], loc='best')
        
        
        """
        -----------------------------------------------------------------------------------------------
          Summarize history for G and D1 loss
        -----------------------------------------------------------------------------------------------
        """
        axs[2].plot(range(1,len(self.training_history['D1_loss'])+1),self.training_history['D1_loss'])
        #axs[2].plot(range(1,len(self.training_history['G_loss'])+1),self.training_history['G_loss'])
        axs[2].set_title('D1 Loss')
        axs[2].set_ylabel('Loss')
        axs[2].set_xlabel('Epoch')
        #axs[2].set_xticks(np.arange(1,len(self.training_history['G_loss'])+1),len(self.training_history['G_loss'])/10)
        axs[2].legend(['Discriminator', 'Generator'], loc='best')
        
        """
        -----------------------------------------------------------------------------------------------
          Summarize history for G and D2 loss
        -----------------------------------------------------------------------------------------------
        """
        axs[3].plot(range(1,len(self.training_history['D2_loss'])+1),self.training_history['D2_loss'])
        #axs[3].plot(range(1,len(self.training_history['G_loss'])+1),self.training_history['G_loss'])
        axs[3].set_title('D2 Loss')
        axs[3].set_ylabel('Loss')
        axs[3].set_xlabel('Epoch')
        #axs[3].set_xticks(np.arange(1,len(self.training_history['G_loss'])+1),len(self.training_history['G_loss'])/10)
        axs[3].legend(['Discriminator', 'Generator'], loc='best')
        
        """
        -----------------------------------------------------------------------------------------------
         Plot graphs
        -----------------------------------------------------------------------------------------------
        """
        plt.show()
        
        
        
    def predict(self, X_test, y_test):
        
         # Images in domain A and B (rotated)
        X1_test = X_test[:int(X_test.shape[0]/2)]
        X2_test = X_test[int(X_test.shape[0]/2):]       
        X2_test = scipy.ndimage.interpolation.rotate(X2_test, 90, axes=(1, 2))
        Y1_test = y_test[:int(y_test.shape[0]/2)]
        Y2_test = y_test[int(y_test.shape[0]/2):]
        
        """
        -----------------------------------------------------------------------------------------------
        Generating a predictions from the discriminator over the testing dataset
        -----------------------------------------------------------------------------------------------
        """ 
        y1_pred = self.d1.predict(X1_test)
        y2_pred = self.d2.predict(X2_test)
        
        """
        -----------------------------------------------------------------------------------------------
         Formating predictions to remove the one_hot_encoding format
        -----------------------------------------------------------------------------------------------
        """ 
        y1_pred = np.argmax(y1_pred[1][:,:-1], axis=1)
        y2_pred = np.argmax(y2_pred[1][:,:-1], axis=1)
        
        """
        -----------------------------------------------------------------------------------------------
         Calculating and plotting a Classification Report
        -----------------------------------------------------------------------------------------------
        """ 
        D1_acc = (accuracy_score(Y1_test, y1_pred) * 100)
        D2_acc = (accuracy_score(Y2_test, y2_pred) * 100)
        D_acc = (D1_acc + D2_acc)/2
        print ('\nOverall accuracy %f%% \n' % D_acc)
        D1_ps = (average_precision_score(Y1_test, y1_pred) * 100)
        D2_ps = (average_precision_score(Y2_test, y2_pred) * 100)
        #D_ps = (D1_ps + D2_ps)/2
        print ('\nAveP: %f%% \n' % D2_ps)
        class_names = ['Non-nuclei', 'Nuclei'] 
        print("Classification report:\n %s\n"
              % (classification_report(Y2_test, y2_pred, target_names=class_names)))
        
        """
        -----------------------------------------------------------------------------------------------
         Output Confusion matrix
        -----------------------------------------------------------------------------------------------
        """ 
        cm1 = confusion_matrix(Y1_test, y1_pred)
        cm2 = confusion_matrix(Y2_test, y2_pred)
        plt.figure()
        plot_confusion_matrix(cm1, class_names, title='Confusion matrix') 
        plt.figure()
        plot_confusion_matrix(cm2, class_names, title='Confusion matrix') 
        
         

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    -----------------------------------------------------------------------------------------------
     Prints and plots the confusion matrix
    -----------------------------------------------------------------------------------------------
    """
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
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
  
        
def load_TMI_data():
    
    """
    -----------------------------------------------------------------------------------------------
     Load the dataset
    -----------------------------------------------------------------------------------------------
    """ 
    dataset = scipy.io.loadmat('TMI2015/training/training.mat')
    
    """
    -----------------------------------------------------------------------------------------------
     Split into train and test. Values are in range [0..1] as float64
    -----------------------------------------------------------------------------------------------
    """
    X_train = np.transpose(dataset['train_x'], (3, 0, 1, 2))
    y_train = list(dataset['train_y'][0])
    
    X_test = np.transpose(dataset['test_x'], (3, 0, 1, 2))
    y_test = list(dataset['test_y'][0])
    
    """
    -----------------------------------------------------------------------------------------------
     Change shape and range.
    -----------------------------------------------------------------------------------------------
    """
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)
    
    """
    -----------------------------------------------------------------------------------------------
     1-> 0 : Non-nucleus. 2 -> 1: Nucleus
    -----------------------------------------------------------------------------------------------
    """
    y_test -= 1
    y_train -= 1
    
    """
    -----------------------------------------------------------------------------------------------
     Resize to 32x32
    -----------------------------------------------------------------------------------------------
    """
    X_train_resized = np.empty([X_train.shape[0], 32, 32, X_train.shape[3]])
    for i in range(X_train.shape[0]):
        X_train_resized[i] = resize(X_train[i], (32, 32, 3), mode='reflect')

    X_test_resized = np.empty([X_test.shape[0], 32, 32, X_test.shape[3]])
    for i in range(X_test.shape[0]):
        X_test_resized[i] = resize(X_test[i], (32, 32, 3), mode='reflect')
    
    """
    -----------------------------------------------------------------------------------------------
     Normalize images from [0..1] to [-1..1]
    -----------------------------------------------------------------------------------------------
    """
    X_train_resized = 2 * X_train_resized - 1
    X_test_resized = 2 * X_test_resized - 1

    return X_train_resized, y_train, X_test_resized, y_test



if __name__ == '__main__':
    
    Model = tf.keras.Model
    
    """
    -----------------------------------------------------------------------------------------------
     Load Data
    -----------------------------------------------------------------------------------------------
    """
    X_train, y_train, X_test, y_test = load_TMI_data()
    
        
    cogan = COGAN()
    start = time.time()
    # Fit/Train the model
    cogan.train(X_train, y_train, X_test, y_test, epochs=200, batch_size=32, save_interval=5)
    end = time.time()
    print ("\nTraining time: %0.1f minutes \n" % ((end-start) / 60))
    
    #saved the trained model
    cogan.save_model()  
    
    """
    -----------------------------------------------------------------------------------------------
     plot training graph
    -----------------------------------------------------------------------------------------------
    """
    cogan.plot_training_history()
    
    """
    -----------------------------------------------------------------------------------------------
     Evaluate the trained D model w.r.t unseen data (i.e. testing set)
    -----------------------------------------------------------------------------------------------
    """
    cogan.evaluate_discriminator(X_test, y_test)
    cogan.predict(X_test, y_test)
    

