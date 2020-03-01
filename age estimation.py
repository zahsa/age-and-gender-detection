import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# limit GPU memory usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import keras
import keras.backend as K
from keras import layers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras_vggface.utils import preprocess_input

from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

# Optimizers
from keras.optimizers import Adam  # Adam optimizer https://arxiv.org/abs/1412.6980
from keras.optimizers import SGD  # Stochastic gradient descent optimizer

from keras.applications.resnet50 import decode_predictions  # ResNet-specific routines for extracting predictions
from keras.preprocessing.image import load_img

from keras import models
from keras import layers
from keras import optimizers

import tensorflow as tf
import math
from keras.callbacks import TensorBoard

from keras.callbacks import ModelCheckpoint

from keras.callbacks import CSVLogger

from functools import partial
from keras.regularizers import l2
from datetime import datetime
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

import pickle


def main():
    with tf.compat.v1.Session(config=config) as sess:
        logdir = "/home/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,
                                                           # histogram_freq=0,
                                                           # batch_size=32,
                                                           # write_graph=True,
                                                           # write_grads=False,
                                                           # write_images=False,
                                                           # embeddings_freq=0,
                                                           # embeddings_layer_names=None,
                                                           # embeddings_metadata=None,
                                                           # embeddings_data=None,
                                                           # update_freq='epoch'
                                                           )
        fc_num = 256
        unfrozen_num = 5
        train_batchsize = 32
        class_num = 14
        epoch_num = 100
        lr = 1e-6
        # checkpoints
        checkpoint_path = '/home/agebest' + "_vggface_agerange_narrow" + str(
            fc_num) + '_' + str(unfrozen_num) + '_' + str(train_batchsize) + '_' + 'lr' + str(
            lr) + datetime.now().strftime("%Y%m%d-%H%M%S") + '.ckpt'

        # weights.{epoch:02d}-{val_loss:.2f}.hdf5

        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         # save_weights_only=True,
                                                         save_best_only=True,
                                                         # Save weights, every 5-epochs.
                                                         # period=5,
                                                         monitor='val_acc',
                                                         verbose=1,
  
        image_size = 224

        import keras_vggface
        from keras_vggface.vggface import VGGFace

        model_arch = VGGFace(weights='vggface', include_top=False, input_shape=(image_size, image_size, 3))



        for layer in model_arch.layers[:-unfrozen_num]:
            layer.trainable = False

       

        model = models.Sequential()
        model.add(model_arch)
        model.summary()
        model._ckpt_saved_epoch = None
        # Add new layers
        model.add(layers.Flatten())
        model.add(layers.Dense(fc_num, activation='relu'))
        model.add(layers.Dropout(0.5))
        # model.add(Dense(fc_num, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(layers.Dense(class_num, activation='softmax'))
        model.summary()

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            brightness_range=[0.5, 1.5],
            # rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        valid_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
        )

      
        train_dir = '/home/age_range_narrow/'

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='categorical',
            shuffle=True,
            seed=2,
            subset='training')

        validation_generator = valid_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='categorical',
            shuffle=True,
            seed=2,
            subset='validation')

        # Compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=lr),
                      #               optimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      metrics=['acc'])


        history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples / train_generator.batch_size,
            epochs=epoch_num,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples / validation_generator.batch_size,
            # callbacks=[tensorboard_callback],
            callbacks=[cp_callback, tensorboard_callback],
            verbose=1
        )
        print("Average validation loss: ", np.average(history.history['loss']))

      
        plotcurves(history, validation_generator, checkpoint_path)


def plotcurves(history, validation_generator, checkpoint_path):
    % matplotlib
    inline
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

  
    fnames = validation_generator.filenames
    ground_truth = validation_generator.classes

    best_model = load_model(checkpoint_path)

    true_y = []
    pred_y = []
    error = 0
    for i in range(len(fnames)):
        y = ground_truth[i]
        true_y.append(y)
        title = 'Original label:{}'.format(
            fnames[i])

        img = load_img('{}/{}'.format(train_dir, fnames[i]), target_size=(image_size, image_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = best_model.predict(x)
        predlabel = np.argmax(pred, axis=1)
        pred_y.append(predlabel)
        if predlabel != y:
            error = error + 1
   

    print('acc rate', (len(pred_y) - error) / len(pred_y))
    print('error rate', error / len(pred_y))

    import sklearn.metrics as metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(ground_truth, pred_y))


if __name__ == '__main__':
    main()