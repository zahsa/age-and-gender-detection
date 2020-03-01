import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
# limit GPU memory usage
# config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

import argparse
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
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

from functools import partial
from keras.regularizers import l2
from datetime import datetime

import pickle

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--input", "-i", type=str, default='/home/zahra/Documents/PyCharmProjects/deepage/age_range_narrow/',
    #                     help="path to input database mat file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=100,
                        help="number of epochs")
    # parser.add_argument("--depth", type=int, default=16,
    #                     help="depth of network (should be 10, 16, 22, 28, ...)")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="validation split ratio")
    parser.add_argument("--img_size", type=int, default=224,
                        help='net input size')
    # parser.add_argument("--val_path",type=str,default='./data/imdb_val.mat',
    #                     help='path to validation data mat file ')
    # parser.add_argument("--pretrained_fil",type=str,default=None,
    #                     help='before pretrained_file for net')
    parser.add_argument("--lr", type=float, default=1e-5,
                        help='the net training learningrate')
    parser.add_argument("--gpus", type=int, default=1,
                        help='how many gpus for trainning')
    parser.add_argument("--db", type=str, default='/home/age_range_narrow/',
                        help="training on which dataset")
    # parser.add_argument("--appa_dir", type=str, required=True,
    #                     help="path to the APPA-REAL dataset")
    # parser.add_argument("--utk_dir", type=str, default=None,
    #                     help="path to the UTK face dataset")
    parser.add_argument("--output_dir", type=str, default="/home/agebest/",
                                                help="checkpoint dir")
    parser.add_argument("--opt", type=str, default="adam",
                        help="optimizer name; 'sgd', 'adam', 'rmsprop")
    parser.add_argument("--model_name", type=str, default="ResNet50",
                        help="model name: 'ResNet50' , 'InceptionResNetV2' , 'vgg' , 'senet' ")
    args = parser.parse_args()
    return args


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008

def lr_schedule(epoch_idx):
    base_lr=0.01,
    decay=0.9
    epoch_idx = int(epoch_idx)
    return base_lr * decay**(epoch_idx)

def lr_scheduler(epoch, mode='progressive_drops'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.00001
        elif epoch > 0.75 * epochs:
            lr = 0.0001
        elif epoch > 0.5 * epochs:
            lr = 0.001
        else:
            lr = 0.01

    print('lr: %f' % lr)
    return lr

def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    elif opt_name == "rmsporp":
        return RMSprop(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    # appa_dir = args.appa_dir
    # utk_dir = args.utk_dir
    model_name = args.model_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    opt_name = args.opt
    image_size = args.img_size
    train_dir = args.db
    import ipdb
    ipdb.set_trace()
    # if model_name == "ResNet50":
    #     image_size = 224
    # elif model_name == "InceptionResNetV2":
    #     image_size = 299
    checkpoint_path = args.output_dir + "_seface_agerange" + str(fc_num) + '_' + str(unfrozen_num) + '_' + str(train_batchsize) + '_' + 'lr' + str(lr) + datetime.now().strftime("%Y%m%d-%H%M%S") + '.ckpt'

    # train_gen = FaceGenerator(appa_dir, utk_dir=utk_dir, batch_size=batch_size, image_size=image_size)
    # val_gen = ValGenerator(appa_dir, batch_size=batch_size, image_size=image_size)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=nb_epochs,
        class_mode='categorical',
        shuffle=True,
        seed=2,
        subset='training')

    validation_generator = valid_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=nb_epochs,
        class_mode='categorical',
        shuffle=True,
        seed=2,
        subset='validation')

    model = get_model(model_name=model_name)
    opt = get_optimizer(opt_name, lr)
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    model.summary()
    # output_dir = Path(__file__).resolve().parent.joinpath(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
    #                                                  # save_weights_only=True,
    #                                                  save_best_only=True,
    #                                                  # Save weights, every 5-epochs.
    #                                                  # period=5,
    #                                                  monitor='val_acc',
    #                                                  verbose=1,
    #                                                  mode='max')

    # callbacks = [
    #     keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    # ]
    cp_callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr=lr)),
                 ModelCheckpoint(checkpoint_path + "/weights.{epoch:03d}-{val_loss:.3f}-{val_age_mae:.3f}.hdf5",
                                 monitor="val_age_mae",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]


    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=nb_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        # callbacks=[tensorboard_callback],
        callbacks=[cp_callback, tensorboard_callback],
        verbose=1,
        workers = 4)
    print("Average validation loss: ", np.average(history.history['loss']))

    np.savez(str(output_dir.joinpath("history.npz")), history=hist.history)

 
    fnames = validation_generator.filenames
    # valid_imglabels = validation_generator.labels
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