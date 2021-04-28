import math, os, shutil
import numpy as np
import csv
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras import regularizers
from keras.utils import plot_model
from keras.models import load_model
K.set_image_dim_ordering("tf")

def init_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def augmentation(input_dir, output_dir, num, datagen):
    for img_name in os.listdir(input_dir):
        img = load_img(input_dir + img_name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=img_name[:-4], save_format='jpg'):
            i += 1
            if i > num:
                break
        shutil.copyfile(input_dir + img_name, output_dir + img_name)

def train(TRAIN_DIR, VALID_DIR, BATCH_SIZE, LOSS, Optimizer):
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    if num_valid_steps == 0:
        num_valid_steps = 1

    gen = keras.preprocessing.image.ImageDataGenerator()
    val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=(224, 224), class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=(224, 224), class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    model = keras.applications.resnet50.ResNet50()
    history = History()
    classes = list(iter(batches.class_indices))
    model.layers.pop()

    regularizer = regularizers.l2(1e-4)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    last = model.layers[-1].output

    last = Dropout(0.7)(last)

    x = Dense(len(classes), activation="sigmoid")(last)

    finetuned_model = Model(model.input, x)

    finetuned_model.compile(optimizer=Optimizer, loss=LOSS, metrics=['accuracy'])

    plot_model(finetuned_model, to_file='./model.png', show_shapes=True)

    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=20)
    checkpointer = ModelCheckpoint('./Results/' + str(feature_no) + '/resnet50_best.h5', verbose=1, save_best_only=True)

    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.loss = []
            self.acc = []
            self.val_loss = []
            self.val_acc = []

        def on_epoch_end(self, batch, logs={}):
            self.loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))
            self.acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))

    history = LossHistory()

    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=1000,
                                  callbacks=[early_stopping, checkpointer, history], validation_data=val_batches,
                                  validation_steps=num_valid_steps)
    finetuned_model.save('./Results/' + str(feature_no) + '/resnet50_final.h5')

    # save logs
    log_dir = './Results/' + str(feature_no) + '/log.csv'
    if os.path.exists(log_dir):
        os.remove(log_dir)
    log_file = open(log_dir, 'wb')

    writer_log = csv.writer(log_file)

    writer_log.writerow(['Epoch', 'Train_loss', 'Train_acc', 'Val_Loss', 'Val_ACC'])
    for epoch_no in range(len(history.loss)):
        writer_log.writerow([str(epoch_no + 1), history.loss[epoch_no], history.acc[epoch_no],
                         history.val_loss[epoch_no], history.val_acc[epoch_no]])


def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


if __name__ == '__main__':

    # feature_no 1&2 represents lesion B-mode US images
    # feature_no 3 represents color doppler US images
    # feature_no 4&5 represents solid components B-mode US images

    for feature_no in ['1&2','3','4&5']:

        # Data augmentation

        init_path('./Data_aug/' + str(feature_no) + '/train_aug')

        for type in ['B', 'M']: # benign, malignant

            input_dir = './Data_scaled/' + str(feature_no) + '/Train_' + type + '/'
            output_dir = './Data_aug/' + str(feature_no) + '/train_aug/' + type + '/'
            init_path(output_dir)

            datagen = ImageDataGenerator(rotation_range=40, horizontal_flip=True,vertical_flip=False,fill_mode='constant',cval=0)

            augmentation(input_dir, output_dir, 4, datagen)

        # Train

        TRAIN_DIR = './Data_aug/'+str(feature_no)+ '/train_aug/'
        VALID_DIR = './Data_aug/'+str(feature_no)+ '/valid/'
        BATCH_SIZE = 32
        LOSS = 'binary_crossentropy'
        Optimizer = SGD(lr=1e-4)

        train(TRAIN_DIR, VALID_DIR, BATCH_SIZE, LOSS, Optimizer)

        # Predict

        pred_results = {}

        model_path = './Results/' + feature_no + '/resnet50_best.h5'
        model = load_model(model_path)

        test_path = './Data_aug/' + feature_no + '/test/'
        for test_img in os.listdir(test_path):
            patient_no = test_img.split('_')[0]
            pred = predict(test_path + test_img, model)[0][1]

            if patient_no in pred_results.keys():
                if pred > pred_results[patient_no]:
                    pred_results[patient_no]=pred
            else:
                pred_results[patient_no] = pred

        res_dir = './Results/'+feature_no+'/res.csv'

        if os.path.exists(res_dir):
            os.remove(res_dir)
        res_file = open(res_dir, 'wb')

        writer_res = csv.writer(res_file)

        writer_res.writerow(['Name', 'Pred'])
        for key in pred_results:
            writer_res.writerow([key,  np.round(float(pred_results[key]), 4)])
