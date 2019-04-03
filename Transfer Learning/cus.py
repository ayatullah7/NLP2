# keras layers
from keras.models import Sequential
from keras.layers import Conv2D, DepthwiseConv2D
from keras.layers import MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization,LeakyReLU
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
# Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# model
classifier = Sequential()

# adding layers
classifier.add(Conv2D(32, kernel_size=3, strides=1, padding="same", kernel_regularizer=regularizers.l2(0.00001), input_shape=(128, 128, 3)))
classifier.add(LeakyReLU())
classifier.add(Conv2D(32, kernel_size=3, strides=1, padding="same", kernel_regularizer=regularizers.l2(0.00001)))
classifier.add(LeakyReLU())
classifier.add(Conv2D(32, kernel_size=3, strides=1, padding="same", kernel_regularizer=regularizers.l2(0.00001)))
classifier.add(LeakyReLU())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.6))

classifier.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
classifier.add(LeakyReLU())
classifier.add(Conv2D(32, kernel_size=3, strides=1, padding="same",))
classifier.add(LeakyReLU())
classifier.add(Conv2D(32, kernel_size=3, strides=1, padding="same",))
classifier.add(LeakyReLU())

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(DepthwiseConv2D( kernel_size=3, strides=1, padding="same",kernel_initializer='he_normal'))
classifier.add(LeakyReLU())
classifier.add(BatchNormalization(axis=3))
classifier.add(Conv2D(32, kernel_size=1, strides=1, padding="same",kernel_initializer='he_normal'))
classifier.add(LeakyReLU())
classifier.add(BatchNormalization(axis=3))

classifier.add(DepthwiseConv2D( kernel_size=3, strides=1, padding="same",kernel_initializer='he_normal'))
classifier.add(LeakyReLU())
classifier.add(BatchNormalization(axis=3))
classifier.add(Conv2D(32, kernel_size=1, strides=1, padding="same",kernel_initializer='he_normal'))
classifier.add(LeakyReLU())
classifier.add(BatchNormalization(axis=3))

classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
classifier.add(LeakyReLU())
classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
classifier.add(LeakyReLU())
classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same",))
classifier.add(LeakyReLU())
classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same",))
classifier.add(LeakyReLU())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.6))

classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
classifier.add(LeakyReLU())
classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
classifier.add(LeakyReLU())
classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same",))
classifier.add(LeakyReLU())
classifier.add(Conv2D(64, kernel_size=3, strides=1, padding="same",))
classifier.add(LeakyReLU())

# classifier.add(GlobalAveragePooling2D())
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dropout(0.4))

classifier.add(Dense(1024, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(units=24, activation='softmax'))

# compiling cnn
classifier.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                    )

classifier.summary()

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.1,
                                    rotation_range=10,
                                    horizontal_flip=True,
                                    height_shift_range=0.1,
                                    width_shift_range=0.1,
                                    fill_mode='nearest',
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    # featurewise_center=True,
                                    # featurewise_std_normalization=True,
                                    )
# train_datagen.mean =  np.array([0.50732507, 0.39511367, 0.39813832], dtype=np.float32).reshape((1,1,3))
# train_datagen.std =  0.05979150389245361

test_datagen = ImageDataGenerator(
                                rescale=1./255,
                                samplewise_center=True,
                                samplewise_std_normalization=True,
                                # featurewise_center=True,
                                # featurewise_std_normalization=True,
                                )
# test_datagen.mean = np.array([0.42023899, 0.30655081, 0.29950314], dtype=np.float32).reshape((1,1,3))
# test_datagen.std = 0.059090827181001726

training_set = train_datagen.flow_from_directory('B',
                                                target_size=(128, 128),
                                                batch_size=170,
                                                class_mode='categorical',
                                                )

test_set = test_datagen.flow_from_directory('A',
                                            target_size=(128, 128),
                                            batch_size=170,
                                            class_mode='categorical',
                                            )

classifier.fit_generator(training_set,
                    steps_per_epoch=156,
                    epochs=30,
                    validation_data=test_set,
                    validation_steps=156,
                    shuffle=True,
                    callbacks=[ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=4, verbose=1, mode='auto')]
                    )


classifier.save('cus.h5')  

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from glob import glob
import time
import seaborn as sns
import itertools
CLASSES = [folder[len('B') + 1:] for folder in glob('B' + '/*')]
CLASSES.sort()



def evaluate_model(generator):
    start_time = time.time()
    evaluations = classifier.evaluate_generator(generator, steps=1)
    for i in range(len(classifier.metrics_names)):
        print("{}: {:.2f}%".format(
            classifier.metrics_names[i], evaluations[i] * 100))
    print('Took {:.0f} seconds to evaluate this set.'.format(
        time.time() - start_time))

    start_time = time.time()
    predictions = classifier.predict_generator(generator, steps=1)
    print('Took {:.0f} seconds to get predictions on this set.'.format(
        time.time() - start_time))

    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes
    print(y_true," = ", y_pred)
    return dict(y_pred=y_pred, y_true=y_true)


def evaluate_validation_dataset():
    predict_datagen = ImageDataGenerator()
    predict_set = predict_datagen.flow_from_directory('E',
                                                target_size=(128, 128),
                                                batch_size=240,
                                                class_mode='categorical',
                                                shuffle=False
                                                )
    
    return evaluate_model(predict_set)

print("run:  =")
CNN_VALIDATION_SET_EVAL = evaluate_validation_dataset()
print(classification_report(**CNN_VALIDATION_SET_EVAL, target_names=CLASSES))