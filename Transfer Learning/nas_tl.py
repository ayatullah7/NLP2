from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, MaxPooling2D, Conv2D, BatchNormalization
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers

train_data_dir = "B"
validation_data_dir = "A"
img_width, img_height = 224,224
batch_size = 64
epochs = 30
nb_validation_steps = 413
nb_train_steps = 414

model = applications.nasnet.NASNetMobile(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
# model = applications.densenet.DenseNet121(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
# model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
# model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
# model = applications.xception.Xception(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3))
# model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3))
# model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3))
# model = applications.MobileNetV2(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
# model = applications.MobileNetV2(weights = "imagenet", include_top=True, input_shape = (img_width, img_height, 3), alpha=1.0, depth_multiplier=1, input_tensor=None, pooling=None, classes=1000)

for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
new_l = model.output
# new_l = Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu")(new_l)
# new_l = Dropout(0.2)(new_l)
# new_l = Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu")(new_l)

# new_l = Flatten()(new_l)
new_l = GlobalAveragePooling2D()(new_l)

new_l = Dense(512, activation="relu")(new_l)
new_l = Dropout(0.5)(new_l)
new_l = Dense(256, activation="relu")(new_l)
new_l = Dropout(0.5)(new_l)
new_l = Dense(256, activation="relu")(new_l)
new_l = Dropout(0.5)(new_l)
new_l = Dense(128, activation="relu")(new_l)


new_l = Dense(24, activation="softmax")(new_l)

# creating the final model 
model_final = Model(inputs = model.input, outputs = new_l)

# compile the model 
model_final.compile(
    # optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
    optimizer = 'adam', 
    loss = "categorical_crossentropy",
    metrics=["accuracy"])
    
# for i, layer in enumerate(model_final.layers):
#     print(i, layer.name)
model_final.summary()

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range=0.1,
    rotation_range=20,
    samplewise_center=True,
    samplewise_std_normalization=True)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
    samplewise_center=True,
    samplewise_std_normalization=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("model_5.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=2, verbose=1, mode='auto')

# Train the model 
model_final.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs = 15,
    validation_data = validation_generator,
    validation_steps=nb_validation_steps,
    shuffle=False,
    callbacks=[reduce_lr]
    )

    
# /////////////////////////////////////////////////////////////////////
# for layer in model_final.layers[:44]:
#     layer.trainable = False
for layer in model_final.layers:
    layer.trainable = True
# compile the model 
model_final.compile(
    # optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
    optimizer = 'adam', 
    loss = "categorical_crossentropy",
    metrics=["accuracy"])
    
# for i, layer in enumerate(model_final.layers):
#     print(i, layer.name)
model_final.summary()

# Train the model 
model_final.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps=nb_validation_steps,
    shuffle=False,
    callbacks=[reduce_lr]
    )
model_final.save('nas.h5')  

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from glob import glob
import time
import seaborn as sns
import itertools
CLASSES = [folder[len(train_data_dir) + 1:] for folder in glob(train_data_dir + '/*')]
CLASSES.sort()



def evaluate_model(generator):
    start_time = time.time()
    evaluations = model_final.evaluate_generator(generator, steps=1)
    for i in range(len(model_final.metrics_names)):
        print("{}: {:.2f}%".format(
            model_final.metrics_names[i], evaluations[i] * 100))
    print('Took {:.0f} seconds to evaluate this set.'.format(
        time.time() - start_time))

    start_time = time.time()
    predictions = model_final.predict_generator(generator, steps=1)
    print('Took {:.0f} seconds to get predictions on this set.'.format(
        time.time() - start_time))

    y_pred = np.argmax(predictions, axis=1)
    y_true = generator.classes
    print(y_true," = ", y_pred)
    return dict(y_pred=y_pred, y_true=y_true)


def evaluate_validation_dataset():
    predict_datagen = ImageDataGenerator()
    predict_set = predict_datagen.flow_from_directory('E',
                                                target_size=(224, 224),
                                                batch_size=240,
                                                class_mode='categorical',
                                                shuffle=False
                                                )
    
    return evaluate_model(predict_set)

print("run:  =")
CNN_VALIDATION_SET_EVAL = evaluate_validation_dataset()
print(classification_report(**CNN_VALIDATION_SET_EVAL, target_names=CLASSES))