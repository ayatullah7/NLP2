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
img_width, img_height = 128, 128
batch_size = 64
epochs = 50
nb_validation_steps = 413
nb_train_steps = 414

# model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
model = applications.xception.Xception(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3))

for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
new_l = model.output

new_l = Flatten()(new_l)
new_l = Dense(1024, activation="relu")(new_l)
new_l = Dropout(0.5)(new_l)
new_l = Dense(512, activation="relu")(new_l)
new_l = Dropout(0.5)(new_l)
new_l = Dense(256, activation="relu")(new_l)
final_l = Dense(24, activation="softmax")(new_l)

# creating the final model 
model_final = Model(inputs = model.input, outputs = final_l)

# compile the model 
model_final.compile(
    # optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
    optimizer = 'adam', 
    loss = "categorical_crossentropy",
    metrics=["accuracy"])

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
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=4, verbose=1, mode='auto')

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
