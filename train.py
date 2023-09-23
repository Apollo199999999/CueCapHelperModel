# Library imports
import tensorflow as tf
from keras import layers
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SHAPE = (75, 75)
BATCH_SIZE = 96

train_dir = "./FERPlus/train"
test_dir = "./FERPlus/test/"

img_datagen = ImageDataGenerator(rescale=1 / 255.,
                                 rotation_range=0.1,
                                 zoom_range=0.1,
                                 #validation_split=0.2
                                 )

train_data = img_datagen.flow_from_directory(train_dir,
                                             target_size=IMAGE_SHAPE,
                                             batch_size=BATCH_SIZE,
                                             class_mode="categorical",
                                             # subset="training",
                                             shuffle=True)

test_data = img_datagen.flow_from_directory(test_dir,
                                            target_size=IMAGE_SHAPE,
                                            batch_size=BATCH_SIZE,
                                            class_mode="categorical",
                                            # subset="validation",
                                            shuffle=True)

# Print class names
class_names = train_data.class_indices
print(class_names)

# Construct CNN structure
model = Sequential()
model.add(keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(75, 75, 3)))
model.add(keras.layers.Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))


model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy"])

model.summary()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint/",
                                                         save_weights_only=False,
                                                         save_best_only=True,
                                                         save_freq="epoch",
                                                         verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                              patience=2, min_lr=0.00001)

# Fit the model
model.fit(train_data, epochs=30, callbacks=[reduce_lr, checkpoint_callback], validation_data=test_data)

model.save("ferModel")
