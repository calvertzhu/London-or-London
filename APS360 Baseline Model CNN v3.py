import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# This should contain two subfolders, "train" and "val", each of which
# has one folder per class (e.g. "London_ON", "London_UK").
# If inconvinient - message on discord for hotfix
data_dir = 'path/to/your/data_folder'

# Create Generators ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Model definition -------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train ----------------------------
epochs = 15
history = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,
    epochs=epochs
)

# Plotting ----------------------------
#   a) Loss curves
plt.figure()
plt.plot(history.history['loss'],    label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

#   b) Error‐rate curves (error = 1 - accuracy)
train_error = [1 - a for a in history.history['accuracy']]
val_error   = [1 - a for a in history.history['val_accuracy']]

plt.figure()
plt.plot(train_error, label='train_error')
plt.plot(val_error,   label='val_error')
plt.title('Training vs. Validation Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.legend()

# If running on mac - may need to adjust permissions 
plt.show()  


# If this doesnt work, try changing 'optimizer = 'adam''
# and manually adjust hyperparameters. Or try SGD w/ momentum 
# if too much overfitting from model.