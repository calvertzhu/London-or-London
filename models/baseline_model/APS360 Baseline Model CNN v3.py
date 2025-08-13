import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.preprocessing import image_dataset_from_directory


# This should contain two subfolders, "train" and "val", each of which
# has one folder per class (e.g. "London_ON", "London_UK").
data_dir = 'report_data'

# Create Datasets ----------------------------
train_ds = image_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

val_ds = image_dataset_from_directory(
    os.path.join(data_dir, 'val'),
    validation_split=None,
    subset=None,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),  # Width/height shift
])

# Apply augmentation and rescaling to training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True) / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))  # Rescale validation data too

# Model definition -------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train ----------------------------
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
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