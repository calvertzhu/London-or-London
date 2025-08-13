import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.preprocessing import image_dataset_from_directory


# This should contain two subfolders, "train" and "val", each of which
# has one folder per class (i.e. "london_on", "london_uk").
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'report_data')

# Create datasets ----------------------------
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

train_dataset = image_dataset_from_directory(
    os.path.join(data_dir, 'train'),
    validation_split=None,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_dataset = image_dataset_from_directory(
    os.path.join(data_dir, 'val'),
    validation_split=None,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Model definition -------------------------------------------------
model = models.Sequential([
    layers.Rescaling(1./255),  # Normalize pixel values
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')  # Use sigmoid for binary classif.
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train ----------------------------
num_epochs = 15
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=num_epochs
)

# Plot Curves ----------------------------
#   a) Loss curves
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# b) Accuracy curves 
train_error = [a for a in history.history['accuracy']]    
val_error = [a for a in history.history['val_accuracy']]

plt.figure()
plt.plot(train_error, label='train_acc')
plt.plot(val_error, label='val_acc')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

#   c) Error‐rate curves (error = 1 - accuracy)
train_error = [1 - a for a in history.history['accuracy']]
val_error = [1 - a for a in history.history['val_accuracy']]

plt.figure()
plt.plot(train_error, label='train_error')
plt.plot(val_error, label='val_error')
plt.title('Training vs. Validation Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.legend()

plt.show()