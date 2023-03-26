import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the preprocessed and normalized data
train_data = ImageDataGenerator().flow_from_directory(
    'data_preprocessed_cropped_normalized/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_data = ImageDataGenerator().flow_from_directory(
    'data_preprocessed_cropped_normalized/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data,
          epochs=10,
          validation_data=test_data)

# Save the trained model
model.save('trained_models/cnn_model.h5')
